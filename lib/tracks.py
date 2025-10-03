import asyncio
import fractions
import logging
import time
from collections import deque

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class VideoProcessingTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        fps: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        # Dynamic FPS calculation variables
        self.fps = fps
        self.frame_ptime = 1.0 / fps
        self.frame_timestamps = deque(
            maxlen=30
        )  # Keep last 30 frame timestamps for averaging
        self.last_frame_time = None
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.last_fps_update = time.time()
        self.min_fps = 1.0  # Minimum FPS to prevent division by zero
        self.max_fps = 60.0  # Maximum FPS cap

        self.frame_processor = None
        self.input_task = None
        self.input_task_running = False

    def _calculate_dynamic_fps(self):
        """Calculate FPS based on recent frame timestamps"""
        if len(self.frame_timestamps) < 2:
            logger.debug(
                f"Not enough frame timestamps ({len(self.frame_timestamps)}), returning current FPS: {self.fps}"
            )
            return self.fps  # Not enough data, return current FPS

        # Calculate average time between frames
        time_diffs = []
        for i in range(1, len(self.frame_timestamps)):
            time_diff = self.frame_timestamps[i] - self.frame_timestamps[i - 1]
            if time_diff > 0:  # Only consider positive time differences
                time_diffs.append(time_diff)

        if not time_diffs:
            logger.debug("No valid time differences found, returning current FPS")
            return self.fps  # No valid time differences

        avg_frame_interval = sum(time_diffs) / len(time_diffs)
        calculated_fps = 1.0 / avg_frame_interval

        # Clamp FPS to reasonable bounds
        calculated_fps = max(self.min_fps, min(self.max_fps, calculated_fps))

        logger.debug(
            f"Calculated FPS: {calculated_fps:.2f} (from {len(time_diffs)} intervals, avg interval: {avg_frame_interval:.4f}s)"
        )
        return calculated_fps

    def _update_fps_if_needed(self):
        """Update FPS and frame_ptime if enough time has passed"""
        current_time = time.time()
        if current_time - self.last_fps_update >= self.fps_update_interval:
            new_fps = self._calculate_dynamic_fps()
            logger.debug(
                f"FPS update check: current={self.fps:.2f}, calculated={new_fps:.2f}, diff={abs(new_fps - self.fps):.2f}"
            )
            if abs(new_fps - self.fps) > 0.1:  # Only update if significant change
                old_fps = self.fps
                self.fps = new_fps
                self.frame_ptime = 1.0 / self.fps
                logger.debug(f"Dynamic FPS updated: {old_fps:.1f} -> {self.fps:.1f}")
            else:
                logger.debug(
                    f"FPS change too small ({abs(new_fps - self.fps):.2f}), not updating"
                )
            self.last_fps_update = current_time

    def get_current_fps(self) -> float:
        """Get the current dynamically calculated FPS"""
        return self.fps

    async def input_loop(self):
        """Background loop that continuously feeds frames to the processor"""
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()

                # Store raw VideoFrame for later processing
                self.frame_processor.put(input_frame)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Stop the input loop on connection errors to avoid spam
                logger.error(f"Error in input loop, stopping: {e}")
                self.input_task_running = False
                break

    # Copied from https://github.com/livepeer/fastworld/blob/e649ef788cd33d78af6d8e1da915cd933761535e/backend/track.py#L267
    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Override to control frame rate"""
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "timestamp"):
            self.timestamp += int(self.frame_ptime * VIDEO_CLOCK_RATE)
            wait = self.start + (self.timestamp / VIDEO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self.start = time.time()
            self.timestamp = 0

        return self.timestamp, VIDEO_TIME_BASE

    def initialize_output_processing(self):
        if not self.frame_processor:
            self.frame_processor = FrameProcessor(
                pipeline_manager=self.pipeline_manager,
                initial_parameters=self.initial_parameters,
                notification_callback=self.notification_callback,
            )
            self.frame_processor.start()

    def initialize_input_processing(self, track: MediaStreamTrack):
        self.track = track
        self.input_task_running = True
        self.input_task = asyncio.create_task(self.input_loop())

    async def recv(self) -> VideoFrame:
        """Return the next available processed frame"""
        # Lazy initialization on first call
        self.initialize_output_processing()
        while self.input_task_running:
            try:
                frame_tensor = self.frame_processor.get()
                if frame_tensor is not None:
                    frame = VideoFrame.from_ndarray(
                        frame_tensor.numpy(), format="rgb24"
                    )

                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    # Track frame timing for dynamic FPS calculation (output frames)
                    current_time = time.time()
                    self.frame_timestamps.append(current_time)

                    # Update FPS dynamically based on recent output frame rate
                    self._update_fps_if_needed()

                    return frame

                # No frame available, wait a bit before trying again
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                raise

        raise Exception("Track stopped")

    async def stop(self):
        self.input_task_running = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        if self.frame_processor is not None:
            self.frame_processor.stop()

        await super().stop()
