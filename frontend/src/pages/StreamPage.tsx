import { useState, useEffect, useRef } from "react";
import { Header } from "../components/Header";
import { InputAndControlsPanel } from "../components/InputAndControlsPanel";
import { VideoOutput } from "../components/VideoOutput";
import { SettingsPanel } from "../components/SettingsPanel";
import { PromptInputWithTimeline } from "../components/PromptInputWithTimeline";
import type { TimelinePrompt } from "../components/PromptTimeline";
import { StatusBar } from "../components/StatusBar";
import { useWebRTC } from "../hooks/useWebRTC";
import { useVideoSource } from "../hooks/useVideoSource";
import { useWebRTCStats } from "../hooks/useWebRTCStats";
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import { PIPELINES } from "../data/pipelines";
import { getDefaultDenoisingSteps, getDefaultResolution } from "../lib/utils";
import type { PipelineId } from "../types";
import type { PromptItem } from "../lib/api";

export function StreamPage() {
  // Use the stream state hook for settings management
  const { settings, updateSettings } = useStreamState();

  // Prompt state
  const [promptItems, setPromptItems] = useState<PromptItem[]>([
    { text: PIPELINES[settings.pipelineId]?.defaultPrompt || "", weight: 100 },
  ]);
  const [interpolationMethod, setInterpolationMethod] = useState<
    "linear" | "slerp"
  >("linear");

  // Track when we need to reinitialize video source
  const [shouldReinitializeVideo, setShouldReinitializeVideo] = useState(false);

  const [isLive, setIsLive] = useState(false);
  const [isTimelineCollapsed, setIsTimelineCollapsed] = useState(false);
  const [selectedTimelinePrompt, setSelectedTimelinePrompt] =
    useState<TimelinePrompt | null>(null);

  // Timeline state for left panel
  const [timelinePrompts, setTimelinePrompts] = useState<TimelinePrompt[]>([]);
  const [timelineCurrentTime, setTimelineCurrentTime] = useState(0);
  const [isTimelinePlaying, setIsTimelinePlaying] = useState(false);

  // External control of timeline selection
  const [externalSelectedPromptId, setExternalSelectedPromptId] = useState<
    string | null
  >(null);

  // Ref to access timeline functions
  const timelineRef = useRef<{
    getCurrentTimelinePrompt: () => string;
    submitLivePrompt: (prompts: PromptItem[]) => void;
    updatePrompt: (prompt: TimelinePrompt) => void;
    clearTimeline: () => void;
    resetPlayhead: () => void;
    resetTimelineCompletely: () => void;
    getPrompts: () => TimelinePrompt[];
    getCurrentTime: () => number;
    getIsPlaying: () => boolean;
  }>(null);

  // Pipeline management
  const {
    isLoading: isPipelineLoading,
    error: pipelineError,
    loadPipeline,
  } = usePipeline();

  // WebRTC for streaming
  const {
    remoteStream,
    isStreaming,
    isConnecting,
    peerConnectionRef,
    startStream,
    stopStream,
    updateVideoTrack,
    sendParameterUpdate,
  } = useWebRTC();

  // Get WebRTC stats for FPS
  const webrtcStats = useWebRTCStats({
    peerConnectionRef,
    isStreaming,
  });

  // Video source for preview (camera or video)
  const {
    localStream,
    isInitializing,
    error: videoSourceError,
    mode,
    videoResolution,
    switchMode,
    handleVideoFileUpload,
  } = useVideoSource({
    onStreamUpdate: updateVideoTrack,
    onStopStream: stopStream,
    shouldReinitialize: shouldReinitializeVideo,
    enabled: PIPELINES[settings.pipelineId]?.category === "video-input",
  });

  const handlePromptsSubmit = (prompts: PromptItem[]) => {
    setPromptItems(prompts);
  };

  const handlePipelineIdChange = (pipelineId: PipelineId) => {
    // Stop the stream if it's currently running
    if (isStreaming) {
      stopStream();
    }

    // Check if we're switching from no-video-input to video-input pipeline
    const currentPipelineCategory = PIPELINES[settings.pipelineId]?.category;
    const newPipelineCategory = PIPELINES[pipelineId]?.category;

    if (
      currentPipelineCategory === "no-video-input" &&
      newPipelineCategory === "video-input"
    ) {
      // Trigger video source reinitialization
      // Otherwise the camera or video file is not visible while switching the pipeline types
      setShouldReinitializeVideo(true);
      // Reset the flag after a short delay to allow the effect to trigger
      setTimeout(() => setShouldReinitializeVideo(false), 100);
    }

    // Update the prompt to the new pipeline's default
    const newDefaultPrompt = PIPELINES[pipelineId]?.defaultPrompt || "";
    setPromptItems([{ text: newDefaultPrompt, weight: 100 }]);

    // Reset timeline completely but preserve collapse state
    if (timelineRef.current) {
      timelineRef.current.resetTimelineCompletely();
    }

    // Reset selected timeline prompt to exit Edit mode and return to Append mode
    setSelectedTimelinePrompt(null);
    setExternalSelectedPromptId(null);

    // Update denoising steps and resolution based on pipeline
    const newDenoisingSteps = getDefaultDenoisingSteps(pipelineId);
    const newResolution = getDefaultResolution(pipelineId);

    // Update the pipeline in settings
    updateSettings({
      pipelineId,
      denoisingSteps: newDenoisingSteps,
      resolution: newResolution,
    });
  };

  const handleResolutionChange = (resolution: {
    height: number;
    width: number;
  }) => {
    updateSettings({ resolution });
  };

  const handleSeedChange = (seed: number) => {
    updateSettings({ seed });
  };

  const handleDenoisingStepsChange = (denoisingSteps: number[]) => {
    updateSettings({ denoisingSteps });
    // Send denoising steps update to backend
    sendParameterUpdate({
      denoising_step_list: denoisingSteps,
    });
  };

  const handleNoiseScaleChange = (noiseScale: number) => {
    updateSettings({ noiseScale });
    // Send noise scale update to backend
    sendParameterUpdate({
      noise_scale: noiseScale,
    });
  };

  const handleNoiseControllerChange = (enabled: boolean) => {
    updateSettings({ noiseController: enabled });
    // Send noise controller update to backend
    sendParameterUpdate({
      noise_controller: enabled,
    });
  };

  const handleManageCacheChange = (enabled: boolean) => {
    updateSettings({ manageCache: enabled });
    // Send manage cache update to backend
    sendParameterUpdate({
      manage_cache: enabled,
    });
  };

  const handleResetCache = () => {
    // Send reset cache command to backend
    sendParameterUpdate({
      reset_cache: true,
    });
  };

  const handleLivePromptSubmit = (prompts: PromptItem[]) => {
    // Use the timeline ref to submit the prompt
    if (timelineRef.current) {
      timelineRef.current.submitLivePrompt(prompts);
    }

    // Also send the updated parameters to the backend immediately
    // Preserve the full blend while live
    sendParameterUpdate({
      prompts,
      prompt_interpolation_method: interpolationMethod,
      denoising_step_list: settings.denoisingSteps || [700, 500],
    });
  };

  const handleTimelinePromptEdit = (prompt: TimelinePrompt | null) => {
    setSelectedTimelinePrompt(prompt);
    // Sync external selection state
    setExternalSelectedPromptId(prompt?.id || null);
  };

  const handleTimelinePromptUpdate = (prompt: TimelinePrompt) => {
    setSelectedTimelinePrompt(prompt);

    // Update the prompt in the timeline
    if (timelineRef.current) {
      timelineRef.current.updatePrompt(prompt);
    }
  };

  // Event-driven timeline state updates for left panel
  const handleTimelinePromptsChange = (prompts: TimelinePrompt[]) => {
    setTimelinePrompts(prompts);
  };

  const handleTimelineCurrentTimeChange = (currentTime: number) => {
    setTimelineCurrentTime(currentTime);
  };

  const handleTimelinePlayingChange = (isPlaying: boolean) => {
    setIsTimelinePlaying(isPlaying);
  };

  // Handle ESC key to exit Edit mode and return to Append mode
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && selectedTimelinePrompt) {
        setSelectedTimelinePrompt(null);
        setExternalSelectedPromptId(null);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedTimelinePrompt]);

  const handlePlayPauseToggle = () => {
    const newPausedState = !settings.paused;
    updateSettings({ paused: newPausedState });
    sendParameterUpdate({
      paused: newPausedState,
    });

    // Deselect any selected prompt when video starts playing
    if (!newPausedState && selectedTimelinePrompt) {
      setSelectedTimelinePrompt(null);
      setExternalSelectedPromptId(null); // Also clear external selection
    }
  };

  // Ref to access the timeline's play/pause handler
  const timelinePlayPauseRef = useRef<(() => Promise<void>) | null>(null);
  // Sync resolution with videoResolution when video source changes
  // Only sync for video-input pipelines
  useEffect(() => {
    const pipelineCategory = PIPELINES[settings.pipelineId]?.category;
    const isVideoInputPipeline = pipelineCategory === "video-input";

    if (videoResolution && !isStreaming && isVideoInputPipeline) {
      updateSettings({
        resolution: {
          height: videoResolution.height,
          width: videoResolution.width,
        },
      });
    }
  }, [videoResolution, isStreaming, settings.pipelineId, updateSettings]);

  const handleStartStream = async () => {
    if (isStreaming) {
      stopStream();
      return;
    }

    try {
      // Always load pipeline with current parameters - backend will handle the rest
      console.log(`Loading ${settings.pipelineId} pipeline...`);

      // Prepare load parameters based on pipeline type
      let loadParams = null;

      // Use settings.resolution if available, otherwise fall back to videoResolution
      const resolution = settings.resolution || videoResolution;

      if (settings.pipelineId === "streamdiffusionv2" && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
          seed: settings.seed ?? 42,
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}, seed: ${loadParams.seed}`
        );
      } else if (settings.pipelineId === "passthrough" && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}`
        );
      } else if ((settings.pipelineId === "longlive" || settings.pipelineId === "mycustom") && resolution) {
        loadParams = {
          height: resolution.height,
          width: resolution.width,
          seed: settings.seed ?? 42,
        };
        console.log(
          `Loading with resolution: ${resolution.width}x${resolution.height}, seed: ${loadParams.seed}`
        );
      }

      const loadSuccess = await loadPipeline(
        settings.pipelineId,
        loadParams || undefined
      );
      if (!loadSuccess) {
        console.error("Failed to load pipeline, cannot start stream");
        return;
      }

      // Check if this pipeline needs video input
      const pipelineCategory = PIPELINES[settings.pipelineId]?.category;
      const needsVideoInput = pipelineCategory === "video-input";

      // Only send video stream for pipelines that need video input
      const streamToSend = needsVideoInput
        ? localStream || undefined
        : undefined;

      if (needsVideoInput && !localStream) {
        console.error("Video input required but no local stream available");
        return;
      }

      // Build initial parameters based on pipeline type
      const initialParameters: {
        prompts?: PromptItem[];
        prompt_interpolation_method?: "linear" | "slerp";
        denoising_step_list?: number[];
        noise_scale?: number;
        noise_controller?: boolean;
        manage_cache?: boolean;
      } = {};

      // Common parameters for pipelines that support prompts
      if (
        settings.pipelineId !== "passthrough" &&
        settings.pipelineId !== "vod"
      ) {
        initialParameters.prompts = promptItems;
        initialParameters.prompt_interpolation_method = interpolationMethod;
        initialParameters.manage_cache = settings.manageCache ?? true;
        initialParameters.denoising_step_list = settings.denoisingSteps || [
          700, 500,
        ];
      }

      // StreamDiffusionV2-specific parameters
      if (settings.pipelineId === "streamdiffusionv2") {
        initialParameters.noise_scale = settings.noiseScale ?? 0.7;
        initialParameters.noise_controller = settings.noiseController ?? true;
      }

      // Reset paused state when starting a fresh stream
      updateSettings({ paused: false });

      // Pipeline is loaded, now start WebRTC stream
      startStream(initialParameters, streamToSend);
    } catch (error) {
      console.error("Error during stream start:", error);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <Header />

      {/* Main Content Area */}
      <div className="flex-1 flex gap-4 px-4 pb-4 min-h-0 overflow-hidden">
        {/* Left Panel - Input & Controls */}
        <div className="w-1/5">
          <InputAndControlsPanel
            className="h-full"
            localStream={localStream}
            isInitializing={isInitializing}
            error={videoSourceError}
            mode={mode}
            onModeChange={switchMode}
            isStreaming={isStreaming}
            isConnecting={isConnecting}
            isPipelineLoading={isPipelineLoading}
            canStartStream={
              PIPELINES[settings.pipelineId]?.category === "no-video-input"
                ? !isInitializing
                : !!localStream && !isInitializing
            }
            onStartStream={handleStartStream}
            onStopStream={stopStream}
            onVideoFileUpload={handleVideoFileUpload}
            pipelineId={settings.pipelineId}
            prompts={promptItems}
            onPromptsChange={setPromptItems}
            onPromptsSubmit={handlePromptsSubmit}
            interpolationMethod={interpolationMethod}
            onInterpolationMethodChange={setInterpolationMethod}
            isLive={isLive}
            onLivePromptSubmit={handleLivePromptSubmit}
            selectedTimelinePrompt={selectedTimelinePrompt}
            onTimelinePromptUpdate={handleTimelinePromptUpdate}
            isVideoPaused={settings.paused}
            isTimelinePlaying={isTimelinePlaying}
            currentTime={timelineCurrentTime}
            timelinePrompts={timelinePrompts}
          />
        </div>

        {/* Center Panel - Video Output + Timeline */}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Video area - takes remaining space but can shrink */}
          <div className="flex-1 min-h-0">
            <VideoOutput
              className="h-full"
              remoteStream={remoteStream}
              isPipelineLoading={isPipelineLoading}
              isConnecting={isConnecting}
              pipelineError={pipelineError}
              isPlaying={!settings.paused}
              onPlayPauseToggle={() => {
                // Use timeline's play/pause handler instead of direct video toggle
                if (timelinePlayPauseRef.current) {
                  timelinePlayPauseRef.current();
                }
              }}
              onStartStream={() => {
                // Use timeline's play/pause handler to start stream
                if (timelinePlayPauseRef.current) {
                  timelinePlayPauseRef.current();
                }
              }}
            />
          </div>
          {/* Timeline area - compact, always visible */}
          <div className="flex-shrink-0 mt-2">
            <PromptInputWithTimeline
              currentPrompt={promptItems[0]?.text || ""}
              currentPromptItems={promptItems}
              onPromptSubmit={text => {
                // Update the left panel's prompt state to reflect current timeline prompt
                setPromptItems([{ text, weight: 100 }]);

                // Send to backend
                sendParameterUpdate({
                  prompts: [{ text, weight: 100 }],
                  prompt_interpolation_method: interpolationMethod,
                  denoising_step_list: settings.denoisingSteps || [700, 500],
                });
              }}
              onPromptItemsSubmit={prompts => {
                // Update the left panel's prompt state to reflect current timeline prompt blend
                setPromptItems(prompts);

                // Send to backend with full blend data
                sendParameterUpdate({
                  prompts,
                  prompt_interpolation_method: interpolationMethod,
                  denoising_step_list: settings.denoisingSteps || [700, 500],
                });
              }}
              disabled={
                settings.pipelineId === "passthrough" ||
                settings.pipelineId === "vod" ||
                isPipelineLoading ||
                isConnecting
              }
              isStreaming={isStreaming}
              isVideoPaused={settings.paused}
              timelineRef={timelineRef}
              onLiveStateChange={setIsLive}
              onLivePromptSubmit={handleLivePromptSubmit}
              onDisconnect={stopStream}
              onStartStream={handleStartStream}
              onVideoPlayPauseToggle={handlePlayPauseToggle}
              onPromptEdit={handleTimelinePromptEdit}
              isCollapsed={isTimelineCollapsed}
              onCollapseToggle={setIsTimelineCollapsed}
              externalSelectedPromptId={externalSelectedPromptId}
              settings={settings}
              onSettingsImport={updateSettings}
              onPlayPauseRef={timelinePlayPauseRef}
              onResetCache={handleResetCache}
              onTimelinePromptsChange={handleTimelinePromptsChange}
              onTimelineCurrentTimeChange={handleTimelineCurrentTimeChange}
              onTimelinePlayingChange={handleTimelinePlayingChange}
            />
          </div>
        </div>

        {/* Right Panel - Settings */}
        <div className="w-1/5">
          <SettingsPanel
            className="h-full"
            pipelineId={settings.pipelineId}
            onPipelineIdChange={handlePipelineIdChange}
            isStreaming={isStreaming}
            resolution={
              settings.resolution || getDefaultResolution(settings.pipelineId)
            }
            onResolutionChange={handleResolutionChange}
            seed={settings.seed ?? 42}
            onSeedChange={handleSeedChange}
            denoisingSteps={settings.denoisingSteps || [700, 500]}
            onDenoisingStepsChange={handleDenoisingStepsChange}
            noiseScale={settings.noiseScale ?? 0.7}
            onNoiseScaleChange={handleNoiseScaleChange}
            noiseController={settings.noiseController ?? true}
            onNoiseControllerChange={handleNoiseControllerChange}
            manageCache={settings.manageCache ?? true}
            onManageCacheChange={handleManageCacheChange}
            onResetCache={handleResetCache}
          />
        </div>
      </div>

      {/* Status Bar */}
      <StatusBar fps={webrtcStats.fps} bitrate={webrtcStats.bitrate} />
    </div>
  );
}
