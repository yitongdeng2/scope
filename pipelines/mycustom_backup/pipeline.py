import sys
from pathlib import Path

import torch

# Add parent directory to path to import real_time_gen_V2
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from real_time_gen_V2.inference import MyStreamer

from ..interface import Pipeline, Requirements
from ..process import preprocess_chunk


class MyCustomPipeline(Pipeline):
    """My custom pipeline for testing"""

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.height = height
        self.width = width
        self.device = device if device is not None else torch.device("cuda")
        self.dtype = dtype
        self.prompts = None

        self.stream = MyStreamer(
            height=height,
            width=width,
            device=device,
            dtype=dtype
        )

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def __call__(
        self,
        input: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        if input is None:
            raise ValueError("Input cannot be None for MyCustomPipeline")

        # Note: The caller must call prepare() before __call__()
        
        # If input is a list of frames, preprocess them
        # This converts list[Tensor] -> Tensor in BCTHW format with values in [-1, 1]
        if isinstance(input, list):
            input = preprocess_chunk(
                input, self.device, self.dtype, height=self.height, width=self.width
            )

        # Pass the preprocessed input to the stream processor
        return self.stream(input)
