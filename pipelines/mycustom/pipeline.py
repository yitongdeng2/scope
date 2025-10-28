import logging
import time

import torch

from ..base.wan2_1.wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from ..blending import PromptBlender
from ..interface import Pipeline, Requirements
from .utils.lora_utils import configure_lora_for_model, load_lora_checkpoint
from ..process import postprocess_chunk

# Add parent directory to path to import real_time_gen_V2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from real_time_gen_V3.inference import InferencePipeline

logger = logging.getLogger(__name__)


class MyCustomPipeline(Pipeline):
    def __init__(
        self,
        config,
        low_memory: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):

        seed = getattr(config, "seed", 42)

        self.stream = InferencePipeline(
            config, low_memory, seed
        ).to(device=device, dtype=dtype)

        self.prompts = None
        self.denoising_step_list = None

        # Prompt blending
        self.prompt_blender = PromptBlender(device, dtype)

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements | None:
        # If caller requested prepare assume cache init
        # Otherwise no cache init
        init_cache = should_prepare

        manage_cache = kwargs.get("manage_cache", None)
        prompts = kwargs.get("prompts", None)
        prompt_interpolation_method = kwargs.get(
            "prompt_interpolation_method", "linear"
        )
        denoising_step_list = kwargs.get("denoising_step_list", None)

        # Check if prompts changed using prompt blender
        if self.prompt_blender.should_update(prompts, prompt_interpolation_method):
            logger.info("prepare: Initiating pipeline prepare for prompt update")
            should_prepare = True

        if (
            denoising_step_list is not None
            and denoising_step_list != self.denoising_step_list
        ):
            should_prepare = True

            if manage_cache:
                init_cache = True

        if should_prepare:
            # Update internal state
            if denoising_step_list is not None:
                self.denoising_step_list = denoising_step_list

            # Apply prompt blending and prepare stream
            self._apply_prompt_blending(
                prompts, prompt_interpolation_method, denoising_step_list, init_cache
            )

        return None

    def __call__(
        self,
        _: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        # Note: The caller must call prepare() before __call__()
        return postprocess_chunk(self.stream())

    def _apply_prompt_blending(
        self,
        prompts=None,
        interpolation_method="linear",
        denoising_step_list=None,
        init_cache: bool = False,
    ):
        """Apply weighted blending of cached prompt embeddings."""
        combined_embeds = self.prompt_blender.blend(
            prompts, interpolation_method, self.stream.text_encoder
        )

        if combined_embeds is None:
            return

        # Set the blended embeddings on the stream
        self.stream.conditional_dict = {"prompt_embeds": combined_embeds}

        # Call stream prepare to update the pipeline with denoising steps
        self.stream.prepare(
            prompts=None, denoising_step_list=denoising_step_list, init_cache=init_cache
        )
