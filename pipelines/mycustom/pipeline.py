import logging
import time

import torch

from ..base.wan2_1.wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from ..blending import PromptBlender
from ..interface import Pipeline, Requirements
from .utils.lora_utils import configure_lora_for_model, load_lora_checkpoint
from ..process import preprocess_chunk, postprocess_chunk

# Add parent directory to path to import real_time_gen_V2
import sys
from pathlib import Path
sys.path.insert(0, "/home/yitong-moonlake/real_time_gen_V3")
from inference import InferencePipeline

logger = logging.getLogger(__name__)


def get_text_prompt():
    subject = "Cinematic scene of a hooded man walking through a historical town built from weathered stone and timber. Cobblestone ground with patches of grass."

    scene = ""
        
    motion = "The camera steadily follows a man from behind, in a third-person perspective typical of adventure or action games. The man remains stable in the lower half of the frame, slightly offset to the left, while the world scrolls and shifts around him. His motion alternates between walking, running, and standing still. The camera adjusts its angle and distance to maintain composition, producing a continuous sense of forward movement and spatial depth as the environment dynamically flows past."

    # mid day vibe
    # aesthetic = "The air is filled with drifting dust and soft rays of sunlight cutting through the mist. The mood is tense yet majestic—an Assassin’s Creed–style world blending realism and myth. Ultra-photorealistic 8K detail, HDR lighting, cinematic composition with shallow depth of field, volumetric light shafts, film-grade color grading with warm sunlight and cool shadow contrast. Inspired by Assassin’s Creed Unity and Kingdom of Heaven, with intricate medieval architecture and atmospheric immersion."
    
    # dusk, edge of the night vibe 
    #aesthetic = "The scene takes place at dusk, as the last traces of daylight fade into deep blue shadows. The air carries a faint haze illuminated by soft, scattered light and the first glimmers of lanterns and window glow. The mood is calm yet mysterious—an Assassin’s Creed–style world at the edge of night, where realism meets mythic atmosphere. Ultra-photorealistic 8K detail, HDR contrast, and cinematic composition with shallow depth of field and subtle volumetric lighting. Film-grade color grading emphasizes cool blues and warm highlights, evoking the mood of twilight scenes in Assassin’s Creed Unity and Kingdom of Heaven."

    # heavily raining vibe
    #aesthetic = "The world is shrouded in heavy rain and dark clouds, with no trace of sunlight piercing the sky. Heavy raindrops streak across the frame, glistening on stone and fabric under the diffuse, gray light. Occasional flashes of lightning. Puddles ripple with every step, and the air feels dense with mist and motion. The mood is somber and cinematic—an Assassin’s Creed–style world drenched in atmosphere, where myth and realism merge in a stormy medieval setting. Ultra-photorealistic 8K detail, HDR tonality, and film-grade color grading emphasize muted tones, wet surfaces, and soft reflections, evoking the moody tension of Assassin’s Creed Unity’s rain-soaked cityscapes."

    # snowy vibe
    #aesthetic = "The world is blanketed in heavy snow, every surface softened under layers of white. Thick flakes drift through the air, swirling in the wind and catching faint glimmers of diffused light. Footsteps leave brief impressions before being buried again. The atmosphere is cold, hushed, and majestic—an Assassin’s Creed–style world transformed into a frozen realm where silence and beauty intertwine. Ultra-photorealistic 8K detail, HDR tonality, and cinematic composition with shallow depth of field and subtle volumetric haze. Film-grade color grading emphasizes cool blues and desaturated tones, evoking the quiet tension and stark beauty of a snowbound medieval landscape."

    # anime vibe
    #aesthetic = "The world is depicted in a painterly, hand-drawn anime style inspired by Hayao Miyazaki. Colors are soft yet vibrant, with expressive brushwork and a sense of wonder in every detail. The air feels alive with drifting particles and subtle movement—leaves rustling, distant clouds gliding, faint shimmer in the light. Architecture and nature coexist harmoniously, blending human craftsmanship with organic forms. Ultra-detailed animation quality with cinematic composition and gentle depth of field. The tone balances warmth and nostalgia with quiet magic, evoking the timeless atmosphere of Spirited Away and Howl’s Moving Castle, where every frame feels alive and emotionally resonant."
    
    # inferno vibe
    # aesthetic = "The world burns in a raging inferno, with fire consuming the horizon and smoke churning across the sky. Embers swirl through the air like sparks of ash, and the ground glows with reflected heat from collapsing structures. The atmosphere is apocalyptic and oppressive—an Assassin’s Creed–style world twisted into a vision of ruin and chaos. The sky is streaked with molten orange and black, as distant flames paint the landscape in shifting patterns of light and shadow. Ultra-photorealistic 8K detail, HDR contrast, and cinematic composition with intense volumetric haze. Film-grade color grading emphasizes deep reds, blacks, and scorched golds, evoking a sense of destruction, despair, and mythic grandeur."
    
    # ultra modern, cyber tokyo vibe
    aesthetic = "The scene unfolds at night in a futuristic Japanese metropolis, where ancient architecture stands illuminated by neon light. Traditional wooden temples and tiled rooftops line narrow streets beneath towering skyscrapers covered in holographic billboards. Vibrant hues of magenta, cyan, and deep blue reflect across rain-soaked cobblestones, merging history and modernity in a luminous dreamscape. Lanterns glow beside digital signs, and drifting mist softens the contrast between old and new. Ultra-photorealistic 8K detail, HDR contrast, and cinematic depth of field enhance the interplay of wet surfaces, reflections, and light shafts. Inspired by Blade Runner 2049 and Ghost in the Shell, reimagined through a distinctly Japanese sense of beauty and transience."

    # tang dynasty noon
    #aesthetic = "The world reflects the grandeur of the Tang dynasty, where ancient Chinese architecture stands in full splendor. Curved tiled roofs, carved wooden beams, red lacquered pillars, and hanging lanterns line the streets. Citizens wear flowing hanfu robes and hair ornaments, moving with calm dignity through open courtyards and stone bridges. The atmosphere feels regal yet lived-in—smoke rising from incense burners, silk banners swaying in the breeze. Ultra-photorealistic 8K detail and HDR contrast capture the richness of lacquer, fabric, and aged wood. Cinematic framing evokes the poetic realism of Hero and House of Flying Daggers, blending historical authenticity with epic visual beauty."

    # tang dynasty night
    #aesthetic = "The scene takes place at night during a grand Tang dynasty festival. Lanterns of every shape and color fill the sky, their reflections shimmering across the river and stone streets. Ornate pavilions and curved rooftops glow with warm light, banners fluttering above the crowd. Citizens in elegant hanfu robes move through the celebration—some dancing, others watching dragon performances and fire-breathers beneath the lantern glow. Incense smoke drifts through the air, blending with the sparks of distant fireworks that paint the night in red and gold. Ultra-photorealistic 8K detail, HDR lighting, and cinematic depth of field capture the movement, color, and rhythm of a lively ancient night, echoing the poetic splendor of Tang-era festivities."
    #aesthetic = "The scene takes place at night during a grand Tang dynasty festival. Lanterns of every shape and color fill the sky, their reflections shimmering across the water and stone streets. Incense smoke drifts through the air, blending with the sparks of distant fireworks that paint the night in red and gold. Ultra-photorealistic 8K detail, HDR lighting, and cinematic depth of field capture the movement, color, and rhythm of a lively ancient night, echoing the poetic splendor of Tang-era festivities."
    
    return subject + " " + scene + " " + motion + " " + aesthetic, 0

class MyCustomPipeline(Pipeline):
    def __init__(
        self,
        config,
        low_memory: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):  

        self.device = device
        self.dtype = dtype
        self.height = 480
        self.width = 832
        self.stream = InferencePipeline()

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements | None:
        if should_prepare: # this is run at reset time
            self.stream.prepare()
            (self.existing_prompt, self.existing_prompt_case) = (None, None)

        # num to request
        num_to_request = 9 if self.stream.current_start == 0 else 12
        return Requirements(input_size=num_to_request) # This is to require n frames from 

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

        (new_prompt, new_prompt_case) = get_text_prompt()
        prompt_refresh = False
        if new_prompt_case != self.existing_prompt_case:
            self.stream.update_prompts(new_prompt)
            prompt_refresh = True
        # update prompt used
        self.existing_prompt, self.existing_prompt_case = new_prompt, new_prompt_case

        output_chunk, output_fresh_chunk, output_depth_chunk, output_scribble_chunk, output_mask_chunk, output_mask_encoded_chunk = self.stream(input, 
                                                                      identity = None,
                                                                      depth= None,#depth_chunk, #None, #depth_chunk, 
                                                                      scribble= None, #scribble_chunk
                                                                      prompt_refresh = prompt_refresh,
                                                                    )

        # Pass the preprocessed input to the stream processor
        return output_chunk

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
