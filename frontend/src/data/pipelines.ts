export type PipelineCategory = "video-input" | "no-video-input";

export interface PipelineInfo {
  name: string;
  about: string;
  projectUrl?: string;
  modified?: boolean;
  category: PipelineCategory;
  defaultPrompt?: string;
}

export const PIPELINES: Record<string, PipelineInfo> = {
  mycustom: {
    name: "MyCustom",
    about:
      "Yitong's custom pipeline for testing",
    modified: true,
    category: "no-video-input",
    defaultPrompt:
      "Cinematic scene of a hooded man walking through a medieval town built from weathered stone and timber. Narrow cobblestone streets wind between tall gothic buildings with archways, hanging banners, and glowing lanterns. The air is filled with drifting dust and soft rays of sunlight cutting through the mist. Merchants and townsfolk move in the distance, adding life to the bustling market square. The mood is tense yet majestic—an Assassin’s Creed–style world blending realism and myth. Ultra-photorealistic 8K detail, HDR lighting, cinematic composition with shallow depth of field, volumetric light shafts, film-grade color grading with warm sunlight and cool shadow contrast. Inspired by Assassin’s Creed Unity and Kingdom of Heaven, with intricate medieval architecture and atmospheric immersion.",
  },
  streamdiffusionv2: {
    name: "StreamDiffusionV2",
    projectUrl: "https://streamdiffusionv2.github.io/",
    about:
      "A streaming pipeline and autoregressive video diffusion model from the creators of the original StreamDiffusion project. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support streaming.",
    modified: true,
    category: "video-input",
    defaultPrompt: "A dog in the grass looking around, photorealistic",
  },
  longlive: {
    name: "LongLive",
    projectUrl: "https://nvlabs.github.io/LongLive/",
    about:
      "A streaming pipeline and autoregressive video diffusion model from Nvidia, MIT, HKUST, HKU and THU. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support smoother prompt switching and improved quality over longer time periods while maintaining fast generation.",
    modified: true,
    category: "no-video-input",
    defaultPrompt:
      "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
  },
  passthrough: {
    name: "Passthrough",
    about:
      "A pipeline that returns the input video without any processing that is useful for testing and debugging.",
    category: "video-input",
  },
  // vod: {
  //   name: "VOD",
  //   about:
  //     "A pipeline that returns a static video file without any processing that is useful for testing and debugging.",
  //   category: "no-video-input",
  // },
};
