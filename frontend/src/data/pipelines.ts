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
    category: "video-input",
    defaultPrompt:
      "A cinematic scene of a man walking through a vast sun-lit meadow surrounded by rolling green hills and wildflowers. Epic scenery with distant mountains and drifting clouds. Golden-hour lighting, warm sunlight, soft atmospheric haze, and gentle wind moving the grass. Realistic depth of field and natural lens flares. 35 mm film look, shallow focus, eye-level camera tracking smoothly behind the man. Photorealistic, ultra-detailed textures, high dynamic range color grading, inspired by The Hobbit and Weta Digital cinematography, filmic tone mapping, volumetric light, cinematic realism.",
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
