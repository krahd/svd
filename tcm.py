import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)

# slower (25%) but uses less memory
#pipe.enable_model_cpu_offload()


# 25% faster, uses a little bit more memory
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Load the conditioning image
#image = load_image("http://files.laurenzo.net/svd/a.png")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")

image = image.resize((1024, 1024))

# seed
generator = torch.manual_seed(42)

# more memory usage
#frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

# less memory usage
pipe.enable_model_cpu_offload()
pipe.unet.enable_forward_chunking()

# less motion
#frames = pipe(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]

# more motion
frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]


export_to_video(frames, "generated.mp4", fps=7)
