import os
import requests
from bs4 import BeautifulSoup
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

def fetch_image_urls(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_urls = [base_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.jpg')]
    return image_urls

def generate_video_for_image(image_url, folder_path, pipe):
    image = load_image(image_url)
    image = image.resize((1024, 1024))

    generator = torch.manual_seed(42)

    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()

    frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]

    video_filename = os.path.join(folder_path, os.path.basename(image_url).replace('.jpg', '.mp4'))
    export_to_video(frames, video_filename, fps=7)

def main(base_url, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    image_urls = fetch_image_urls(base_url)
    
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    
    pipe.to("cuda")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    for image_url in image_urls:
        generate_video_for_image(image_url, folder_path, pipe)

if __name__ == "__main__":
    import sys
    base_url = sys.argv[1]
    folder_path = sys.argv[2]
    main(base_url, folder_path)

