import os
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from io import BytesIO

def fetch_image_urls(base_url):
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    image_urls = [base_url.rstrip('/') + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.jpg')]
    return image_urls

def load_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        image = cv2.resize(image, (1024, 1024))
        return image
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None

def generate_video_for_image(image_url, folder_path, pipe):
    print(f"Processing image: {image_url}")
    image = load_image(image_url)
    if image is None:
        print(f"Skipping image: {image_url}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    generator = torch.manual_seed(42)

    pipe.enable_model_cpu_offload()
    pipe.unet.enable_forward_chunking()

    frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]

    video_filename = os.path.join(folder_path, os.path.basename(image_url).replace('.jpg', '.mp4'))
    export_to_video(frames, video_filename, fps=7)
    print(f"Generated video: {video_filename}")

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

