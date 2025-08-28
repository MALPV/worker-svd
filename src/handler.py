
import os
import runpod
import torch
import time
import random
import string
import hashlib
import mimetypes
import requests
import imageio
import io
from pathlib import Path
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# --- Model Loading ---
# This is done once when the worker starts.
print("Loading model...")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")
print("Model loaded.")

# --- Upload Function (from worker-mochi) ---
def upload_file_to_uploadthing(
    file_path: str | Path,
    max_retries: int = 2,
    initial_delay: float = 5.0,
) -> tuple[requests.Response, requests.Response, str]:
    """
    Upload file to UploadThing with retry mechanism.
    """
    attempt = 0
    last_error = None
    file_path = Path(file_path)

    while attempt <= max_retries:
        try:
            if attempt > 0:
                delay = initial_delay * (2 ** (attempt - 1))
                print(f"Retry attempt {attempt}/{max_retries} after {delay:.1f}s delay...")
                time.sleep(delay)

            file_name = file_path.name
            file_extension = file_path.suffix
            random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            md5_hash = hashlib.md5(random_string.encode()).hexdigest()
            new_file_name = f"{md5_hash}{file_extension}"
            file_size = file_path.stat().st_size
            file_type, _ = mimetypes.guess_type(str(file_path))

            with open(file_path, "rb") as file:
                file_content = file.read()

            file_info = {"name": new_file_name, "size": file_size, "type": file_type}
            uploadthing_api_key = os.getenv('UPLOADTHING_API_KEY')
            
            if not uploadthing_api_key:
                raise ValueError("UPLOADTHING_API_KEY environment variable not set")

            headers = {"x-uploadthing-api-key": uploadthing_api_key}
            data = {
                "contentDisposition": "inline",
                "acl": "public-read",
                "files": [file_info],
            }

            presigned_response = requests.post(
                "https://api.uploadthing.com/v6/uploadFiles",
                headers=headers,
                json=data,
            )
            presigned_response.raise_for_status()
            
            presigned = presigned_response.json()["data"][0]
            upload_url = presigned["url"]
            fields = presigned["fields"]

            files = {"file": file_content}
            upload_response = requests.post(upload_url, data=fields, files=files)
            upload_response.raise_for_status()

            print(f"File uploaded successfully: {presigned['fileUrl']}")
            return presigned_response, upload_response, new_file_name

        except Exception as e:
            last_error = e
            print(f"Upload attempt {attempt + 1} failed: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException):
                print(f"Request error details: {e.response.text if e.response else 'No response'}")
            attempt += 1

    raise last_error


# --- Main Handler ---
@torch.inference_mode()
def generate(job):
    values = job["input"]
    video_path = None
    try:
        # 1. Get parameters
        image_url = values.get("image_url")
        if not image_url:
            raise ValueError("Missing 'image_url' in input")

        height = values.get("height", 576)
        width = values.get("width", 1024)
        num_frames = values.get("num_frames", 25)
        fps = values.get("fps", 7)
        motion_bucket_id = values.get("motion_bucket_id", 127)
        noise_aug_strength = values.get("noise_aug_strength", 0.02)
        seed = values.get("seed", int(time.time()))

        # 2. Download and prepare image
        print("Downloading image...")
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image = image.resize((width, height))

        # 3. Set seed
        generator = torch.manual_seed(seed)

        # 4. Generate video frames
        print("Generating video frames...")
        frames = pipe(
            image,
            decode_chunk_size=8,
            generator=generator,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_frames=num_frames,
        ).frames[0]

        # 5. Save frames as video file
        print("Combining frames into video...")
        video_path = Path("/tmp/output.mp4")
        imageio.mimsave(video_path, frames, fps=fps)

        # 6. Upload video
        print("Uploading video...")
        presigned_response, _, _ = upload_file_to_uploadthing(video_path)
        video_url = presigned_response.json()['data'][0]['fileUrl']

        return {
            "result": video_url,
            "status": "SUCCESS"
        }

    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return {
            "status": "ERROR",
            "error": str(e),
            "result": None
        }
    finally:
        # 7. Clean up temporary file
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up temporary file: {video_path}")


# --- Start Server ---
if __name__ == "__main__":
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": generate})
