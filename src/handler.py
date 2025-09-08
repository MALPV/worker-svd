import os
import runpod
import torch
import time
import asyncio
import execution
import server
from nodes import NODE_CLASS_MAPPINGS, load_custom_node
import random
import string
import hashlib
import mimetypes
import requests
from pathlib import Path
from PIL import Image
import io

# Setup ComfyUI server
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server)

# Load ComfyUI custom nodes
# Assuming the custom nodes for SVD are in a similar location as mochi's
# This might need adjustment based on the actual custom node structure for SVD
# For now, I'll assume a placeholder name 'ComfyUI-SVD'
load_custom_node("/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite")

# Initialize ComfyUI nodes for the SVD pipeline
# These are based on the ComfyUI SVD examples. The exact names might need verification.
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
SVD_img2vid_Conditioning = NODE_CLASS_MAPPINGS["SVD_img2vid_Conditioning"]()
VideoLinearCFGGuidance = NODE_CLASS_MAPPINGS["VideoLinearCFGGuidance"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VHS_VideoCombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()


# Load models at startup
with torch.inference_mode():
    # Load SVD model
    svd_model, _, _ = CheckpointLoaderSimple.load_checkpoint(
        "svd_xt.safetensors"
    )
    # Load VAE
    vae = VAELoader.load_vae("vae-ft-mse-840000-ema-pruned.safetensors")[0]


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


@torch.inference_mode()
def generate(input):
    values = input["input"]
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
        steps = values.get("steps", 30)
        cfg = values.get("cfg", 2.5)

        # 2. Download and prepare image
        print("Downloading image...")
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        
        # Save image temporarily to be loaded by LoadImage node
        temp_image_path = f"/tmp/input_image.png"
        image.save(temp_image_path)
        
        # 3. Load image using ComfyUI node
        loaded_image, _ = LoadImage.load_image(temp_image_path)

        # 4. SVD pipeline
        positive, negative = SVD_img2vid_Conditioning.encode(
            svd_model,
            loaded_image,
            noise_aug_strength,
            motion_bucket_id,
            0, # video_angle - not used in the example
            0, # video_flip - not used in the example
            width,
            height,
            num_frames,
            1, # batch_size
        )

        # This node seems to be a custom one, let's assume it's available
        # If not, we might need to implement the logic or find an alternative
        model = VideoLinearCFGGuidance.patch(svd_model, 1.0)[0]

        latent = KSampler.sample(
            model,
            seed,
            steps,
            cfg,
            "euler", # sampler_name
            "normal", # scheduler
            positive,
            negative,
            1, # latent_image
            denoise=1.0,
        )[0]

        frames = VAEDecode.decode(vae, latent)[0]

        # 5. Combine frames into video
        print("Combining frames into video.")
        out_video = VHS_VideoCombine.combine_video(
            images=frames,
            frame_rate=fps,
            loop_count=0,
            filename_prefix="SVD",
            format="video/h264-mp4",
            save_output=True,
            pix_fmt="yuv420p",
            crf=17,
        )

        # 6. Upload video
        print("Uploading video.")
        _, output_files = out_video["result"][0]
        video_path = output_files[-1]

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
        # 7. Clean up temporary files
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up temporary file: {video_path}")
        if os.path.exists("/tmp/input_image.png"):
            os.remove("/tmp/input_image.png")


if __name__ == "__main__":
    runpod.serverless.start({"handler": generate})