# worker-svd

> Generate videos from an initial image using Stable Video Diffusion (SVD) as an endpoint on RunPod.

## Features

- Video generation using [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- Automatic model loading from Hugging Face Hub
- [UploadThing](https://uploadthing.com/) integration for video upload

## API Reference

### Input Parameters

```json
{
  "input": {
    "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png",
    "height": 576,
    "width": 1024,
    "num_frames": 25,
    "fps": 7,
    "motion_bucket_id": 127,
    "noise_aug_strength": 0.02,
    "seed": 1337
  }
}
```

#### Parameters

| Parameter            | Description                                                              | Default       |
| -------------------- | ------------------------------------------------------------------------ | ------------- |
| `image_url`          | **Required**. URL of the input image to animate.                         | `null`        |
| `height`             | Output video height. The model was trained on 576x1024.                  | `576`         |
| `width`              | Output video width. The model was trained on 576x1024.                   | `1024`        |
| `num_frames`         | Number of frames to generate.                                            | `25`          |
| `fps`                | Frames per second for the output video.                                  | `7`           |
| `motion_bucket_id`   | Controls the amount of motion in the generated video. Higher is more motion. | `127`         |
| `noise_aug_strength` | How much noise to add to the input image.                                | `0.02`        |
| `seed`               | Random seed for reproducible results.                                    | Current time  |

## Deployment

Deploy this worker on RunPod using the [GitHub Integration](https://docs.runpod.io/serverless/github-integration).

## Development

This project includes a `docker-compose.yml` for local development. Ensure you have an `.env` file with the required API keys (see `.env.example`).

Run the following command to start the local worker with GPU support:

```bash
docker-compose up -d --build
```

## License

[MIT License](LICENSE)
