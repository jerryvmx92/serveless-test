"""Handler for FLUX image generation."""

import base64
from io import BytesIO

import runpod
import torch
from diffusers import FluxPipeline

from utils.logging_config import logger
from utils.validation import JobInput


class ModelManager:
    """Manages the FLUX model instance."""

    def __init__(self):
        """Initialize the model manager."""
        self.pipe = None

    def init_model(self):
        """Initialize the FLUX model with appropriate settings."""
        try:
            if self.pipe is None:
                logger.info("Initializing FLUX model")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Using device: %s", device)

                self.pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
                )

                if device == "cuda":
                    self.pipe.enable_model_cpu_offload()
                else:
                    self.pipe = self.pipe.to(device)

                logger.info("Model initialization complete")
        except Exception as e:
            logger.error("Failed to initialize model: %s", str(e), exc_info=True)
            raise
        return self.pipe


# Create model manager instance
model_manager = ModelManager()


def init_model():
    """Initialize the model (for backward compatibility)."""
    return model_manager.init_model()


def handler(job):
    """Handler function that will be used to process jobs."""
    try:
        # Validate input
        job_input = JobInput(input=job["input"])
        params = job_input.input

        logger.info("Processing job with prompt: %s", params.prompt)

        # Initialize model if not already initialized
        if model_manager.pipe is None:
            model_manager.init_model()

        # Generate the image
        image = model_manager.pipe(
            prompt=params.prompt,
            guidance_scale=params.guidance_scale,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            max_sequence_length=params.max_sequence_length,
        ).images[0]

        # Convert PIL image to base64 string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info("Image generation complete")

        return {
            "image_base64": image_base64,
            "prompt": params.prompt,
            "parameters": {
                "height": params.height,
                "width": params.width,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "max_sequence_length": params.max_sequence_length,
            },
        }

    except Exception as e:
        logger.error("Error processing job: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    init_model()
    runpod.serverless.start({"handler": handler})
