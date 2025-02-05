""" Handler for FLUX image generation. """

import os
import torch
import base64
from io import BytesIO
from diffusers import FluxPipeline
import runpod

# Load the model into memory
pipe = None

def init_model():
    global pipe
    if pipe is None:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    return pipe

# Initialize the model when the worker starts
init_model()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    
    # Get parameters from the job input, falling back to environment variables if not provided
    prompt = job_input.get('prompt', os.getenv('PROMPT', 'A cat holding a sign that says hello world'))
    height = int(job_input.get('height', os.getenv('HEIGHT', 768)))
    width = int(job_input.get('width', os.getenv('WIDTH', 1360)))
    num_inference_steps = int(job_input.get('num_inference_steps', os.getenv('NUM_INFERENCE_STEPS', 4)))
    guidance_scale = float(job_input.get('guidance_scale', os.getenv('GUIDANCE_SCALE', 0.0)))
    max_sequence_length = int(job_input.get('max_sequence_length', os.getenv('MAX_SEQUENCE_LENGTH', 256)))

    # Generate the image
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
    ).images[0]
    
    # Convert PIL image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        "image_base64": image_base64,
        "prompt": prompt,
        "parameters": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length
        }
    }

runpod.serverless.start({"handler": handler})
#dsda