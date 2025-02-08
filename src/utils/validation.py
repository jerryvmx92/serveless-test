"""Input validation for the FLUX worker."""

from pydantic import BaseModel, Field, field_validator


class GenerationParameters(BaseModel):
    """Validation model for image generation parameters."""

    prompt: str = Field(
        default="A cat holding a sign that says hello world",
        description="Text prompt for image generation",
    )
    height: int = Field(
        default=768,
        ge=64,  # greater than or equal to 64
        le=1024,  # less than or equal to 1024
        description="Height of the generated image",
    )
    width: int = Field(default=1360, ge=64, le=1024, description="Width of the generated image")
    num_inference_steps: int = Field(
        default=4, ge=1, le=100, description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        default=0.0, ge=0.0, le=20.0, description="Scale for classifier-free guidance"
    )
    max_sequence_length: int = Field(
        default=77, ge=1, le=77, description="Maximum sequence length for the prompt"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate that the prompt is not empty and is reasonable length."""
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 500:
            raise ValueError("Prompt is too long (max 500 characters)")
        return v


class JobInput(BaseModel):
    """Validates and processes job input for the FLUX worker."""

    input: GenerationParameters = Field(..., description="Input parameters for image generation")
