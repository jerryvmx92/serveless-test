"""Tests for the FLUX image generation handler."""

import base64
import os
import sys
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image
from pydantic import ValidationError

# Add the src directory to the path so we can import the handler
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from handler import handler, init_model, model_manager


class TestHandler(unittest.TestCase):
    """Test cases for the FLUX image generation handler."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock image for testing
        self.test_image = Image.new("RGB", (64, 64), color="red")
        buffered = BytesIO()
        self.test_image.save(buffered, format="PNG")
        self.test_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Create mock pipe
        self.mock_pipe = MagicMock()
        self.mock_pipe.return_value = MagicMock(images=[self.test_image])

        # Patch the model manager
        patcher = patch.object(model_manager, "pipe", self.mock_pipe)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_basic_generation(self):
        """Test basic image generation with default parameters."""
        job = {
            "input": {
                "prompt": "A test image of a cat",
                "height": 768,
                "width": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 0.0,
                "max_sequence_length": 77,
            }
        }

        result = handler(job)

        # Check if result contains expected keys
        self.assertIn("image_base64", result)
        self.assertIn("prompt", result)
        self.assertIn("parameters", result)

        # Verify the mock was called with correct parameters
        self.mock_pipe.assert_called_once_with(
            prompt="A test image of a cat",
            guidance_scale=0.0,
            height=768,
            width=1024,
            num_inference_steps=4,
            max_sequence_length=77,
        )

    def test_custom_parameters(self):
        """Test image generation with custom parameters."""
        job = {
            "input": {
                "prompt": "A test image",
                "height": 512,
                "width": 512,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "max_sequence_length": 77,
            }
        }

        result = handler(job)

        # Verify parameters were respected
        self.assertEqual(result["parameters"]["height"], 512)
        self.assertEqual(result["parameters"]["width"], 512)
        self.assertEqual(result["parameters"]["num_inference_steps"], 2)
        self.assertEqual(result["parameters"]["guidance_scale"], 1.0)
        self.assertEqual(result["parameters"]["max_sequence_length"], 77)

        # Verify the mock was called with correct parameters
        self.mock_pipe.assert_called_once_with(
            prompt="A test image",
            guidance_scale=1.0,
            height=512,
            width=512,
            num_inference_steps=2,
            max_sequence_length=77,
        )

    def test_empty_prompt(self):
        """Test that empty prompt raises validation error."""
        job = {
            "input": {
                "prompt": "",
                "height": 768,
                "width": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 0.0,
                "max_sequence_length": 77,
            }
        }

        with self.assertRaises(ValidationError) as context:
            handler(job)

        self.assertIn("Prompt cannot be empty", str(context.exception))

        # Verify the mock was not called
        self.mock_pipe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
