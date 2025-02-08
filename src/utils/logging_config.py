"""Logging configuration for the FLUX worker."""

import logging


def get_logger(name="flux_worker"):
    """Get a configured logger instance.

    Args:
        name (str): Name for the logger instance.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger_instance = logging.getLogger(name)

    # Only configure if handlers haven't been set up
    if not logger_instance.handlers:
        # Set up logging format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)

        # Set level
        logger_instance.setLevel(logging.INFO)

    return logger_instance


# Create the default logger instance
logger = get_logger()
