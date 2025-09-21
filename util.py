import logging

def setup_logger(log_file: str, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(level=level)  # Set the logging level

    # Create handlers
    console_handler = logging.StreamHandler()  # For console output
    file_handler = logging.FileHandler(log_file)  # For file output

    # Set logging levels for handlers
    console_handler.setLevel(level=level)  # Console: INFO and above
    file_handler.setLevel(level=level)    # File: DEBUG and above

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger