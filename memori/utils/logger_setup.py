import logging
from pathlib import Path

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logger
def setup_logger(name: str = "memori", log_file: str = "memori.log", level=logging.INFO):
    """Set up a logger for Memori with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler
    file_handler = logging.FileHandler(LOG_DIR / log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Memori logger is set up successfully!")
