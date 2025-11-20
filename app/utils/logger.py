import sys
from loguru import logger
from app.config import settings


def setup_logger():
    """Configure application logger with appropriate format and level."""
    
    # Remove default handler
    logger.remove()
    
    # Add custom handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # Add file handler for errors
    logger.add(
        "logs/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days"
    )
    
    return logger


# Initialize logger on module import
app_logger = setup_logger()
