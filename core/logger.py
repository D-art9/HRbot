# Logging utilities
import logging
import os

LOG_PATH = "data/logs/agent.log"

os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
