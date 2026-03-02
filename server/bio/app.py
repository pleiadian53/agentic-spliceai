"""AgenticSpliceAI Lab entry point.

Usage:
    conda run -n agentic-spliceai python -m server.bio.app
"""

import logging
import uvicorn

from . import config
from .bio_service import app  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    uvicorn.run(
        "server.bio.bio_service:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        reload_dirs=["server/bio"],
    )
