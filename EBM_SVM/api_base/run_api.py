"""
Application startup script using Uvicorn ASGI server.

This script provides a convenient way to start the FastAPI application
with standard configuration.

Author: Development Team
Version: 1.0.0
"""

import uvicorn
import logging
from app.config import settings

logger = logging.getLogger(__name__)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    workers: int = 1
) -> None:
    """
    Run the FastAPI server.
    
    Args:
        host (str): Server host address. Default: 0.0.0.0
        port (int): Server port. Default: 8000
        reload (bool): Enable auto-reload on code changes. Default: True
        workers (int): Number of worker processes. Default: 1
    """
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    reload = not ("--no-reload" in sys.argv)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    print(f"Starting API server on http://{host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")
    
    run_server(host=host, port=port, reload=reload, workers=workers)
