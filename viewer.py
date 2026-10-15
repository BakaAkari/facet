"""
Entry point for the FastAPI API server.

Usage:
    python viewer.py                    # Development (auto-reload)
    python viewer.py --production       # Production mode

Or directly with uvicorn:
    uvicorn api:create_app --factory --reload --port 5000
"""

import os
import sys
import argparse

# Ensure the script's directory is in Python path for local imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

def main():
    parser = argparse.ArgumentParser(description='Facet API Server')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)),
                        help='Port to listen on (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (production)')
    args = parser.parse_args()

    from api import CONFIGURE_LOGGING_ENV, configure_logging
    configure_logging()
    # uvicorn serves from processes it spawns (each --workers worker, the --reload
    # child); they inherit the environment, and the app factory configures their
    # logging from this flag, so the app's INFO lines reach the log there too.
    os.environ[CONFIGURE_LOGGING_ENV] = "1"

    import uvicorn

    # Disable uvicorn's access log: its default format records the full request
    # line including the query string, which would leak share/frame/scan tokens
    # (passed as query params) into plaintext logs. The app's own
    # RequestLoggingMiddleware logs the path only.
    if args.production:
        uvicorn.run(
            "api:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers,
            access_log=False,
        )
    else:
        uvicorn.run(
            "api:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[_script_dir],
            access_log=False,
        )


if __name__ == '__main__':
    main()
