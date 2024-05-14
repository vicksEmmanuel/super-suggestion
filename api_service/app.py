import os
import multiprocessing
import sys
from fastapi import FastAPI, HTTPException
import uvicorn.config
from auth.auth_middleware import AuthMiddleware
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from util.util import load_env
from grpc_server import start_grpc_server
from router import code_generator, file_manager
from starlette.responses import JSONResponse
import uvicorn


try:
    load_env()
except FileNotFoundError:
    print("No .env file found")


def make_app():
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):  # pylint: disable=unused-argument
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    app.add_middleware(AuthMiddleware, exclude_routes=[])

    app.include_router(code_generator.router)
    app.include_router(file_manager.router, prefix="/file")

    return app


def start_uvicorn_server():
    host = os.getenv('HOST_URL_HTTP')
    port = os.getenv('HOST_PORT_HTTP')
    uvicorn.run(make_app(), host=host, port=int(port), log_level="info")

if __name__ == "__main__":
    p = multiprocessing.Process(target=start_uvicorn_server)
    p.start()
    start_grpc_server()
    uvicorn.join()
