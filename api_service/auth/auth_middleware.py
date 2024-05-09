from typing import Callable
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, exclude_routes: list = None):
        super().__init__(app)
        self.exclude_routes = exclude_routes if exclude_routes else []

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse | Request:
        auth_header = request.headers.get('Authorization')
        if request.url.path not in self.exclude_routes:
            if not auth_header:
                return JSONResponse(status_code=401, content={'detail': 'Auth is invalid or missing'})
            else:
                # TODO: Check the database to see if they have a valid API key
                pass

        response = await call_next(request)
        return response