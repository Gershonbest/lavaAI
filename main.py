from fastapi import FastAPI
from api.app.routers import route
import uvicorn


def include_router(app):
    app.include_router(route.router)


def start_app():
    app = FastAPI(title="AI Task")
    include_router(app)

    return app