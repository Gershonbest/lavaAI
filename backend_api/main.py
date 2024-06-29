from fastapi import FastAPI
from api.app.routers import route
import uvicorn

def include_router(app):
    app.include_router(route.router)


def start_app():
    app = FastAPI(title="Lava AI Application", 
                  description= "API for all kind of AI task for business and Educational purpose")
    include_router(app)

    return app

if __name__ == "__main__":
    app = start_app()
    uvicorn.run(app=app, host="127.0.0.1", port=8000)