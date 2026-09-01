from fastapi import FastAPI

from .routers import assam
from .routers import uttarakhand
from .routers import odisha


app = FastAPI(
    title="Disaster Resilience & Relocation API",
    description="Disaster risk and relocation backend for Assam, Uttarakhand and Odisha",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Disaster Resilience API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/states")
def get_states():
    return {
        "states": [
            "Assam",
            "Uttarakhand",
            "Odisha"
        ]
    }


app.include_router(assam.router)
app.include_router(uttarakhand.router)
app.include_router(odisha.router)