from fastapi import FastAPI

app = FastAPI(
    title="Disaster Resilience & Relocation API",
    description="Backend API for disaster risk assessment and relocation planning",
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