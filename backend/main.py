from fastapi import FastAPI
from backend.routes import detect, risk

app = FastAPI(
    title="ORBITGUARD Backend",
    description="Space Debris Collision Risk System API",
    version="1.0.0"
)
app.include_router(detect.router, prefix="/api")
app.include_router(risk.router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ORBITGUARD Backend running"}