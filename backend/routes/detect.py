from fastapi import APIRouter, UploadFile, File
from backend.schemas import DetectionResponse
from backend.services.detector import run_detction
import os
import shutil

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/detect", response_model=DetectionResponse)
async def detect_debris(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    debris_count, avg_area, max_area = run_detction(file_path)

    return DetectionResponse(
        debris_count=debris_count,
        avg_area=avg_area,
        max_area=max_area,
        image_path=file_path
    )