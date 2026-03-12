from pydantic import BaseModel
from typing import List

class DetectionResponse(BaseModel):
    debris_count: int
    avg_area: float
    max_area: float
    image_path: str

class RiskRequest(BaseModel):
    debris_count: int
    avg_area: float
    max_area: float
    orbital_altitude_km: float
    velocity_kms: float

class RiskResponse(BaseModel):
    current_risk: float
    risk_30_day: float
    risk_90_day: float