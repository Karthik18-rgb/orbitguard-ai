from fastapi import APIRouter
from backend.schemas import RiskRequest, RiskResponse
from backend.services.risk_model import predict_risk

router = APIRouter()

@router.post("/risk", response_model=RiskResponse)
def calculate_risk(data:RiskRequest):
    result = predict_risk(data.dict())
    return RiskResponse(
        current_risk=result["current_risk"],
        risk_30_day=result["risk_30_day"],
        risk_90_day=result["risk_90_day"]
    )