from orchestrator.analyze import analyze_texts,extract_keywords
from model.orchestrarator_model import TextsRequest
from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def analyze(request: TextsRequest):
    result = analyze_texts(request.texts)
    return result