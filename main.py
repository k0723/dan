from router.orchestrator_router import router as orchestrator_router
from fastapi import FastAPI

app = FastAPI()


app.include_router(orchestrator_router, prefix="/orchestrator")
