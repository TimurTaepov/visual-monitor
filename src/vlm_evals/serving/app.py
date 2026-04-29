from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from vlm_evals.serving.backend_manager import backend_manager
from vlm_evals.serving.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    backend_manager.close_all()


app = FastAPI(title="VLM Serving Evals", version="0.1.0", lifespan=lifespan)
app.include_router(router)
