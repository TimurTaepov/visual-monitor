from __future__ import annotations

from fastapi import FastAPI

from vlm_evals.serving.router import router

app = FastAPI(title="VLM Serving Evals", version="0.1.0")
app.include_router(router)

