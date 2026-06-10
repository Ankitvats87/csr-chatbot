from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/")
async def root_redirect():
    return RedirectResponse(url="/admin")


@router.get("/health")
async def health(request: Request) -> dict:
    deps = request.app.state.deps
    return {
        "status": "ok",
        "sqlite": deps.sqlite.health_ok(),
        "pinecone": deps.pinecone.health_ok(),
        "llm_provider": deps.settings.llm_provider,
    }
