import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402  (env must be loaded before the LLM client is built)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.api.routes.learning_resources_router import router  # noqa: E402

# Comma-separated list of allowed frontend origins; same convention as the
# main backend, so a new deployment target is an env var, not a code change.
_DEFAULT_ORIGINS = "http://localhost:3000,https://kanso-web-app.vercel.app"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(
    title="kanso AI",
    description="Finds a video, article and exercise to go deeper on a note's topic.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "kanso AI service. See /docs for the interactive reference."}


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
