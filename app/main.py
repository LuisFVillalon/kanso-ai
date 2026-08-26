from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.learning_resources_router import router
from dotenv import load_dotenv
import os

load_dotenv() 

app = FastAPI(title="TaskMaster AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://kanso-web-app.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Kanso AI"}

@app.get("/health")
async def health():
    return {"status": "ok"}