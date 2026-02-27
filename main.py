from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import asyncio

from app.utils.response_handler import ( global_exception_handler, http_exception_handler, validation_exception_handler)
from app.config.logging import init_logging_system

init_logging_system()

from app.routers.v1.routes import router as ml_router
app = FastAPI( title="Machine Learning model", version="1.0.0", docs_url="/docs", redoc_url="/redoc", Debug=False)

app.include_router( ml_router, prefix="/api")

app.add_middleware( CORSMiddleware, allow_origins=["*"],  allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    pass

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/meta/favicon.ico")

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")
