from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import asyncio

from app.utils.response_handler import ( global_exception_handler, http_exception_handler, validation_exception_handler)
from app.config.logging import init_logging_system
from app.build.load_model_v1 import model

init_logging_system()

from app.routers.v1.routes import router as ml_router
from app.routers.v1.train_routes import router as train_router
app = FastAPI(
    title="Plant Disease Detection API",
    description="""
    Production-ready ML API for:
    -  Plant species identification
    -  Leaf disease detection
    -  Model training & evaluation
    """,
    version="1.0.0",
    contact={
        "name": "SRINADH CHINTAKINDI",
        "email": "srinadhch03@gmail.com",
    },
    license_info={
        "name": "MIT License",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    debug=False
)

app.include_router( ml_router, prefix="/api")
app.include_router(train_router, prefix="/api")

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
