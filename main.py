from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import asyncio
import logging

from app.utils.response_handler import ( global_exception_handler, http_exception_handler, validation_exception_handler)
from app.config.logging import init_logging_system
# from app.utils.load_model_version import load_model
# from app.utils.model_download import download_model
# from app.utils.model_version import is_model_exists

from app.core.metadata import tags_metadata
from app.documentation.project_documentation import project_description


init_logging_system()

logger = logging.getLogger(__name__)

from app.routers.v1.routes import router as dl_router_v1
from app.routers.v1.train_routes import router as train_router_v1
from app.routers.v2.train_routes import router as train_router_v2
from app.routers.v2.routes import router as dl_router_v2
from app.routers.task_router import router as task_router
from app.routers.v2.database_router import router as database_router
from app.routers.v2.model_router import router as model_router

app = FastAPI(
    title="Plant Disease Detection API",
    description=project_description,
    version="1.4.2",
    contact={
        "name": "SRINADH CHINTAKINDI",
        "email": "srinadhch03@gmail.com",
    },
    license_info={
        "name": "MIT License",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    debug=True
)

app.include_router(dl_router_v1, prefix="/api", tags=["Inference v1"])
app.include_router(dl_router_v2, prefix="/api",tags=["Inference v2"])
app.include_router(train_router_v1, prefix="/api",tags=["Neural network pipelines v1"])
app.include_router(train_router_v2, prefix="/api",tags=["Neural network pipelines v2"])
app.include_router(task_router, prefix="/api", tags=["Task monitoring","Inference v1", "Neural network pipelines v1", "Neural network pipelines v2" , "Inference v2"])
app.include_router(database_router, prefix="/api", tags=["Database Pipelines"])
app.include_router(model_router, prefix="/api", tags=["Model and versions"])

app.add_middleware( CORSMiddleware, allow_origins=["*"],  allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    pass
    # model_versions = is_model_exists()
    # if model_versions.get("small"):
    #     logger.info("Existing model detected.") if model_versions.get("small") else logger.info("No model detected, downloading model...")
    # else:
    #     download_model()
    # load_model("small")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/meta/favicon.ico")

@app.get("/", tags=["Root"])
async def home():
    return RedirectResponse(url="/docs")
