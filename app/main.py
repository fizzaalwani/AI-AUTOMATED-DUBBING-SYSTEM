from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.database import engine
from app.database.tables import Base

from app.api import video_api
from app.api import speech_api
from app.api import translation_api
from app.api import voice_synthesis_api
from app.api import video_generator_api
from app.api import pipeline_api

from app.utils.logger import logger

Base.metadata.create_all(bind=engine)
logger.info("Database initialized successfully.")

app = FastAPI(
    title="Automated Video Dubbing API",
    version="1.0.0",
    description="Automated Video Dubbing System with Voice Preservation"
)
logger.info("FastAPI application created.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(video_api.router)
app.include_router(speech_api.router)
app.include_router(translation_api.router)
app.include_router(voice_synthesis_api.router)
app.include_router(video_generator_api.router)
app.include_router(pipeline_api.router)
logger.info("All API routers registered successfully.")

@app.get("/")
def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Automated Video Dubbing Backend Running"}


#streamlit run app/frontend.py
#uvicorn app.main:app --reload
#uvicorn app.main:app --reload --host 127.0.0.1 --port 8000


# cd C:\Users\fizza\Downloads\postgresql-18.4-2\pgsql

# bin\pg_ctl.exe start -D "C:\Users\fizza\Downloads\postgresql-18.4-2\pgsql\data"
