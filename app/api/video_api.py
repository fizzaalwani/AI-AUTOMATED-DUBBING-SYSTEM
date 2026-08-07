import shutil
import uuid
from pathlib import Path
from app.utils.logger import logger

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile

from sqlalchemy.orm import Session

from app.config import UPLOAD_DIR

from app.database.database import get_db
from app.database.tables import Video
from app.schemas import VideoUploadResponse

router = APIRouter(prefix="/videos", tags=["Videos"])

ALLOWED_EXTENSIONS = {".mp4"}

@router.post("/upload", response_model=VideoUploadResponse, status_code=201)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Upload request received: {file.filename}")
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        logger.warning(f"Unsupported file format: {file.filename}")
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    unique_filename = f"{uuid.uuid4()}{extension}"

    save_path = UPLOAD_DIR / unique_filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        logger.info(f"Video saved successfully: {save_path}")

    new_video = Video(
        original_filename=file.filename,
        stored_filename=unique_filename,
        status="Uploaded",
    )

    try:
        db.add(new_video)
        db.commit()
        db.refresh(new_video)
        logger.info(f"Video metadata saved successfully. video_id={new_video.video_id}")

    except Exception as e:
        logger.error(f"Failed to save video metadata: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the video: {str(e)}")

    return {"message": "Video uploaded successfully.", "video": new_video}