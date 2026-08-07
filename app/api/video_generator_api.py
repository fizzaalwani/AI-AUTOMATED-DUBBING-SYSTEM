from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from app.utils.logger import logger
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.tables import Video
from app.database.tables import VoiceSynthesis

from app.services.video_generator import dub_video

from app.config import OUTPUT_DIR
from app.config import UPLOAD_DIR


router = APIRouter( prefix="/video-dubber", tags=["Video Dubber"])

@router.post("/generate")
def generate_video( video_id: str, db: Session = Depends(get_db)):
    logger.info(f"Video dubbing request received for video_id={video_id}")
    video = db.query(Video).filter(Video.video_id == video_id).first()

    if video is None:
        logger.warning(f"Video not found: {video_id}")
        raise HTTPException( status_code=404, detail="Video not found.")
    
    voice = db.query(VoiceSynthesis).filter(VoiceSynthesis.video_id == video_id).first()

    if voice is None:
        logger.warning(f"Generated speech not found for video_id={video_id}")
        raise HTTPException( status_code=404, detail="Generated speech not found.")

    video.status = "Processing"
    db.commit()

    logger.info("Dubbed video generation started.")
    result = dub_video(
        video_path=UPLOAD_DIR / video.stored_filename,
        dubbed_audio_path=voice.generated_audio_path,
        output_dir=OUTPUT_DIR
    )

    try:
        voice.dubbed_video_path = str(result["dubbed_video"])
        voice.video_processing_time = result["processing_time"]
        video.status = "Completed"
        db.commit()
        logger.info(f"Dubbed video saved successfully: {voice.dubbed_video_path}")

    except Exception as e:
        logger.error(f"Failed to save dubbed video: {e}")
        db.rollback()
        raise

    return {
        "message": "Video generated successfully.",
        "dubbed_video": voice.dubbed_video_path,
        "processing_time": voice.video_processing_time
    }
