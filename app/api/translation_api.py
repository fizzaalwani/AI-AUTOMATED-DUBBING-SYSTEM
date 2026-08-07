from app.utils.logger import logger

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.tables import Transcript
from app.database.tables import Translation
from app.database.tables import Video

from app.services.translator import translate

router = APIRouter(prefix="/translation", tags=["Translation"])

@router.post("/translate")
def translate_video(video_id: str,db: Session = Depends(get_db)):
    logger.info(f"Translation request received for video_id={video_id}")

    video = db.query(Video).filter(Video.video_id == video_id).first()
    if video is None:
        logger.warning(f"Video not found: {video_id}")
        raise HTTPException( status_code=404, detail="Video not found.")

    transcript = db.query(Transcript).filter(Transcript.video_id == video_id).first()
    if transcript is None:
        logger.warning(f"Transcript not found for video_id: {video_id}")
        raise HTTPException(status_code=404, detail="Transcript not found.")

    video.status = "Processing"
    db.commit()
    logger.info("Transcript translation started")
    translation_result = translate(transcript.transcript)

    translation = Translation(
        video_id=video.video_id,
        source_language=translation_result["source_language"],
        target_language=translation_result["target_language"],
        translation_model=translation_result["model"],
        translated_text=translation_result["translated_text"],
        processing_time=translation_result["processing_time"]
    )

    try:
        db.add(translation)
        video.status = "Translated"
        video.status_message = "Translation completed."
        db.commit()
        db.refresh(translation)
        logger.info(f"Translation saved successfully. translation_id={translation.translation_id}")

    except Exception as e:
        logger.error(f"Failed to save translation: {e}")
        db.rollback()
        raise

    return {
        "message": "Translation completed.",
        "translation_id": translation.translation_id,
        "translated_text": translation.translated_text,
        "processing_time": translation.processing_time
    }
