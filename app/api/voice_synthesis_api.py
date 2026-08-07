from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from app.utils.logger import logger

from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.tables import Video
from app.database.tables import Transcript
from app.database.tables import Translation
from app.database.tables import VoiceSynthesis

from app.config import OUTPUT_DIR
from app.services.voice_synthesis import generate_speech

router = APIRouter(prefix="/voice-generation", tags=["Voice Generation"])

@router.post("/generate")
def generate(video_id: str, db: Session = Depends(get_db)):
    logger.info(f"Voice generation request received for video_id={video_id}")
    video = db.query(Video).filter(Video.video_id == video_id).first()

    if video is None:
        logger.warning(f"Video not found: {video_id}")
        raise HTTPException( status_code=404, detail="Video not found.")

    transcript = db.query(Transcript).filter(Transcript.video_id == video_id).first()

    if transcript is None:
        logger.warning(f"Transcript not found for video_id={video_id}")
        raise HTTPException(status_code=404, detail="Transcript not found.")

    translation = db.query(Translation).filter(Translation.video_id == video_id).first()

    if translation is None:
        logger.warning(f"Translation not found for video_id={video_id}")
        raise HTTPException(status_code=404, detail="Translation not found.")

    video.status = "Processing"
    db.commit()

    logger.info("Speech generation started.")
    result = generate_speech(
    video_id=video.video_id,
    reference_audio=video.audio_path,
    reference_text=transcript.transcript,
    generation_text=translation.translated_text,
    output_dir=OUTPUT_DIR
    )

    voice = VoiceSynthesis(
    video_id=video.video_id,
    tts_model=result["tts_model"],
    reference_audio_path=video.audio_path,
    generated_audio_path=result["generated_audio"],
    processing_time=result["processing_time"]
    )

    try:
        db.add(voice)
        video.status = "VoiceGenerated"
        db.commit()
        db.refresh(voice)
        logger.info(f"Voice generation saved successfully. voice_id={voice.voice_id}")

    except Exception:
        logger.error(f"Error occurred while saving voice generation. voice_id={voice.voice_id}")
        db.rollback()
        raise

    return {
        "message": "Voice generated successfully.",
        "voice_id": voice.voice_id,
        "generated_audio": voice.generated_audio_path,
        "processing_time": voice.processing_time
    }
