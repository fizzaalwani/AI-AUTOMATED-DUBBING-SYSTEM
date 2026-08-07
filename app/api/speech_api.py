from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.config import OUTPUT_DIR
from app.config import UPLOAD_DIR

from app.database.database import get_db
from app.database.tables import Transcript
from app.database.tables import Video

from app.services.audio_extractor import extract_audio
from app.services.speech_to_text import transcribe
from app.utils.logger import logger

router = APIRouter(prefix="/speech", tags=["Speech To Text"])

@router.post("/transcribe")
def transcribe_video(video_id: str, db: Session = Depends(get_db)):
    logger.info(f"Speech transcription requested for video_id={video_id}")

    video = db.query(Video).filter(Video.video_id == video_id).first()

    if video is None:
        logger.warning(f"Video not found. video_id={video_id}")
        raise HTTPException(status_code=404, detail="Video not found.")
    
    logger.info(f"Video found: {video.original_filename}")
    video.status = "Processing"

    db.commit()
    video_path = UPLOAD_DIR / video.stored_filename

    logger.info("Audio extraction started")
    audio_result = extract_audio(video_path, video.video_id, OUTPUT_DIR)

    video.audio_path = str(audio_result["audio_path"])
    db.commit()
    logger.info(f"Audio extracted successfully: {audio_result['audio_path']}")

    logger.info("Whisper transcription started")
    transcript_result = transcribe(audio_result["audio_path"])
    logger.info(
    f"Transcription completed. Language={transcript_result['language']} "
    f"Time={transcript_result['processing_time']} sec"
    )

    transcript = Transcript(
        video_id=video.video_id,
        whisper_model=transcript_result["model"],
        language=transcript_result["language"],
        transcript=transcript_result["text"],
        segments=transcript_result["segments"],
        processing_time=transcript_result["processing_time"]
    )

    try:
        db.add(transcript)
        video.status = "Transcribed"
        db.commit()
        db.refresh(transcript)
        logger.info(f"Transcript saved successfully. transcript_id={transcript.transcript_id}")
        
    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save transcript.")

    return {
        "message": "Speech transcription completed.",
        "transcript_id": transcript.transcript_id,
        "language": transcript.language,
        "transcript": transcript.transcript,
        "processing_time": transcript.processing_time,
    }