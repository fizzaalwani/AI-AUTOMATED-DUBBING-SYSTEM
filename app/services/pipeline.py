import shutil
from app.utils.logger import logger
from app.database.tables import Video
from app.database.tables import Transcript
from app.database.tables import Translation
from app.database.tables import VoiceSynthesis

from app.services.audio_extractor import extract_audio
from app.services.speech_to_text import transcribe
from app.services.translator import translate
from app.services.voice_synthesis import generate_speech

from app.services.video_generator import (
    dub_video,
    combine_audio_segments,
    get_audio_duration,
)

from app.config import UPLOAD_DIR
from app.config import OUTPUT_DIR


def process_pipeline(file, db):
    try:
        logger.info(f"Pipeline started for file: {file.filename}")
        video = Video(
            original_filename=file.filename,
            stored_filename=file.filename,
            status="Uploading",
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        logger.info(f"Video record created. Video ID: {video.video_id}")

        video_path = UPLOAD_DIR / video.stored_filename

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            logger.info(f"Video saved to {video_path}")
        video.status = "Processing"
        db.commit()

        # The whole video is processed as a single piece (no splitting).
        logger.info("Audio extraction started.")
        audio_result = extract_audio(video_path, video.video_id, OUTPUT_DIR)
        logger.info("Audio extraction completed.")
        video.audio_path = str(audio_result["audio_path"])
        db.commit()

        logger.info("Speech recognition started.")
        transcript_result = transcribe(audio_result["audio_path"])
        transcript = Transcript(
            video_id=video.video_id,
            whisper_model=transcript_result["model"],
            language=transcript_result["language"],
            transcript=transcript_result["text"],
            segments=transcript_result["segments"],
            processing_time=transcript_result["processing_time"],
        )
        db.add(transcript)
        db.commit()
        db.refresh(transcript)
        logger.info("Speech recognition completed.")

        logger.info("Translation started.")
        translation_result = translate(transcript.transcript, transcript.segments)
        translation = Translation(
            video_id=video.video_id,
            translation_model=translation_result["model"],
            source_language=translation_result["source_language"],
            target_language=translation_result["target_language"],
            translated_text=translation_result["translated_text"],
            translated_segments=translation_result["translated_segments"],
            processing_time=translation_result["processing_time"],
        )
        db.add(translation)
        db.commit()
        db.refresh(translation)
        logger.info("Translation completed.")

        logger.info("Voice synthesis started.")
        voice_result = generate_speech(
            video_id=video.video_id,
            reference_audio=video.audio_path,
            reference_text=transcript.transcript,
            reference_segments=transcript.segments,
            translated_segments=translation_result["translated_segments"],
            output_dir=OUTPUT_DIR,
        )

        timed_audio_path = OUTPUT_DIR / f"{video.video_id}_timed_dubbed.wav"

        # Timeline must match the video length so audio isn't cut short.
        video_duration = get_audio_duration(video_path)

        combine_audio_segments(
            generated_segments=voice_result["generated_segments"],
            output_audio=timed_audio_path,
            total_duration=video_duration,
        )
        voice = VoiceSynthesis(
            video_id=video.video_id,
            tts_model=voice_result["tts_model"],
            reference_audio_path=video.audio_path,
            generated_audio_path=str(timed_audio_path),
            processing_time=voice_result["processing_time"],
        )
        db.add(voice)
        db.commit()
        db.refresh(voice)
        logger.info("Voice synthesis completed.")

        logger.info("Video dubbing started.")
        dubbed_result = dub_video(video_path, timed_audio_path, OUTPUT_DIR)
        voice.dubbed_video_path = str(dubbed_result["dubbed_video"])
        voice.video_processing_time = dubbed_result["processing_time"]
        video.status = "Completed"
        db.commit()
        logger.info("Video dubbing completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline failed for {file.filename}: {e}")
        raise

    logger.info(f"Pipeline finished successfully for video {video.video_id}")
    return {
        "status": "success",
        "message": "Video dubbing completed successfully.",
        "data": {
            "video_id": str(video.video_id),
            "translation": translation.translated_text,
            "generated_audio": voice.generated_audio_path,
            "dubbed_video": voice.dubbed_video_path,
            "processing_time": round(
                transcript.processing_time
                + translation.processing_time
                + voice.processing_time
                + dubbed_result["processing_time"],
                2,
            ),
        },
    }