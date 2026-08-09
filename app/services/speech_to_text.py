import time
from faster_whisper import WhisperModel
from app.utils.logger import logger

model = None

def get_model():
    global model

    if model is None:
        model = WhisperModel(
            "large-v3",
            device="cpu",
            compute_type="int8"
        )
        logger.info("Whisper model loaded successfully.")
    return model

def transcribe(audio_path):
    try:
        whisper = get_model()
        logger.info(f"Transcription started for {audio_path}")
        
        start_time = time.time()
        segments, info = whisper.transcribe(str(audio_path), language="ur")
        logger.info("Whisper inference completed.")


        transcript = ""
        timestamps = []
        confidence_scores = []

        for segment in segments:
            transcript += segment.text + " "
            timestamps.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

            if hasattr(segment, "avg_logprob"):
                confidence_scores.append(segment.avg_logprob)

        transcript = transcript.strip()
        processing_time = round(time.time() - start_time, 2)
        average_confidence = None

        if confidence_scores:
            average_confidence = round(sum(confidence_scores) / len(confidence_scores), 3)

        logger.info(
        f"Transcript generated successfully. "
        f"Language={info.language}, "
        f"Segments={len(timestamps)}, "
        f"Processing Time={processing_time}s"
        )

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise
    
    return {
        "text": transcript,
        "language": info.language,
        "segments": timestamps,
        "processing_time": processing_time,
        "model": "larger-v3"
    }