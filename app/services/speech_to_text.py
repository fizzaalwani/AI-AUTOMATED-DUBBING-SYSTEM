import re
import time
from faster_whisper import WhisperModel
from app.utils.logger import logger

model = None

# Whisper often "hears" these in music or silence.
HALLUCINATIONS = re.compile(
    r"^\W*(music|موسیقی|\[music\]|\(music\)|thank you|subscribe)\W*$",
    re.IGNORECASE,
)


def get_model():
    global model
    if model is None:
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully.")
    return model


def transcribe(audio_path, video_duration=None):
    try:
        whisper = get_model()
        logger.info(f"Transcription started for {audio_path}")
        start_time = time.time()

        def run(use_vad):
            segs, inf = whisper.transcribe(
                str(audio_path),
                language="ur",
                vad_filter=use_vad,
                vad_parameters={"min_silence_duration_ms": 400,
                                "threshold": 0.3},
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            kept = []
            for segment in segs:
                text = segment.text.strip()
                logger.info(
                    f"Whisper segment {segment.start:.2f}-{segment.end:.2f} "
                    f"no_speech={segment.no_speech_prob:.2f}: {text!r}"
                )
                if not text or HALLUCINATIONS.match(text):
                    continue
                if segment.no_speech_prob > 0.8:
                    continue
                start = float(segment.start)
                end = float(segment.end)
                if video_duration:
                    end = min(end, float(video_duration))
                if end <= start:
                    continue
                kept.append((text, start, end, segment.avg_logprob))
            return kept, inf

        kept, info = run(True)
        if not kept:
            logger.warning("No speech found with VAD; retrying without VAD.")
            kept, info = run(False)
        if not kept:
            raise RuntimeError(
                "No speech was detected in the audio. If this clip has "
                "dialogue, check the vocal separation output "
                "(<video_id>_separation/vocals_16k.wav)."
            )

        transcript_parts = [k[0] for k in kept]
        timestamps = [{"start": k[1], "end": k[2], "text": k[0]} for k in kept]
        confidence_scores = [k[3] for k in kept]

        transcript = " ".join(transcript_parts)
        processing_time = round(time.time() - start_time, 2)
        average_confidence = (
            round(sum(confidence_scores) / len(confidence_scores), 3)
            if confidence_scores else None
        )

        logger.info(
            f"Transcript generated successfully. Language={info.language}, "
            f"Segments={len(timestamps)}, Confidence={average_confidence}, "
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
        "model": "large-v3",
    }