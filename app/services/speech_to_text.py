# import re
# import time
# from faster_whisper import WhisperModel
# from app.utils.logger import logger

# model = None

# # Whisper often "hears" these in music or silence.
# HALLUCINATIONS = re.compile(
#     r"^\W*(music|موسیقی|\[music\]|\(music\)|thank you|subscribe)\W*$",
#     re.IGNORECASE,
# )


# def get_model():
#     global model
#     if model is None:
#         model = WhisperModel("large-v3", device="cpu", compute_type="int8")
#         logger.info("Whisper model loaded successfully.")
#     return model


# def transcribe(audio_path, video_duration=None):
#     try:
#         whisper = get_model()
#         logger.info(f"Transcription started for {audio_path}")
#         start_time = time.time()

#         def run(use_vad):
#             segs, inf = whisper.transcribe(
#                 str(audio_path),
#                 language="ur",
#                 vad_filter=use_vad,
#                 vad_parameters={"min_silence_duration_ms": 400,
#                                 "threshold": 0.3},
#                 condition_on_previous_text=False,
#                 no_speech_threshold=0.6,
#             )
#             kept = []
#             for segment in segs:
#                 text = segment.text.strip()
#                 logger.info(
#                     f"Whisper segment {segment.start:.2f}-{segment.end:.2f} "
#                     f"no_speech={segment.no_speech_prob:.2f}: {text!r}"
#                 )
#                 if not text or HALLUCINATIONS.match(text):
#                     continue
#                 if segment.no_speech_prob > 0.8:
#                     continue
#                 start = float(segment.start)
#                 end = float(segment.end)
#                 if video_duration:
#                     end = min(end, float(video_duration))
#                 if end <= start:
#                     continue
#                 kept.append((text, start, end, segment.avg_logprob))
#             return kept, inf

#         kept, info = run(True)
#         if not kept:
#             logger.warning("No speech found with VAD; retrying without VAD.")
#             kept, info = run(False)
#         if not kept:
#             raise RuntimeError(
#                 "No speech was detected in the audio. If this clip has "
#                 "dialogue, check the vocal separation output "
#                 "(<video_id>_separation/vocals_16k.wav)."
#             )

#         transcript_parts = [k[0] for k in kept]
#         timestamps = [{"start": k[1], "end": k[2], "text": k[0]} for k in kept]
#         confidence_scores = [k[3] for k in kept]

#         transcript = " ".join(transcript_parts)
#         processing_time = round(time.time() - start_time, 2)
#         average_confidence = (
#             round(sum(confidence_scores) / len(confidence_scores), 3)
#             if confidence_scores else None
#         )

#         logger.info(
#             f"Transcript generated successfully. Language={info.language}, "
#             f"Segments={len(timestamps)}, Confidence={average_confidence}, "
#             f"Processing Time={processing_time}s"
#         )

#     except Exception as e:
#         logger.error(f"Whisper transcription failed: {e}")
#         raise

#     return {
#         "text": transcript,
#         "language": info.language,
#         "segments": timestamps,
#         "processing_time": processing_time,
#         "model": "large-v3",
#     }
import re
import time

import numpy as np
import soundfile as sf
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


MAX_PIECE = 8.0      # split segments longer than this
PAUSE = 0.6          # ...at pauses at least this long
MIN_PIECE = 1.5


def split_at_pauses(segment, text):
    """Use word timestamps for the true speech start/end, and break long
    segments at natural pauses so each English line is timed separately."""
    words = getattr(segment, "words", None) or []
    if not words:
        return [(float(segment.start), float(segment.end), text, 1.0)]

    pieces, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        cur_len = prev.end - cur[0].start
        if (w.start - prev.end >= PAUSE and cur_len >= MIN_PIECE) or \
           cur_len >= MAX_PIECE:
            pieces.append(cur)
            cur = [w]
        else:
            cur.append(w)
    pieces.append(cur)

    out = []
    for p in pieces:
        start, end = float(p[0].start), float(p[-1].end)
        piece_text = "".join(w.word for w in p).strip()
        # a couple of "words" stretched over seconds = music/noise, not speech
        if (end - start) > 2.0 and len(p) / (end - start) < 0.6:
            logger.info(f"Dropped sparse segment: {piece_text!r}")
            continue
        prob = sum(w.probability for w in p) / len(p)
        out.append((start, end, piece_text, prob))
    return out


def load_audio(path):
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def db(x):
    return 20 * np.log10(x + 1e-9)


def refine_with_energy(audio, sr, start, end, floor_db):
    """Return (start, end) trimmed to where the vocals are really audible,
    or None if the whole piece is quiet (music / noise, not speech)."""
    frame = int(sr * 0.02)
    a, b = int(start * sr), int(end * sr)
    seg = audio[a:b]
    n = len(seg) // frame
    if n == 0:
        return None
    levels = db(np.sqrt((seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1)))
    loud = np.where(levels > floor_db)[0]
    if len(loud) == 0:
        return None
    new_start = start + loud[0] * 0.02
    new_end = start + (loud[-1] + 1) * 0.02
    return max(new_start - 0.05, start), min(new_end + 0.1, end)


def transcribe(audio_path, video_duration=None):
    try:
        whisper = get_model()
        logger.info(f"Transcription started for {audio_path}")
        start_time = time.time()

        audio, sr = load_audio(audio_path)

        def run(use_vad):
            segs, inf = whisper.transcribe(
                str(audio_path),
                language="ur",
                vad_filter=use_vad,
                vad_parameters={"min_silence_duration_ms": 400,
                                "threshold": 0.3},
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                word_timestamps=True,   # real speech onset / offset
            )
            kept = []
            candidates = []
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
                for start, end, piece_text, prob in split_at_pauses(segment, text):
                    if video_duration:
                        end = min(end, float(video_duration))
                    if end <= start:
                        continue
                    if prob < 0.35:
                        logger.info(f"Dropped low-confidence piece "
                                    f"({prob:.2f}): {piece_text!r}")
                        continue
                    candidates.append(
                        (piece_text, start, end, segment.avg_logprob))

            # Energy check on the separated vocals: a real line is within
            # ~25 dB of the loudest line; music / noise is far quieter.
            # Also trims each line to where the voice really starts and stops.
            if candidates:
                peaks = []
                for t, st, en, lp in candidates:
                    seg_audio = audio[int(st * sr):int(en * sr)]
                    peaks.append(db(np.sqrt((seg_audio ** 2).mean()))
                                 if len(seg_audio) else -120)
                loudest = max(peaks)
                for (t, st, en, lp), level in zip(candidates, peaks):
                    if level < loudest - 25:
                        logger.info(f"Dropped quiet piece "
                                    f"({level:.0f} dB): {t!r}")
                        continue
                    refined = refine_with_energy(
                        audio, sr, st, en, floor_db=loudest - 35)
                    if refined is None:
                        logger.info(f"Dropped silent piece: {t!r}")
                        continue
                    if abs(refined[0] - st) > 0.2:
                        logger.info(f"Start moved {st:.2f}s -> "
                                    f"{refined[0]:.2f}s for {t!r}")
                    kept.append((t, refined[0], refined[1], lp))
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