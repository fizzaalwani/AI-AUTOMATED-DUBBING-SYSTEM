from pathlib import Path
import subprocess
import shutil
import time
from app.utils.logger import logger

MODEL_NAME = "F5TTS_v1_Base"

# F5-TTS works best with a clean 5-12 second reference clip whose text
# matches the audio exactly. Passing the whole video audio with the whole
# transcript makes it clip the audio and mismatch the text.
REF_MIN_SECONDS = 5.0
REF_MAX_SECONDS = 12.0


def build_reference_clip(reference_audio, reference_segments, out_path):
    """
    Cut a short reference clip (5-12s) from the original audio and return
    (clip_path, matching_text). Picks the first run of consecutive Whisper
    segments that fits the window.
    """
    chosen = []
    total = 0.0

    for seg in reference_segments:
        dur = float(seg["end"]) - float(seg["start"])
        if dur <= 0:
            continue

        if not chosen:
            chosen = [seg]
            total = dur
        elif total + dur <= REF_MAX_SECONDS:
            chosen.append(seg)
            total += dur
        else:
            break

        if total >= REF_MIN_SECONDS:
            break

    if not chosen:
        raise RuntimeError("No usable segments for reference audio.")

    start = float(chosen[0]["start"])
    end = min(float(chosen[-1]["end"]), start + REF_MAX_SECONDS)
    text = " ".join(s["text"].strip() for s in chosen)

    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{end - start:.3f}",
            "-i", str(reference_audio),
            "-ac", "1",
            "-y", str(out_path),
        ],
        check=True,
    )

    logger.info(f"Reference clip: {start:.2f}s-{end:.2f}s ({end - start:.1f}s)")
    return out_path, text


def generate_speech(
    video_id,
    reference_audio,
    reference_text,
    reference_segments,
    translated_segments,
    output_dir,
):
    logger.info(f"Starting voice generation for video_id={video_id}")
    start_time = time.time()

    cli_path = shutil.which("f5-tts_infer-cli")
    if cli_path is None:
        raise RuntimeError("f5-tts_infer-cli was not found in PATH.")

    output_dir = Path(output_dir)
    segments_dir = output_dir / f"{video_id}_voice_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    ref_clip, ref_text = build_reference_clip(
        reference_audio,
        reference_segments,
        segments_dir / "reference.wav",
    )

    generated_segments = []

    for index, segment in enumerate(translated_segments):
        translated_text = segment["text"].strip()
        if not translated_text:
            continue

        segment_output = segments_dir / f"segment_{index}.wav"

        logger.info(f"Generating voice for segment {index}: {translated_text}")

        command = [
            cli_path,
            "--model", MODEL_NAME,
            "--ref_audio", str(ref_clip),
            "--ref_text", ref_text,
            "--gen_text", translated_text,
            "--output_file", str(segment_output),
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Voice generation failed for segment {index}: {e}")
            raise RuntimeError(f"Voice generation failed for segment {index}: {e}")

        generated_segments.append({
            "index": index,
            "start": segment["start"],
            "end": segment["end"],
            "text": translated_text,
            "audio_path": str(segment_output),
        })

    processing_time = round(time.time() - start_time, 2)
    logger.info(f"Voice generation completed successfully ({processing_time}s)")

    return {
        "generated_segments": generated_segments,
        "tts_model": MODEL_NAME,
        "processing_time": processing_time,
    }