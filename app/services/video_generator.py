from pathlib import Path
import subprocess
import time

import numpy as np
import soundfile as sf

from app.utils.logger import logger

SAMPLE_RATE = 24000          # F5-TTS output rate
MAX_COMFORT_SPEEDUP = 1.35   # beyond this speech starts to sound rushed
HARD_MAX_SPEEDUP = 2.0       # absolute limit (single atempo filter)


def get_audio_duration(path):
    """Duration in seconds (works for audio and video files)."""
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def fit_segment_to_slot(input_audio, output_audio, target_duration, available_duration):
    """
    Fit one TTS clip into the speaker's time slot.

    - Shorter than the slot  -> left untouched (natural pacing, no slow-mo).
    - Longer than the slot   -> sped up gently (max 1.35x) to match the slot.
    - Still too long         -> sped up more (max 2x) so it never runs into
                                the next segment, then hard-trimmed.
    """
    current = get_audio_duration(input_audio)
    if current <= 0:
        raise RuntimeError(f"Invalid audio duration: {input_audio}")

    speed = 1.0
    if current > target_duration:
        speed = min(current / target_duration, MAX_COMFORT_SPEEDUP)

    if current / speed > available_duration:
        speed = min(current / available_duration, HARD_MAX_SPEEDUP)

    speed = max(speed, 1.0)

    logger.info(
        f"Fit segment: tts={current:.2f}s slot={target_duration:.2f}s "
        f"available={available_duration:.2f}s speed={speed:.3f}x"
    )

    command = [
        "ffmpeg", "-v", "error",
        "-i", str(input_audio),
        # trim leading/trailing silence F5-TTS adds, then change tempo
        "-af",
        "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
        "areverse,"
        "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
        "areverse,"
        f"atempo={speed:.4f}",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-t", f"{available_duration:.3f}",
        "-y", str(output_audio),
    ]
    subprocess.run(command, check=True)
    return output_audio


def combine_audio_segments(generated_segments, output_audio, total_duration=None):
    """
    Place each TTS segment at its original Whisper start time on a silent
    timeline and write one WAV the same length as the video.

    Mixing is done with numpy (plain addition), so there is no amix volume
    drop and no ffmpeg input-count limits.
    """
    output_audio = Path(output_audio)

    if not generated_segments:
        raise RuntimeError("No generated audio segments were provided.")

    segments = sorted(
        (
            {
                "index": s["index"],
                "start": float(s["start"]),
                "end": float(s["end"]),
                "audio_path": Path(s["audio_path"]),
            }
            for s in generated_segments
            if float(s["end"]) - float(s["start"]) > 0
        ),
        key=lambda s: s["start"],
    )

    if not segments:
        raise RuntimeError("No valid audio segments were available.")

    timeline_end = max(s["end"] for s in segments)
    # Keep the audio exactly as long as the video (a segment ending past
    # the video would otherwise make the audio longer than the picture).
    total_duration = float(total_duration) if total_duration else timeline_end
    segments = [s for s in segments if s["start"] < total_duration]

    temp_dir = output_audio.parent / f"{output_audio.stem}_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    timeline = np.zeros(int(total_duration * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)

    try:
        for i, seg in enumerate(segments):
            target = seg["end"] - seg["start"]

            if i + 1 < len(segments):
                available = segments[i + 1]["start"] - seg["start"]
            else:
                available = total_duration - seg["start"]
            available = max(available, target)

            fitted = temp_dir / f"fitted_{seg['index']}.wav"
            fit_segment_to_slot(seg["audio_path"], fitted, target, available)

            data, sr = sf.read(str(fitted), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            offset = int(round(seg["start"] * sr))
            end = min(offset + len(data), len(timeline))
            timeline[offset:end] += data[: end - offset]

        timeline = np.clip(timeline[: int(total_duration * SAMPLE_RATE)], -1.0, 1.0)
        sf.write(str(output_audio), timeline, SAMPLE_RATE, subtype="PCM_16")

        logger.info(f"Timed dubbed audio created: {output_audio} ({total_duration:.2f}s)")
        return output_audio

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def mix_with_background(dub_audio, background_audio, output_audio, ducked=False):
    """
    Mix the dub over the background (music / ambience).

    The background is lowered automatically whenever the dub is speaking
    (sidechain ducking) and returns to full volume between lines.

    ducked=True is used when no clean background exists (the full original
    audio, Urdu speech included, is used instead), so it is lowered much
    harder.
    """
    ratio, threshold, floor = (20, 0.01, 0.25) if ducked else (6, 0.02, 0.8)
    bg_gain = 0.35 if ducked else 0.9

    filter_graph = (
        f"[1:a]aresample=24000,aformat=channel_layouts=mono,"
        f"volume={bg_gain}[bg];"
        f"[0:a]aresample=24000,asplit=2[dub][key];"
        f"[bg][key]sidechaincompress=threshold={threshold}:ratio={ratio}:"
        f"attack=20:release=400:makeup=1[ducked];"
        f"[dub][ducked]amix=inputs=2:duration=first:normalize=0,"
        f"alimiter=limit=0.95[out]"
    )

    subprocess.run(
        ["ffmpeg", "-v", "error",
         "-i", str(dub_audio), "-i", str(background_audio),
         "-filter_complex", filter_graph, "-map", "[out]",
         "-c:a", "pcm_s16le", "-y", str(output_audio)],
        check=True,
    )
    logger.info(f"Dub mixed with background: {output_audio}")
    return output_audio


def dub_video(video_path, dubbed_audio_path, output_dir):
    """Replace the video's audio track with the synchronized dubbed audio."""
    logger.info("Dubbed video generation started.")
    start_time = time.time()

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{video_path.stem}_dubbed.mp4"

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(dubbed_audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-y", str(output_video),
    ]

    try:
        subprocess.run(command, check=True)
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Dubbed video generated: {output_video} ({processing_time}s)")
    except subprocess.CalledProcessError as e:
        logger.error(f"Video dubbing failed: {e}")
        raise

    return {"dubbed_video": output_video, "processing_time": processing_time}