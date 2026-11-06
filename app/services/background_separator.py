from pathlib import Path
import subprocess
import sys

from app.utils.logger import logger


def separate_vocals(video_path, video_id, output_dir):
    """
    Split a video's audio into vocals and background (music / ambience)
    using Demucs, so the background can be kept under the dub.

    Returns:
        {
          "vocals_16k": mono 16 kHz vocals for Whisper / voice reference,
          "background": stereo 44.1 kHz music + effects (None if Demucs
                        is unavailable),
          "full_audio": the original audio (fallback background),
        }
    """
    output_dir = Path(output_dir)
    work = output_dir / f"{video_id}_separation"
    work.mkdir(parents=True, exist_ok=True)

    full_audio = work / "full_44k.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video_path),
         "-vn", "-ac", "2", "-ar", "44100", "-y", str(full_audio)],
        check=True,
    )

    try:
        logger.info("Separating vocals from background with Demucs...")
        subprocess.run(
            [sys.executable, "-m", "demucs.separate",
             "-n", "htdemucs", "--two-stems=vocals", "--device", "cpu",
             "-o", str(work), str(full_audio)],
            check=True,
        )
        stem_dir = work / "htdemucs" / full_audio.stem
        vocals = stem_dir / "vocals.wav"
        background = stem_dir / "no_vocals.wav"
        if not vocals.exists() or not background.exists():
            raise RuntimeError("Demucs finished but output files are missing.")
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
        logger.warning(
            f"Demucs separation unavailable ({e}). Falling back to the "
            f"original audio, ducked under the dub. "
            f"Install with: pip install demucs"
        )
        vocals = full_audio
        background = None

    vocals_16k = work / "vocals_16k.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(vocals),
         "-ac", "1", "-ar", "16000", "-y", str(vocals_16k)],
        check=True,
    )

    logger.info("Vocal separation completed.")
    return {
        "vocals_16k": vocals_16k,
        "background": background,
        "full_audio": full_audio,
    }