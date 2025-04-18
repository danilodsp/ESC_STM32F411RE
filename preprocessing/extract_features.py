"""
Feature extraction utilities for audio data.

This module provides functions to extract log-mel spectrogram features
from audio files and process directories of audio data.
"""
import os
import logging
import numpy as np
import librosa

SR: int = 16000
N_MELS: int = 40
DURATION: float = 1.0  # seconds

logging.basicConfig(level=logging.INFO)


def extract_logmel(
    audio_path: str,
    sr: int = SR,
    n_mels: int = N_MELS,
    duration: float = DURATION
) -> np.ndarray:
    """
    Extract log-mel spectrogram features from an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sampling rate.
        n_mels (int): Number of mel bands.
        duration (float): Duration to which audio is padded/truncated (in seconds).

    Returns:
        np.ndarray: Log-mel spectrogram (frames x n_mels).
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        y = librosa.util.fix_length(y, int(sr * duration))
        mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
        logmel = librosa.power_to_db(mel).T
        return logmel
    except Exception as e:
        logging.error(f"Failed to process {audio_path}: {e}")
        raise


def process_directory(
    base_dir: str,
    out_dir: str,
    sr: int = SR,
    n_mels: int = N_MELS,
    duration: float = DURATION
) -> None:
    """
    Process a directory of audio files, extracting features and saving
    them as .npy files.

    Args:
        base_dir (str): Directory containing audio files (organized by
            label subfolders).
        out_dir (str): Output directory for .npy feature files.
        sr (int): Sampling rate for feature extraction.
        n_mels (int): Number of mel bands.
        duration (float): Duration for each audio sample.
    """
    os.makedirs(out_dir, exist_ok=True)
    for subdir, _, files in os.walk(base_dir):
        label = os.path.basename(subdir)
        for file in files:
            if file.lower().endswith('.wav'):
                path = os.path.join(subdir, file)
                try:
                    feature = extract_logmel(
                        path,
                        sr=sr,
                        n_mels=n_mels,
                        duration=duration
                    )
                    out_path = os.path.join(
                        out_dir,
                        f"{label}_{os.path.splitext(file)[0]}.npy"
                    )
                    np.save(out_path, feature)
                    logging.info(
                        f"Saved features for {file} to {out_path}"
                    )
                except Exception:
                    continue


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract log-mel features from audio dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with audio files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for features."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=SR,
        help="Sampling rate."
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=N_MELS,
        help="Number of mel bands."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION,
        help="Duration in seconds."
    )
    args = parser.parse_args()
    process_directory(
        args.input_dir,
        args.output_dir,
        sr=args.sr,
        n_mels=args.n_mels,
        duration=args.duration
    )


if __name__ == "__main__":
    main()
