"""Turn an uploaded clip into 16 kHz mono PCM."""

from __future__ import annotations

import io
import wave

import numpy as np


class BadAudio(Exception):
    pass


def pcm_from_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise BadAudio("The recording was empty")
    if data[:4] == b"RIFF":
        pcm, rate, channels, width = _wav_parts(data)
        if rate == 16000 and channels == 1 and width == 2:
            if pcm.size == 0:
                raise BadAudio("The recording was empty")
            return pcm
    pcm = _decode_with_av(data)
    if pcm.size == 0:
        raise BadAudio("The recording was empty")
    return pcm


def _wav_parts(data: bytes) -> tuple[np.ndarray, int, int, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise BadAudio("That audio could not be read") from exc
    if width != 2:
        return np.zeros(0, dtype=np.int16), rate, channels, width
    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
        channels = 1
    return pcm.copy(), rate, channels, width


def _decode_with_av(data: bytes) -> np.ndarray:
    try:
        import av
    except ImportError as exc:
        raise BadAudio("That audio could not be read") from exc
    try:
        container = av.open(io.BytesIO(data))
    except av.FFmpegError as exc:
        raise BadAudio("That audio could not be read") from exc
    stream = next((item for item in container.streams if item.type == "audio"), None)
    if stream is None:
        raise BadAudio("That audio could not be read")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    pieces: list[np.ndarray] = []
    for frame in container.decode(stream):
        _collect(resampler.resample(frame), pieces)
    _collect(resampler.resample(None), pieces)
    if not pieces:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(pieces).astype(np.int16, copy=False)


def _collect(frames, pieces: list[np.ndarray]) -> None:
    if frames is None:
        return
    if not isinstance(frames, list):
        frames = [frames]
    for frame in frames:
        if frame is None:
            continue
        pieces.append(np.asarray(frame.to_ndarray()).reshape(-1))
