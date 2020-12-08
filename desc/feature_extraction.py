from pathlib import Path
from typing import Tuple

from librosa import load, feature, yin
import numpy as np


def load_wav(path: Path) -> Tuple[np.array, int]:
    return load(path, sr=None)


def centroid(wave_form, sample_rate, hop_length):
    return feature.spectral_centroid(
        y=wave_form, sr=sample_rate, hop_length=hop_length
    ).T


def flatness(wave_form, sample_rate, hop_length):
    return feature.spectral_flatness(y=wave_form, hop_length=hop_length).T


def rolloff(wave_form, sample_rate, hop_length):
    return feature.spectral_rolloff(
        y=wave_form, sr=sample_rate, hop_length=hop_length
    ).T


def rms(wave_form, sample_rate, hop_length):
    return feature.rms(y=wave_form, hop_length=hop_length).T


def f0(wave_form, sample_rate, hop_length, fmin=80, fmax=10000):
    return yin(wave_form, fmin=fmin, fmax=fmax, hop_length=hop_length).reshape(-1, 1)


def extract(wave_form, sample_rate, hop_length):
    return np.hstack(
        [
            extract_fun(wave_form, sample_rate, hop_length)
            for extract_fun in [centroid, flatness, rolloff, rms, f0]
        ]
    )
