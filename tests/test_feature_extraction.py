import pytest
import numpy as np

import soundfile as sf

from desc import feature_extraction


@pytest.fixture
def sample_rate():
    return 44100


@pytest.fixture
def duration(request):
    return getattr(request, "param", 1)


@pytest.fixture
def wave_form(sample_rate, duration):
    return np.zeros(sample_rate * duration)


@pytest.fixture
def hop_length():
    return 15400


@pytest.fixture
def wav_path(tmp_path, sample_rate, wave_form):
    path = tmp_path.with_suffix(".wav")
    sf.write(path, wave_form, sample_rate)
    return path


@pytest.mark.parametrize("duration", [1, 2], indirect=True)
def test_load_wav(wav_path, wave_form, sample_rate, duration):
    y, sr = feature_extraction.load_wav(wav_path)
    assert sr == sample_rate
    assert np.allclose(y, wave_form)


@pytest.fixture
def extract_fun(request):
    return request.param


@pytest.mark.parametrize(
    "extract_fun",
    [
        feature_extraction.centroid,
        feature_extraction.flatness,
        feature_extraction.rolloff,
        feature_extraction.rms,
        feature_extraction.f0,
    ],
)
def test_extractor(extract_fun, wave_form, sample_rate, hop_length):
    feat = extract_fun(wave_form, sample_rate, hop_length)
    assert feat.shape == (3, 1)


def test_extract(wave_form, sample_rate, hop_length):
    feats = feature_extraction.extract(wave_form, sample_rate, hop_length)
    assert feats.shape == (3, 5)


def test_extract_centroid(wave_form, sample_rate, hop_length):
    feat = feature_extraction.centroid(wave_form, sample_rate, hop_length)
    assert np.allclose(feat, [[0, 0, 0]])
