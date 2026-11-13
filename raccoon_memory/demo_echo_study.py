"""Offline echo experiment, deliberately not connected to the live listener.

It assumes a known playback waveform and a calibration interval containing
playback only. Real speakers, clocks, rooms and concurrent voices have not
been validated; the current app does not expose the required reference signal.
"""
import numpy as np
from scipy import signal


def cancel_calibrated_echo(reference, microphone, *, rate=16000, calibration_samples=19200, max_delay_ms=250):
    ref = np.asarray(reference, dtype=np.float64)
    mic = np.asarray(microphone, dtype=np.float64)
    if (ref.ndim != 1 or mic.ndim != 1 or len(ref) != len(mic) or not np.isfinite(ref).all()
            or not np.isfinite(mic).all() or rate <= 0):
        raise ValueError('equal-length finite mono audio is required')
    end = int(calibration_samples)
    max_lag = int(rate * max_delay_ms / 1000)
    if end > len(ref) or end < max_lag + rate // 4:
        raise ValueError('insufficient calibration interval')
    reference_part, input_part = ref[:end], mic[:end]
    corr = signal.correlate(input_part, reference_part, mode='full', method='fft')
    lags = signal.correlation_lags(len(input_part), len(reference_part), mode='full')
    selected = (lags >= 0) & (lags <= max_lag)
    candidates = corr[selected]
    lag = int(lags[selected][np.argmax(np.abs(candidates))])
    x, y = reference_part[:end - lag], input_part[lag:end]
    energy = float(np.dot(x, x))
    if energy < 1e-8 or float(np.dot(y, y)) < 1e-8:
        return {'status': 'unreliable', 'reason': 'insufficient reference energy', 'audio': None}
    gain = float(np.dot(x, y) / energy)
    coherence = float(abs(np.dot(x, y)) / np.sqrt(energy * np.dot(y, y)))
    if coherence < .35 or abs(gain) > 2:
        return {'status': 'unreliable', 'reason': 'reference did not correlate reliably', 'audio': None,
                'coherence': coherence, 'delay_samples': lag}
    aligned = np.zeros_like(ref)
    aligned[lag:] = ref[:len(ref) - lag] if lag else ref
    return {'status': 'calibrated_experiment', 'audio': (mic - gain * aligned).astype(np.float32),
            'delay_samples': lag, 'gain': gain, 'coherence': coherence}
