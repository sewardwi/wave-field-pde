"""
datasets — dataset loaders for wave field experiments.

SC09 (audio, prior work) is imported lazily so the PDE modules don't pull in
audio-only deps (soundfile/torchaudio). Import it explicitly when needed:
    from datasets.sc09 import SC09
"""

__all__ = ["SC09", "TARGET_SR", "TARGET_LEN"]


def __getattr__(name):
    # Lazy re-export of the audio loader (PEP 562), so `datasets.SC09` still
    # works but only triggers the soundfile/torchaudio import on first access.
    if name in __all__:
        from . import sc09
        return getattr(sc09, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
