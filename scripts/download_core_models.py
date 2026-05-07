from pyannote.audio import Pipeline
from speechbrain.inference.speaker import EncoderClassifier
from silero_vad import load_silero_vad

print("Downloading SpeechBrain ECAPA...")
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/hf/speechbrain_ecapa",
)

print("Downloading Silero VAD...")
load_silero_vad()

print("Downloading pyannote speaker diarization...")
Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=True,
)

print("Done.")
