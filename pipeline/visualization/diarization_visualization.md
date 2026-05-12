# diarization visualization

This document matches [pipeline/visualization/diarization_visualization.py](/Users/gmmrr/Documents/GitHub/src-separation/pipeline/visualization/diarization_visualization.py).

This viewer is organized as:

- a source-audio dropdown at the top
- a vertical stack of all JSON results for that source
- one waveform panel per JSON, with colored speaker time windows
- speaker buttons and an "active speakers" label under each waveform
- shared playback controls at the bottom

## Install

Activate your environment, then install the required packages:

```bash
uv pip install pyside6 pyqtgraph numpy librosa soundfile
```

If Qt Multimedia is already available on your system, the viewer should run directly.

## JSON format

The viewer recursively scans `data/test_diarization` and groups JSON files by the `input` field inside each file.

Each diarization JSON should contain at least:

```json
{
  "input": "/path/to/audio.wav",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.2},
    {"speaker": "SPEAKER_01", "start": 2.1, "end": 5.0}
  ]
}
```

If `speakers` is missing, the tool derives speaker IDs from `segments`.

## Default usage

Run it directly to scan `data/test_diarization`:

```bash
uv run pipeline/visualization/diarization_visualization.py
```

To scan another root directory:

```bash
uv run pipeline/visualization/diarization_visualization.py \
  --json-root data/test_diarization
```

To force a specific source audio:

```bash
uv run pipeline/visualization/diarization_visualization.py \
  --media data/test_diarization/IyLqUS7hRvo_std_vocals.wav \
  --json-root data/test_diarization
```

## UI behavior

- The dropdown selects a source audio file.
- All JSON files for that source are shown together.
- Each JSON panel shows:
  - waveform
  - colored time windows
  - speaker buttons
  - active speaker count and names
- The bottom playback bar controls the entire source audio.

## Options

- `--json-root`: root directory to scan recursively, default `data/test_diarization`
- `--media`: source audio to display
- `--waveform-sample-rate`: waveform display sample rate, default `16000`
- `--waveform-points`: maximum number of waveform points to keep, default `5000`

## Interaction

- Click the waveform to seek.
- Use the bottom Play / Pause button to control playback.
- Drag the bottom slider to seek.
- Click a speaker button to jump to that speaker's first segment.
- Overlaps are highlighted together.

## Notes

- This viewer is for comparing multiple diarization results from the same source audio.
- If you want to compare different models, make sure they all output JSON for the same media.
- If waveform loading fails, playback may still work, but the waveform will not be shown.
