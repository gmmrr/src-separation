# usef tfgridnet

This document matches [pipeline/extraction/usef_tfgridnet.py](/Users/gmmrr/Documents/GitHub/src-separation/pipeline/extraction/usef_tfgridnet.py).

This pipeline runs inference only with the official USEF-TFGridNet checkpoint.
It does not train or fine-tune anything.

## What it does

- reads a mix audio file or a directory of mix files
- reads an auxiliary target-speaker reference file or directory
- runs the USEF-TFGridNet checkpoint from `models/checkpoints/usef_tse/ZBang_USEF-TSE/chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar`
- writes one separated WAV and one JSON metadata file per input

## Install

The script relies on the same core audio stack as `usef_sepformer.py`:

```bash
uv pip install torch soundfile librosa numpy scipy
```

You also need the USEF-TSE repository under:

```text
models/external_repos/USEF-TSE
```

## Default paths

- mix: `data/raw/test_samples`
- aux: `data/test_extraction/IyLqUS7hRvo_std_vocals.wav`
- output: `data/test_extraction/usef_tfgridnet`
- sample rate: `8000`

## Example usage

Run the default demo:

```bash
uv run pipeline/extraction/usef_tfgridnet.py
```

Run with explicit inputs:

```bash
uv run pipeline/extraction/usef_tfgridnet.py \
  --mix data/raw/test_samples \
  --aux data/test_extraction/IyLqUS7hRvo_std_vocals.wav \
  --output-dir data/test_extraction/usef_tfgridnet
```

Run a single mix/aux pair:

```bash
uv run pipeline/extraction/usef_tfgridnet.py \
  --mix data/raw/test_samples/example.wav \
  --aux data/test_extraction/IyLqUS7hRvo_std_vocals.wav
```

## Output

For each input file, the script writes:

- `*.wav`: the separated waveform
- `*.json`: metadata including mix path, aux path, checkpoint path, sample rate, and duration

The output file stem follows the same pattern as `usef_sepformer.py`, with the TFGridNet suffix:

- `*_usef_tfgridnet.wav`
- `*_usef_tfgridnet.json`

## Notes

- If `--mix` is a directory, every audio file in it is processed.
- If `--aux` is a directory, the script tries to match the auxiliary file name to the mix file stem.
- `--device auto` uses CUDA if available, otherwise CPU.
- If an output file already exists, the script skips it unless you pass `--force`.
