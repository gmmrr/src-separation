# pyannote speech separation

這支腳本是 [pipeline/blind_separation/pyannote_speech_separation_ami_1.0.py](/Users/gmmrr/Documents/GitHub/src-separation/pipeline/blind_separation/pyannote_speech_separation_ami_1.0.py) 的說明。

它使用的是官方 `pyannote/speech-separation-ami-1.0` pipeline。
這個 pipeline 底層對應到 `pyannote/separation-ami-1.0`，但單獨的 model 版本只能切 5 秒 chunk。
如果你要直接處理整段音檔，應該用這個 pipeline 版本。

## 需要安裝哪些套件

先確認你的虛擬環境已啟用，再安裝：

```bash
uv pip install "pyannote.audio[separation]==3.3.2"
uv pip install soundfile scipy torch
```

如果你已經有 `pyannote.audio`，但 separation pipeline 載入失敗，優先先確認：
- `pyannote.audio` 版本
- Hugging Face token
- 你是否已經在 Hugging Face 上接受模型條款

## 需要先做什麼

你必須先在 Hugging Face 上接受這兩個模型的條款：
- `pyannote/separation-ami-1.0`
- `pyannote/speech-separation-ami-1.0`

然後準備 token：

```bash
export HF_TOKEN=your_huggingface_token
```

也可以直接用 `--hf-token` 傳入。

如果你已經執行過 `hf auth login`，這支腳本也會自動讀取本機快取的 token，不一定要再手動設 `HF_TOKEN`。

## 怎麼跑

單一檔案：

```bash
uv run pipeline/blind_separation/pyannote_speech_separation_ami_1.0.py \
  --input data/test_blind_separation/three_speakers.wav
```

資料夾：

```bash
uv run pipeline/blind_separation/pyannote_speech_separation_ami_1.0.py \
  --input data/test_blind_separation
```

如果要指定 token：

```bash
uv run pipeline/blind_separation/pyannote_speech_separation_ami_1.0.py \
  --input data/test_blind_separation/three_speakers.wav \
  --hf-token YOUR_HF_TOKEN
```

## 輸出會放哪裡

預設輸出目錄是：

```text
data/test_blind_separation/pyannote_speech_separation_ami_1.0/<input_stem>/
```

每個輸入檔會產生：
- `*_separation.json`
- `*.rttm`
- 每個 speaker 一個 `wav`

JSON 裡會包含：
- `segments`
- `overlaps`
- `speakers`
- `source_files`

## 注意事項

- 這支腳本會先用 `soundfile` 把音檔讀進來，再丟給 pyannote，所以可以避開 `torchcodec` 的讀檔問題。
- 預設會把音檔 resample 到 `16kHz`，因為官方 pipeline 就是吃 `16kHz` mono 音訊。
- 如果你想關掉進度提示，可以加 `--no-progress`。
