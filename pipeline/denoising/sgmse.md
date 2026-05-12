# SGMSE Denoising

這份說明對應目前的 [`pipeline/denoising/sgmse.py`](./sgmse.py)。

這支腳本是**直接推論版**：
- 會讀取 `models/external_repos/sgmse` 裡的 SGMSE 原始碼
- 會載入 `models/checkpoints/sgmse/ears_wham.ckpt`
- 不會訓練，也不會重新 fine-tune
- 會把輸入音訊做 denoising 後輸出 `.wav` 和 `.json`

## 1. 安裝套件

先確認你目前環境裡有以下套件。

### 必要套件

```bash
uv pip install librosa soundfile scipy numpy tqdm torch torchaudio pytorch-lightning torch-ema
```

### SGMSE 常見相依

SGMSE 的 checkpoint 和本地 repo 通常還會碰到這些套件：

```bash
uv pip install pesq pystoi torch-pesq sdeint tensorboard h5py pandas matplotlib
```

### 說明

- `librosa` / `soundfile` / `scipy`：用來讀音檔和做重採樣
- `torch-ema`：checkpoint 載入時常會用到 EMA 狀態
- `pesq` / `pystoi` / `torch-pesq` / `sdeint`：SGMSE repo 內部常見依賴

如果你之前已經為其他 pipeline 裝過這些套件，通常可以直接沿用。

## 2. 輸入資料怎麼放

目前這支腳本支援兩種輸入：

### 單一檔案

例如你的檔案是：

```text
data/test_denoising/denoising.wav
```

可以直接這樣跑：

```bash
uv run pipeline/denoising/sgmse.py --input data/test_denoising/denoising.wav
```

### 整個資料夾

如果你把很多音檔放在同一個資料夾，例如：

```text
data/test_denoising/
  denoising.wav
  other.wav
  nested/
    more.wav
```

可以直接把資料夾丟進去：

```bash
uv run pipeline/denoising/sgmse.py --input data/test_denoising
```

腳本會遞迴找出裡面的音檔並逐一處理。

### 支援格式

目前支援：
- `.wav`
- `.flac`
- `.mp3`
- `.m4a`
- `.ogg`

## 3. `denoising.wav` 要怎麼處理

你現在的測試檔是：

```text
data/test_denoising/denoising.wav
```

最簡單的做法是直接把它當單一檔案丟給腳本：

```bash
uv run pipeline/denoising/sgmse.py \
  --input data/test_denoising/denoising.wav \
  --output-dir data/test_denoising/sgmse
```

這支腳本會自動做兩件事：

1. 讀取音檔
2. 自動重採樣成模型使用的 sample rate

所以你**不需要先手動 resample**。

如果你的音檔是 stereo，腳本會先轉成 mono 再做推論。

## 4. 輸出結果在哪裡

如果輸入是：

```text
data/test_denoising/denoising.wav
```

預設輸出會是：

```text
data/test_denoising/sgmse/denoising/denoising_sgmse.wav
data/test_denoising/sgmse/denoising/denoising_sgmse.json
```

如果你是直接丟一個資料夾，輸出會保留相對路徑結構，每個檔案都會各自輸出成一個子資料夾。

JSON 會記錄：
- 輸入檔案
- checkpoint 路徑
- 裝置 `cpu` / `cuda`
- sample rate
- 模型參數摘要
- sampler 參數，例如 `N`、`snr`、`corrector`

## 5. 直接執行範例

### 預設跑法

```bash
uv run pipeline/denoising/sgmse.py
```

### 指定單一檔案

```bash
uv run pipeline/denoising/sgmse.py \
  --input data/test_denoising/denoising.wav \
  --output-dir data/test_denoising/sgmse
```

### 指定 checkpoint

如果你想改用別的 SGMSE checkpoint，可以覆蓋 `--checkpoint`：

```bash
uv run pipeline/denoising/sgmse.py \
  --input data/test_denoising/denoising.wav \
  --checkpoint models/checkpoints/sgmse/ears_wham.ckpt
```

### 強制覆寫輸出

如果輸出檔已存在，預設不會重跑。
要強制重做，請加 `--force`：

```bash
uv run pipeline/denoising/sgmse.py \
  --input data/test_denoising/denoising.wav \
  --output-dir data/test_denoising/sgmse \
  --force
```

## 6. 參數說明

目前常用參數：

- `--N`
  - 反向 diffusion 步數
  - 預設 `30`
  - 數字越大通常越慢，但結果可能更穩
- `--sampler-type`
  - `pc` 或 `ode`
  - `pc` 是預設
- `--corrector`
  - `ald`、`langevin`、`none`
  - 預設 `ald`
- `--corrector-steps`
  - corrector 內部步數
  - 預設 `1`
- `--snr`
  - corrector 用的 SNR
  - 預設 `0.5`
- `--t-eps`
  - 最小 reverse time
  - 預設 `0.03`

## 7. 常見問題

### 7.1 `pyannote` 那種 import 問題不會出現在這裡嗎？

不會。這支腳本是 SGMSE 自己的本地 repo 載入，不是 pyannote。

### 7.2 音檔格式不對

先用標準 wav 最穩。
如果是奇怪編碼的 mp3 或 m4a，請先轉成 wav 再跑。

### 7.3 跑很慢

這是 diffusion 模型的正常特性。
你可以先降低 `--N` 測流程，再把它調回預設值跑完整品質。

### 7.4 目錄裡有很多檔案

可以直接把資料夾丟給 `--input`。
腳本會逐一處理並保留子目錄結構。
