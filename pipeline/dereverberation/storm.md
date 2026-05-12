# StoRM Dereverberation

這份說明對應目前的 [`pipeline/dereverberation/storm.py`](./storm.py)。

這支腳本是**直接推論版**：
- 會讀取 `models/external_repos/storm` 裡的 StoRM 原始碼
- 會載入 `models/checkpoints/storm/WSJ0+Reverb/epoch=237-pesq=0.00.ckpt`
- 不會訓練，也不會重新 fine-tune
- 會把輸入音訊做 dereverberation 後輸出 `.wav` 和 `.json`

## 1. 安裝套件

先安裝 StoRM 這條推論路徑會用到的缺件。

### 必要套件

```bash
uv pip install ninja
```

`ninja` 很重要，因為 StoRM 的 `ncsnpp` backbone 在 import 時會嘗試編譯 CUDA extension。沒有 `ninja` 時，模型通常會卡在載入階段。

### 建議一起確認有的套件

如果你的環境還沒裝齊，建議補這些：

```bash
uv pip install torch torchaudio soundfile numpy scipy tqdm pytorch-lightning torch-ema tensorboard h5py pandas matplotlib pesq pystoi sdeint pyroomacoustics
```

### 注意

`models/external_repos/storm/requirements.txt` 是舊版專案用的鎖版依賴，很多版本是針對 Python 3.8 的。
如果你現在是 Python 3.12，不一定能直接整包照抄安裝。
比較穩的做法是：

1. 先裝上面列的必要/常用套件
2. 如果還缺某個模組，再單獨 `uv pip install <package>`

## 2. 輸入資料怎麼放

目前這支腳本支援兩種輸入：

### 單一檔案

例如你的檔案是：

```text
data/test_dereverberation/reverb.wav
```

可以直接這樣跑：

```bash
uv run pipeline/dereverberation/storm.py --input data/test_dereverberation/reverb.wav
```

### 整個資料夾

如果你把很多音檔放在同一個資料夾，例如：

```text
data/test_dereverberation/
  reverb.wav
  other.wav
  nested/
    more.wav
```

可以直接把資料夾丟進去：

```bash
uv run pipeline/dereverberation/storm.py --input data/test_dereverberation
```

腳本會遞迴找出裡面的音檔並逐一處理。

### 支援格式

目前支援：
- `.wav`
- `.flac`
- `.mp3`
- `.m4a`
- `.ogg`

## 3. `reverb.wav` 要怎麼處理

你現在的檔案是：

```text
data/test_dereverberation/reverb.wav
```

最簡單的做法是直接把它當單一檔案丟給腳本：

```bash
uv run pipeline/dereverberation/storm.py \
  --input data/test_dereverberation/reverb.wav \
  --output-dir data/test_dereverberation/storm
```

這支腳本會自動做兩件事：

1. 讀取音檔
2. 自動重採樣到 `16 kHz`

所以你**不需要先手動 resample 成 16k**。

如果你的音檔是 stereo，腳本會先取第一個 channel 來做推論。

## 4. 輸出結果在哪裡

如果輸入是：

```text
data/test_dereverberation/reverb.wav
```

預設輸出會是：

```text
data/test_dereverberation/storm/reverb_storm.wav
data/test_dereverberation/storm/reverb_storm.json
```

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
uv run pipeline/dereverberation/storm.py
```

### 明確指定單一檔案

```bash
uv run pipeline/dereverberation/storm.py \
  --input data/test_dereverberation/reverb.wav \
  --output-dir data/test_dereverberation/storm
```

### 指定 checkpoint

如果你想改用別的 StoRM checkpoint，可以覆蓋 `--ckpt`：

```bash
uv run pipeline/dereverberation/storm.py \
  --input data/test_dereverberation/reverb.wav \
  --ckpt models/checkpoints/storm/WSJ0+Reverb/epoch=237-pesq=0.00.ckpt
```

### 強制覆寫輸出

如果輸出檔已存在，預設不會重跑。
要強制重做，請加 `--force`：

```bash
uv run pipeline/dereverberation/storm.py \
  --input data/test_dereverberation/reverb.wav \
  --output-dir data/test_dereverberation/storm \
  --force
```

## 6. 常見問題

### 6.1 卡在 `ninja` 或 import extension

先確認有裝：

```bash
uv pip install ninja
```

### 6.2 輸入檔不存在

確認檔案真的存在：

```text
data/test_dereverberation/reverb.wav
```

如果你改了檔名，就要把 `--input` 改成新的路徑。

### 6.3 音檔太短或格式怪

建議先用標準 wav 檔。
如果來源檔案很短、很雜訊、或不是一般 speech/reverb 音檔，結果可能不穩定。

### 6.4 想處理整個資料夾

直接把資料夾當 `--input` 傳入即可。
腳本會把每個檔案各自輸出成獨立結果。
