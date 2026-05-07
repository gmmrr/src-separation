# TFGridNet Blind Separation

這份 `pipeline/tfgridnet.py` 是給 **已經有 checkpoint** 的情況直接做 blind separation / enhancement 用的，不包含訓練。

預設會使用：

- 程式碼來源：`models/external_repos/espnet/`
- checkpoint：`models/checkpoints/tfgridnet/yoshiki_wsj0_2mix_spatialized_enh_tfgridnet_waspaa2023_raw/exp/enh_train_enh_tfgridnet_waspaa2023_raw/25epoch.pth`
- 訓練設定：`models/checkpoints/tfgridnet/yoshiki_wsj0_2mix_spatialized_enh_tfgridnet_waspaa2023_raw/exp/enh_train_enh_tfgridnet_waspaa2023_raw/config.yaml`
- demo input：`data/test_blind_separate/IyLqUS7hRvo_std_vocals.wav`
- demo output：`data/test_blind_separate/tfgridnet_outputs/`

## 1. 需要的 dependencies

最小推論環境：

```bash
pip install torch torchaudio numpy librosa soundfile
```

如果你的本地 `espnet` 沒有安裝進環境，請安裝本地 repo：

```bash
pip install -e models/external_repos/espnet
```

## 2. 怎麼跑

### 預設 demo

```bash
python pipeline/tfgridnet.py
```

### 指定單一檔案

```bash
python pipeline/tfgridnet.py \
  --input data/test_blind_separate/IyLqUS7hRvo_std_vocals.wav \
  --output-dir data/test_blind_separate/tfgridnet_outputs
```

### 指定資料夾

如果你把 blind 測試檔放進一個資料夾，也可以直接丟資料夾進去。

```bash
python pipeline/tfgridnet.py \
  --input data/test_blind_separate \
  --output-dir data/test_blind_separate/tfgridnet_outputs
```

## 3. 輸出

每個輸入檔會輸出到一個子資料夾，結構像這樣：

- `00_original.wav`
- `01_tfgridnet.wav`
- `01_tfgridnet.json`
- `02_tfgridnet.wav`
- `02_tfgridnet.json`

其中 `00_original.wav` 是原始輸入的複製，方便你直接對照。
這顆 `yoshiki` checkpoint 是 2-source TFGridNet，所以正常會輸出兩個 separated wav。

JSON 會記錄：

- input 路徑
- output 路徑
- checkpoint 路徑
- sample rate
- device
- raw peak
- normalized peak
- stream index

## 4. 模型行為

- 這個 checkpoint 是 `TFGridNet` spatialized 2-source model
- 會把輸入 upmix 到 8 channels 後，以 `8000 Hz` 做 blind separation
- 寫檔前會先做 peak normalization，避免輸出被硬剪裁

## 5. 注意事項

- 執行時如果看到 `flash_attn` 的 warning，可以先忽略，這是 fallback 到 ESPnet 預設實作，不是錯誤。
- 這個模型是 enhancement / separation 類型，不需要 reference。

## 6. 瀏覽器 demo

如果你要做人工聽辨，請直接用 [pipeline/demo_tfgridnet.py](/Users/gmmrr/Documents/GitHub/src-separation/pipeline/demo_tfgridnet.py)。
它會用 Streamlit 顯示 `00_original.wav`、`01_tfgridnet.wav`、`02_tfgridnet.wav`，並用 Web Audio API 支援同步播放和逐軌靜音。
