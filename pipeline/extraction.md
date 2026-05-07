# USEF-TSE 推論流程

這份 `pipeline/usef_tse.py` 是給 **已經有 checkpoint** 的情況直接做推論用的，不包含訓練。

預設會使用：

- 程式碼來源：`models/external_repos/USEF-TSE/`
- checkpoint：`models/checkpoints/usef_tse/ZBang_USEF-TSE/chkpt/USEF-SepFormer/wsj0-2mix/temp_best.pth.tar`
- demo aux/reference：`data/test_usef_tse/IyLqUS7hRvo_std_vocals/`
- demo mix：`data/test_usef_tse/` 第一層的 `.wav`

## 1. 需要的 dependencies

最小推論環境：

```bash
pip install torch numpy librosa soundfile
```

如果你想跑官方 repo 的原始 `eval.py`，才另外需要：

```bash
pip install hyperpyyaml mir_eval pypesq
```

## 2. 安裝前確認

先確認這些路徑存在：

```text
models/external_repos/USEF-TSE/
models/checkpoints/usef_tse/ZBang_USEF-TSE/chkpt/USEF-SepFormer/wsj0-2mix/model.py
models/checkpoints/usef_tse/ZBang_USEF-TSE/chkpt/USEF-SepFormer/wsj0-2mix/temp_best.pth.tar
data/test_usef_tse/IyLqUS7hRvo_std_vocals/
```

如果你的環境沒有把 USEF-TSE repo 抓下來，請先把它放到 `models/external_repos/USEF-TSE/`。

## 3. 怎麼跑

### 預設 demo

這會使用：

- `data/test_usef_tse/` 第一層的 `.wav` 當 mix
- `data/test_usef_tse/IyLqUS7hRvo_std_vocals/` 當 aux/reference

```bash
python pipeline/usef_tse.py
```

### 指定單一 mix

```bash
python pipeline/usef_tse.py \
  --mix data/test_usef_tse/IyLqUS7hRvo_std_vocals.wav \
  --aux data/test_usef_tse/IyLqUS7hRvo_std_vocals \
  --output-dir data/test_usef_tse/usef_tse_outputs
```

### 指定資料夾批次跑

如果 `--aux` 是單一檔案，它會套用到所有 mix。

```bash
python pipeline/usef_tse.py \
  --mix data/test_usef_tse \
  --aux data/test_usef_tse/IyLqUS7hRvo_std_vocals \
  --output-dir data/test_usef_tse/usef_tse_outputs
```

如果 `--aux` 也是資料夾，則檔名需要和 mix 的 `stem` 對得起來，或是 aux 資料夾只有一個檔案。

如果 `--mix` 只有單一音檔、而 `--aux` 是一個包含多個 reference 的資料夾，腳本會把每個 reference 都各自跑一次，輸出會放在 `output-dir/<mix stem>/` 底下。

如果 `--mix` 是資料夾、`--aux` 也是資料夾，則會退回成按檔名 stem 配對；如果 `--aux` 只有單一檔案，則同一個 reference 會套用到所有 mix。

## 4. 輸出

每個 mix 會輸出到一個子資料夾，結構像這樣：

- `01_<reference name>_usef_tse.wav`
- `01_<reference name>_usef_tse.json`
- `02_<reference name>_usef_tse.wav`
- `02_<reference name>_usef_tse.json`

模型輸出在寫檔前會做 peak normalization，避免 demo 音檔因為超出 `[-1, 1]` 而被硬剪裁。

JSON 會記錄：

- mix 路徑
- aux 路徑
- checkpoint 路徑
- sample rate
- device

## 5. 模型行為

- 這個 checkpoint 是 `USEF-SepFormer`
- 內部會把輸入音檔 resample 成 `8000 Hz`
- 會轉成 mono 做推論
- 不做訓練、不做 fine-tune

## 6. 如果你要換自己的資料

你只要改兩個參數：

```bash
--mix <你的混音檔或資料夾>
--aux <你的 target speaker reference 檔或資料夾>
```

通常 target speaker reference 檔長度不用太長，但要能代表目標說話人。
