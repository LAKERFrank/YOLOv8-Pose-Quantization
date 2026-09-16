# YOLOv8-Pose-Quantization

YOLOv8 Pose 專案，目前說明從影片自動產生姿態標記資料，再進行單通道灰階模型訓練（training）與微調（fine-tuning）的流程。量化等其他功能將後續補充。

## 目錄結構

原本的 `sportxai/` 改為 `dataset/`，其中存放標記資料的 `dataset/` 子目錄改為 `labeled-dataset/`。

```text
YOLOv8-Pose-Quantization/
├── dataset_generator.py
├── train.py
├── finetune.py
├── dataset/
│   ├── raw-data/                 # 原始影片，可放在多層子目錄
│   └── labeled-dataset/          # 自動產生
│       ├── images/              # 640 × 640 灰階 PNG
│       ├── labels/              # YOLO Pose 格式 TXT
│       ├── preview/             # 抽樣影像、標記與骨架預覽
│       └── data.yaml            # 訓練資料設定
├── trains/
│   └── office/                  # 下方 training 指令指定的輸出位置
└── finetune/
    └── office-finetune/         # 下方 fine-tuning 指令指定的輸出位置
```

## 環境準備

請在 repo 根目錄執行以下 terminal 指令；使用前需安裝 Python，並準備可執行這些腳本的 Ultralytics、PyTorch、OpenCV 與 NumPy 環境。

```sh
python -m pip install ultralytics torch opencv-python numpy
```

以下範例的 `--device 0` 使用第一張 GPU，需要支援 CUDA 的 PyTorch 環境；若使用 CPU，改成 `--device cpu`。本 repo 尚未固定套件版本。

## 1. 從影片產生標記資料

將影片放入 `dataset/raw-data/`。`dataset_generator.py` 會遞迴搜尋所有子目錄，讀取支援副檔名的影片，透過預訓練 Pose 模型自動標記，再輸出到 `dataset/labeled-dataset/`。

支援搜尋的副檔名：`.mp4`、`.avi`、`.mov`、`.mkv`、`.m4v`、`.wmv`、`.webm`、`.mpg`、`.mpeg`；實際解碼能力取決於 OpenCV 環境。

以下為 Linux / Bash 的背景執行範例，使用 `yolo26x-pose.pt` 權重。`nohup` 搭配 `&` 讓程式在背景執行，未另外指定輸出時通常會將紀錄寫入 `nohup.out`。兩組指令擇一執行；`--overwrite` 會刪除既有標記資料後重新產生。

### 有 ROI 版本

在原始影像的 `(0, 200)` 到 `(640, 640)` 區域內偵測：

```sh
nohup python3 dataset_generator.py \
  --root dataset \
  --weights yolo26x-pose.pt \
  --target_fps 0.5 \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.6 \
  --roi 0,200,640,640 \
  --overwrite \
  --preview_n 30 &
```

### 沒有 ROI 版本

使用完整影像進行偵測：

```sh
nohup python3 dataset_generator.py \
  --root dataset \
  --weights yolo26x-pose.pt \
  --target_fps 30 \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.6 \
  --overwrite \
  --preview_n 30 &
```

常用參數：

| 參數 | 說明 | 預設值 |
| --- | --- | --- |
| `--root` | 包含 `raw-data/` 的資料根目錄 | `dataset` |
| `--weights` | 自動標記使用的 Pose 權重 | `yolov8x-pose-p6.pt` |
| `--target_fps` | 每秒目標取樣影格數 | `120` |
| `--conf` | 人物偵測信心門檻 | `0.25` |
| `--kpt_thr` | 關鍵點信心門檻 | `0.15` |
| `--preview_n` | 抽樣預覽數量 | `10` |
| `--roi` | 原始影像座標的 ROI：`x1,y1,x2,y2` | 整張影像 |
| `--overwrite` | 刪除既有 `labeled-dataset/` 後重新產生 | 不啟用 |

ROI 座標需依實際影片調整。也可用 `--roi_cr0` 至 `--roi_cr3` 分別指定 `CameraReader_0.mp4` 至 `CameraReader_3.mp4` 的 ROI。輸出的影像仍保留完整畫面，等比例縮放並補黑邊到 640 × 640，標記座標會映射回完整影像。

標記包含 `person` 類別與 17 個人體關鍵點；每個偵測至少需有 13 個關鍵點達到信心門檻才會保留。請先查看 `preview/` 確認自動標記品質。

產生的 `data.yaml` 包含 `channels: 1`，並記錄資料集的絕對路徑；搬移資料集後需更新 `path`。目前 `train` 與 `val` 都使用 `images/`，尚未切分獨立驗證集，因此驗證數值不能代表對未見資料的泛化能力。

## 2. Training

`train.py` 讀取 `labeled-dataset/data.yaml`，預設從 `yolov8n-pose.pt` 預訓練權重開始訓練。程式將第一層卷積改為單通道，並停用 HSV 色彩增強。

以下保留完整的 training 指令設定，使用 Linux / Bash 的單一反斜線換行。

**目前版本相容性：** `train.py` 尚未提供 `--lr0`、`--lrf`、`--degrees`、`--translate`、`--scale`、`--shear`、`--perspective`、`--flipud`、`--fliplr`、`--mosaic` 參數。

```sh
nohup python3 train.py \
  --data data.yaml \
  --model yolov8n-pose.pt \
  --epochs 300 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 12 \
  --project trains \
  --name office \
  --lr0 0.001 \
  --lrf 0.01 \
  --degrees 2.0 \
  --translate 0.02 \
  --scale 0.15 \
  --shear 0.0 \
  --perspective 0.0 \
  --flipud 0.0 \
  --fliplr 0.5 \
  --mosaic 0.0 &
```

上述 `--project trains --name office` 指定的輸出目錄為 `trains/office/`；成功訓練後，模型權重位於：

```text
trains/office/weights/best.pt
trains/office/weights/last.pt
```

## 3. Fine-tuning

`finetune.py` 使用既有模型權重與 `labeled-dataset/data.yaml` 繼續微調，支援單通道模型，並使用較保守的姿態資料增強設定。

```sh
nohup python3 finetune.py \
  --data data.yaml \
  --weights best.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 12 \
  --project finetune \
  --name office-finetune \
  --lr0 0.00007 \
  --lrf 0.01 \
  --degrees 2 \
  --translate 0.02 \
  --scale 0.10 \
  --fliplr 0.5 \
  --mosaic 0.0 &
```

目前 `finetune.py` 支援上述所有參數。

上述指令透過 `--project finetune --name office-finetune` 將結果輸出到 `finetune/office-finetune/`，模型權重位於：

```text
finetune/office-finetune/weights/best.pt
finetune/office-finetune/weights/last.pt
```

查看完整參數：

```sh
python dataset_generator.py --help
python train.py --help
python finetune.py --help
```

## 4. Quantization

量化相關程式放在 `quantization_process/`。一般情況下只需設定 `run_all_quantization.sh` 開頭的參數，再執行一次腳本；它會依序產生固定 batch size 為 1 與 10 的 ONNX 模型及 TensorRT INT8 engine。

先進入量化程式目錄：

```sh
cd quantization_process
```

修改 `run_all_quantization.sh` 中的設定：

```sh
export MODEL_PT="/path/to/best.pt"
export DATA_YAML="/path/to/data.yaml"
export CALIB_FRAC=1
```

主要參數：

| 參數 | 說明 |
| --- | --- |
| `MODEL_PT` | 要進行量化的 YOLO Pose `.pt` 權重。模型需為本專案使用的單通道灰階模型。 |
| `DATA_YAML` | INT8 calibration 使用的資料集設定檔。程式會讀取 YAML 中的 `train` 路徑並搜尋圖片。 |
| `CALIB_FRAC` | 從 training dataset 隨機抽取多少比例進行 calibration。`1` 代表全部、`0.5` 代表 50%、`0.1` 代表 10%。batch 10 至少會取 10 張圖片。 |

其他設定：

| 參數 | 說明 | 腳本設定值 |
| --- | --- | --- |
| `IMGSZ` | 模型輸入影像尺寸 | `640` |
| `DEVICE` | 使用的 CUDA GPU 編號 | `0` |
| `WS` | TensorRT 建置 engine 可使用的 workspace 大小，單位為 GiB | `8` |
| `SEED` | 抽取 calibration 圖片時使用的隨機種子 | `0` |

執行量化：

```sh
bash run_all_quantization.sh
```

若要在背景執行並保存 log：

```sh
nohup bash run_all_quantization.sh > quantization.log 2>&1 &
```

腳本會依序執行以下流程：

1. 匯出固定 batch 1 的 `pose.onnx`。
2. 使用 calibration dataset 建立 batch 1 的 `int8.engine`。
3. 匯出固定 batch 10 的 `pose_batch10.onnx`。
4. 使用 calibration dataset 建立 batch 10 的 `int8_batch10.engine`。

輸出目錄由 Python 程式的 `OUT_DIR` 環境變數決定。若未設定，會使用程式內建路徑；建議在 `run_all_quantization.sh` 的設定區加入自己的輸出位置，例如：

```sh
export OUT_DIR="/path/to/quantized-weights"
```

完整的主要輸出如下：

```text
quantized-weights/
├── pose.onnx
├── int8.engine
├── calib.cache
├── pose_batch10.onnx
├── int8_batch10.engine
└── calib_batch10.cache
```

`CALIB_FRAC` 只影響 INT8 calibration 使用的圖片數量，不會改變原始 dataset。程式會以 `SEED` 固定隨機抽樣結果。若更換 `MODEL_PT`、`DATA_YAML` 或 calibration 設定，請先刪除輸出目錄內既有的 `calib.cache` 與 `calib_batch10.cache`，避免 TensorRT 重複使用舊的 calibration cache。

目前腳本預設只建立 INT8 engine；FP16 與 FP32 的建置開關在 Python 程式中預設關閉。

## 5. Measurement

量測相關程式放在 `measurement/`，分為兩種用途：

- `val_engine.py`：在指定 dataset 上驗證 TensorRT engine，輸出 box／pose mAP、FPS 與各階段平均耗時。
- `inferenceSpeed.sh`：使用 TensorRT 的 `trtexec` 進行效能壓測，觀察 engine 本身的 latency、throughput 與資料傳輸影響。

### 使用 `val_engine.py` 驗證模型

執行指令：

```sh
env -u PYTHONPATH /tmp/ultra_export_venv/bin/python val_engine.py \
  --models \
  /workspaces/CameraSensor/LayerSensing/Pose/weights/int8.engine \
  --datasets \
  /workspaces/CameraSensor/Quantization/pose-dataset/datasets/coco-pose/coco-pose.yaml \
  /workspaces/CameraSensor/Quantization/pose-dataset/datasets/groundtruth-upright/data.yaml \
  --imgsz 640 \
  --device 0 \
  --conf 0.25 \
  --iou 0.7 \
  --outdir ./val_engine_results
```

`env -u PYTHONPATH` 會先移除目前的 `PYTHONPATH`，再使用 `/tmp/ultra_export_venv/` 中的 Python 環境執行，避免其他 Python 套件路徑影響驗證環境。

主要參數：

| 參數 | 說明 |
| --- | --- |
| `--models` | 一個或多個要驗證的 TensorRT `.engine`。 |
| `--datasets` | 一個或多個 dataset YAML。程式使用各 dataset 的 validation split。 |
| `--imgsz` | 輸入影像尺寸。需與建立 engine 時使用的尺寸一致。 |
| `--device` | 使用的 CUDA GPU 編號。 |
| `--conf` | 預測結果的 confidence threshold。 |
| `--iou` | NMS 使用的 IoU threshold。 |
| `--outdir` | 驗證結果的輸出目錄。 |

程式會對每一個 model 與每一個 dataset 組合進行驗證。以上指令包含一個 engine 和兩個 datasets，因此結果中會有兩筆紀錄，分別代表 COCO Pose 與 ground truth upright dataset 的量測結果。

常用結果欄位：

| 欄位 | 意義 |
| --- | --- |
| `box_map`、`box_map50`、`box_map75` | Bounding box 的 COCO mAP、mAP@0.50 與 mAP@0.75。 |
| `pose_map`、`pose_map50`、`pose_map75` | 人體關鍵點的 COCO OKS mAP、mAP@0.50 與 mAP@0.75。 |
| `fps` | 只依 `infer_time_s` 計算的每秒處理影像數，不包含前處理與後處理。 |
| `speed_preprocess_ms` | 每張影像讀取、letterbox、色彩／灰階轉換與 normalization 的平均時間。 |
| `speed_inference_ms` | 每張影像執行 TensorRT wrapper 的平均時間。此程式的計時包含 buffer 配置、Host-to-Device 傳輸、engine 執行、Device-to-Host 傳輸與 CUDA stream 同步。 |
| `speed_postprocess_ms` | 每張影像進行輸出解碼、confidence 過濾、NMS、座標還原與整理評估資料的平均時間。 |
| `speed_loss_ms` | 此驗證程式不計算 loss，因此結果為空值。 |

這些速度是整個 validation dataset 的平均值。第一次執行可能受到 CUDA context 初始化、GPU 溫度、時脈或同機其他程序影響；比較不同 engine 時應使用相同硬體、dataset、batch size 與參數。

#### `val_engine.py` 結果輸出位置

由於指令先進入 `measurement/`，且指定 `--outdir ./val_engine_results`，結果會輸出到：

```text
measurement/val_engine_results/
├── results.csv
├── results.json
└── results.md
```

- `results.csv`：方便使用試算表整理或比較多組實驗。
- `results.json`：保留結構化的完整結果，方便其他程式讀取。
- `results.md`：方便直接閱讀的 Markdown 表格。

若加上 `--save-preds-json`，每個 model／dataset 組合還會在相同目錄產生一份 `<engine>__<dataset>.preds.json`，內容包含 bounding box 與 keypoint predictions。

### 使用 `inferenceSpeed.sh` 量測 TensorRT 效能

先修改 `inferenceSpeed.sh` 中 `--loadEngine` 後方的路徑，使其指向要量測的 engine，然後執行：

執行指令：
```sh
bash inferenceSpeed.sh
```

此腳本透過 `/usr/bin/trtexec` 重複執行 engine，主要選項的意義如下：

| 選項 | 說明 |
| --- | --- |
| `--warmUp=1000` | 先預熱 1000 ms，降低 CUDA 初始化與剛開始執行造成的偏差。 |
| `--duration=30` | 正式量測至少持續 30 秒。 |
| `--iterations=2000` | 設定量測 iteration 數量。 |
| `--useSpinWait` | 等待 GPU 時使用 spin-wait，通常能降低 latency 波動，但會提高 CPU 使用率。 |
| `--noDataTransfers` | 不量測輸入與輸出的 Host／Device 資料傳輸，用來觀察較接近 engine 純 GPU 執行的效能。 |

`trtexec` log 中通常應關注：

- `Throughput`：每秒完成多少次 inference。固定 batch 大於 1 時，每秒影像數約為 throughput 乘以 batch size。
- `GPU Compute Time`：GPU 執行 engine 的時間。
- `Host Latency`：主機端觀察到的 latency；包含 enqueue 及啟用資料傳輸時的 H2D／D2H 時間。
- latency percentile：例如 median、90%、95% 或 99%，用來觀察一般延遲及較慢情況，不能只看平均值。

`inferenceSpeed.sh` 前二行命令帶有 `--noDataTransfers`，用於觀察 engine 計算效能；後二行則包含資料傳輸，更接近包含 TensorRT I/O 的執行情境。這兩種量測都不包含影像讀檔、resize、normalization、NMS 或 keypoint 後處理，因此不能直接視為應用程式完整的端到端速度。端到端各階段耗時應參考 `val_engine.py` 的結果。

#### `inferenceSpeed.sh` 結果輸出位置

從 `measurement/` 執行時，腳本會將 `trtexec` 的標準輸出與錯誤輸出寫入：

```text
measurement/
├── batch1.log
└── batch10.log
```