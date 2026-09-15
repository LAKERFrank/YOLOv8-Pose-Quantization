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

在原始影像的 `(0, 200)` 到 `(640, 640)` 區域內偵測，目標每秒取樣 0.5 張（約每 2 秒一張）：

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

使用完整影像進行偵測，目標每秒取樣 30 張：

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

**目前版本相容性：** `train.py` 尚未提供 `--lr0`、`--lrf`、`--degrees`、`--translate`、`--scale`、`--shear`、`--perspective`、`--flipud`、`--fliplr`、`--mosaic` 參數，以下完整指令目前會出現 `unrecognized arguments`，需待程式支援後才能直接使用。本次僅記錄指令，不修改訓練程式；若先移除這些參數執行，學習率與資料增強將使用程式及套件的設定，不等同於以下完整設定。

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

`--data data.yaml` 指向目前工作目錄中的設定檔。若直接使用產生器的輸出，請改成 `--data dataset/labeled-dataset/data.yaml`。

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

目前 `finetune.py` 支援上述所有參數。`data.yaml` 與 `best.pt` 為目前工作目錄下的檔案；若沿用前述流程，請分別改成 `dataset/labeled-dataset/data.yaml` 與 `trains/office/weights/best.pt`。

上述指令透過 `--project finetune --name office-finetune` 將結果輸出到 `finetune/office-finetune/`，模型權重位於：

```text
finetune/office-finetune/weights/best.pt
finetune/office-finetune/weights/last.pt
```

兩支程式保留原始設定：未指定 `--project` 與 `--name` 時，`train.py` 的輸出位置為 `runs/train/`，`finetune.py` 則為 `finetune/counterclockwise-invert-100/`。若指定的輸出目錄已存在，Ultralytics 可能自動加上數字尾碼；微調時請指定實際產生的權重路徑。

查看完整參數：

```sh
python dataset_generator.py --help
python train.py --help
python finetune.py --help
```
