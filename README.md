# CAPTCHA Recognition System

A standardized CAPTCHA recognition system based on ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE hybrid loss.

- Clean code, no legacy baggage
- Advanced architecture with baseline comparisons
- One-click reproducible with rigorous evaluation
- Ready to package as API for commercial service
- Supports Docker one-click deployment
- Supports MLOps automatic fine-tuning (hard sample collection + auto-trigger + model iteration)

## API Service

Provides FastAPI inference interface supporting single image and batch recognition.

### Quick Start

```bash
# Method 1: Run directly
python -m api.app

# Method 2: Docker deployment (requires NVIDIA GPU + Docker)
docker build -t captcha-api .
docker run --gpus all -p 8000:8000 captcha-api
```

### API Endpoints

**`GET /health`** — Health check
```bash
curl http://localhost:8000/health
# → {"status":"ok","device":"cuda","model_loaded":true}
```

**`POST /predict`** — Single image recognition
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@captcha.png"
# → {"text":"AbC123","confidence":0.99,"inference_time_ms":15.2}
```

**`POST /predict/batch`** — Batch recognition
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@img1.png" \
  -F "files=@img2.png"
```

### Python Client Example

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("captcha.png", "rb")}
)
result = resp.json()
print(f"CAPTCHA: {result['text']}, Confidence: {result['confidence']}")
```

### API Performance

| Metric | Value |
|------|-----|
| Test Set Accuracy | **86.78%** (4000 images) |
| Character-level Accuracy | **96.93%** |
| Single Image Inference | **~15-50ms** (GPU) |
| BEAM_WIDTH | 10 |

## Docker Deployment

### Requirements

- NVIDIA GPU (recommended 8GB+ VRAM)
- Docker (19.03+)
- NVIDIA Container Toolkit

### Build and Run

```bash
docker build -t captcha-api .
docker run --gpus all -p 8000:8000 captcha-api
```

### Deployment Package Download

The project is packaged as `captcha-api-deploy.tar.gz` (399MB), containing complete inference code and model weights:

```bash
tar xzf captcha-api-deploy.tar.gz
cd captcharecognition
docker build -t captcha-api .
docker run --gpus all -p 8000:8000 captcha-api
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# One-click run full pipeline
python main.py --mode full
```

## Project Structure

```
captcharecognition/
├── config.yaml                     # Global configuration (single parameter source)
├── chars_config.yaml               # Character set configuration
├── image_config.yaml               # Image size configuration
├── main.py                         # Unified entry point (7-step pipeline)
├── requirements.txt                # Training dependencies
├── requirements-inference.txt      # Inference dependencies
├── Dockerfile                      # Docker build file
├── .dockerignore                   # Docker ignore rules
│
├── api/
│   └── app.py                        # FastAPI inference service
├── models/
│   ├── model.py                      # Core model (ConvNeXt-Tiny + CBAM + Transformer/BiLSTM)
│   ├── baseline_vgg_cnn_lstm.py      # Baseline model (VGG + BiLSTM)
│   ├── train.py                      # Core model training script
│   ├── dataset.py                    # Dataset (fixed + streaming dual mode)
│   ├── transforms.py                 # Data augmentation
│   ├── hybrid_loss_fixed.py          # CTC+CE hybrid loss
│   └── evaluate.py                   # Evaluation tools
│
├── generate/
│   ├── generate_dataset.py         # CAPTCHA generator (7 fonts, high complexity)
│   └── generate_fixed_dataset.py   # Fixed validation/test set generator
│
├── preprocess/
│   ├── preprocess_dataset.py       # Preprocessing (grayscale + Gaussian + resize)
│   ├── clean.py                    # Data cleaning
│   └── split_dataset.py            # Dataset splitting
│
├── utils/
│   ├── config_loader.py            # Configuration loader
│   ├── decoder.py                  # CTC decoding (greedy + beam search)
│   ├── chars.py                    # Character mapping
│   ├── device_manager.py           # Device management
│   ├── training_utils.py           # Training utilities
│   ├── training_recorder.py        # Training recorder
│   ├── common.py                   # Common utilities
│   ├── metrics.py                  # Evaluation metrics
│   ├── directory_manager.py        # Directory management
│   └── logger.py                   # Logging configuration
│
├── scripts/
│   ├── monitor_mlops.sh            # MLOps monitoring script (background auto fine-tuning)
│   ├── mlops_crontab.txt           # Startup crontab configuration template
│   ├── check_training_prerequisites.sh # Pre-training check script
│   └── check_engineering_rules.py  # Engineering standards check
│
├── train_baseline_vgg.py           # Baseline model training script
├── evaluate_all_models.py          # Unified evaluation script
├── generate_comparison_report.py   # Four-model comparison report
├── run_ablation.py                 # Ablation experiments (5 groups)
└── evaluate_traditional.py         # Traditional methods evaluation
```

## Core Model

| Component | Configuration | Description |
| -------- | -------------------- | ------------------------------------------------ |
| Backbone | ConvNeXt V2-Tiny | ImageNet-22K pretrained, ~28M parameters |
| Attention | CBAM | Channel attention + Spatial attention |
| Sequence Modeling | TransformerEncoder (default) / BiLSTM | hidden=512, nhead=8, layers=2, dropout=0.3 |
| Loss Function | CTC(0.6) + CE(0.4) | + Label Smoothing(0.1) |
| Decoding Strategy | Beam Search | beam width=10, + TextCorrector error correction |
| Input Size | 64×256 | Height × Width |
| Pooling Size | 2×8 (time steps=16) | Adaptive average pooling |

### Training Strategy

| Parameter | Value |
| ------ | ---------------------------- |
| Batch Size | 128 |
| Initial Learning Rate | 1×10⁻⁵ |
| Peak Learning Rate | 5×10⁻⁴ |
| Epochs | 120 |
| Early Stopping Patience | 15 |
| Gradient Clipping Norm | 1.0 |
| Optimizer | AdamW (weight\_decay=5×10⁻⁵) |
| LR Scheduler | OneCycleLR + Warmup (pct_start=0.15) |
| Mixed Precision | AMP enabled |
| EMA Momentum | 0.999 |

## Baseline Model (VGG-style CNN + BiLSTM)

| Component | Configuration | Description |
| ------ | ------------------ | --------------------------------- |
| CNN Encoder | 4 VGG-style blocks | 3→32→64→128→256 channels |
| Height Pooling | AdaptiveAvgPool2d | Compress height to 1 |
| Width Expansion | Interpolate to 16 time steps | Provide sufficient sequence length |
| Sequence Modeling | BiLSTM | hidden=256, layers=2, dropout=0.3 |
| Loss Function | CTC(0.6) + CE(0.4) | Hybrid loss |
| Parameters | ~3.8M | Lightweight model |

## CAPTCHA Generation Parameters

| Parameter | Value | Description |
| ------ | ---------------- | ------------- |
| Image Size | 200×60 | Generation size (width × height) |
| Character Set | 62 classes | 0-9, A-Z, a-z |
| Character Length | 4-6 characters | Variable length |
| Font Pool | 7 fonts | High diversity |
| Adhesion Probability | 15% | |
| Rotation Range | ±6° | |
| Scale Range | 0.96-1.04 | |
| Interference Lines | 2-4 lines | |
| Interference Line Thickness | 1-3 | |
| Gaussian Noise σ | 2.0-5.0 | |
| Salt & Pepper Noise Ratio | 0.001-0.003 | |
| Speckle Noise Intensity | 0.02-0.06 | |
| Wave Distortion | Amplitude 0.6-1.8, Probability 30% | |

## Full Pipeline (One-click Run)

```bash
python main.py --mode full
```

Executes in sequence:

1. Generate fixed validation/test sets (4000 each) + preprocessing
2. Streaming training of core model (ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE)
3. Streaming training of baseline model (VGG + BiLSTM + CTC/CE)
4. Unified evaluation of both models
5. Run two traditional methods (OpenCV, KNN)
6. Generate four-model comparison report
7. Ablation experiments (5 groups)

### Step-by-step Execution

```bash
# Generate fixed dataset
python main.py --mode generate_fixed

# Preprocess fixed dataset
python main.py --mode preprocess_fixed

# Train core model
python main.py --mode train_core

# Train baseline model
python main.py --mode train_baseline

# Evaluate all models
python main.py --mode evaluate

# Run traditional methods
python main.py --mode traditional

# Generate comparison report
python main.py --mode comparison

# Run ablation experiments
python main.py --mode ablation

# MLOps mode: auto fine-tuning based on hard samples
python main.py --mode mlops --hard_sample_dir data/hard_samples --threshold 500 --lr 1e-6 --epochs 5
```

### Or Run Sub-scripts Directly (see real-time tqdm progress)

```bash
python generate/generate_fixed_dataset.py
python preprocess/preprocess_dataset.py --mode fixed
python models/train.py --model_type captcha --streaming
python train_baseline_vgg.py --streaming
python evaluate_all_models.py
```

## Output Directories

- `checkpoints/` — Best model weights (`best_model.pth` for core model, `baseline_vgg_best.pth` for baseline model)
- `results/` — Comparison charts, JSON, ablation tables
- `runs/` — TensorBoard logs
- `data/fixed/` — Permanent evaluation benchmark (val/test, 4000 each)
- `data/preprocessed/` — Preprocessed fixed dataset
- `data/hard_samples/` — MLOps hard samples (misrecognized samples for auto fine-tuning)
- `logs/` — Training logs, MLOps monitoring logs

## Ablation Experiments

Run full ablation experiments (5 groups):

```bash
python run_ablation.py
```

The ablation experiments include 5 groups:

1. **full**: Complete model (ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE)
2. **no_cbam**: Remove CBAM attention module
3. **no_ctc**: Remove CTC loss (CE only)
4. **resnet34**: Replace backbone with ResNet-34
5. **no_bilstm**: Remove BiLSTM (CNN only + CTC/CE)

## MLOps Automatic Fine-tuning

### Overview

The MLOps module implements a "continuous learning" loop: when the model encounters difficult-to-recognize samples during evaluation, they are automatically collected as hard samples. Once a threshold is reached, fine-tuning is automatically triggered, enabling online model iteration and optimization.

### Workflow

1. **Collect Hard Samples**: Automatically save misrecognized samples during evaluation
2. **Background Monitoring**: Periodically check if the number of hard samples reaches the threshold
3. **Auto Fine-tuning**: Fine-tune the model using 80% streaming data + 20% hard samples
4. **Quality Gate**: Only replace the old model if the new model has higher accuracy
5. **Auto Backup**: Automatically backup the old model before replacement

### Usage

**Manual Hard Sample Collection:**
```bash
# Collect misrecognized samples to data/hard_samples/ during evaluation
python evaluate_all_models.py --collect_hard_samples

# Custom save directory
python evaluate_all_models.py --collect_hard_samples --hard_samples_dir data/custom_hard_samples
```

**Manual Fine-tuning Trigger:**
```bash
# Auto fine-tune model when hard samples reach threshold
python main.py --mode mlops

# Custom parameters
python main.py --mode mlops \
    --hard_sample_dir data/hard_samples \
    --threshold 500 \
    --lr 1e-6 \
    --epochs 5
```

**Background Auto Monitoring:**
```bash
# Start background monitoring (check every 5 minutes, auto-trigger when hard samples >= 500)
nohup ./scripts/monitor_mlops.sh > logs/mlops_monitor_nohup.log 2>&1 &

# View monitoring logs
tail -f logs/mlops_monitor.log

# View trigger records
cat logs/mlops_trigger.log

# Stop monitoring
pkill -f "monitor_mlops.sh"
```

**Startup Auto-start:**
```bash
# Use crontab to auto-start monitoring script
crontab scripts/mlops_crontab.txt

# View configured crontab
crontab -l
```

### Fine-tuning Parameters

| Parameter | Default | Description |
|------|--------|------|
| `--hard_sample_dir` | `data/hard_samples` | Hard sample directory |
| `--threshold` | 500 | Sample threshold to trigger fine-tuning |
| `--lr` | 1e-6 | Fine-tuning learning rate (very low, fine polishing) |
| `--epochs` | 5 | Fine-tuning epochs |
| Mix Ratio | 80/20 | Streaming data / hard samples ratio |
| CTC/CE Weight | 0.6/0.4 | Consistent with original training |
| Mixed Precision | AMP | Enabled |
| LR Scheduler | OneCycleLR | Enabled |

## Configuration Management

All parameters are managed in `config.yaml` and read through `utils/config_loader.py`.

| Config File | Purpose |
|----------|------|
| `config.yaml` | Global configuration (parameter center) |
| `chars_config.yaml` | Character set configuration |
| `image_config.yaml` | Image size configuration |

**No hardcoding or duplicate definitions** — all config values are read through a unified interface.

## Requirements

### Training / Evaluation
- Python 3.8+
- CUDA 11.7+ (GPU acceleration, optional)
- Recommended: 16GB+ RAM, 8GB+ VRAM

### Docker Deployment
- NVIDIA GPU (recommended 8GB+ VRAM)
- Docker 19.03+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- Supports `--gpus all` parameter

## FAQ

**Q: Training is slow?**
A: Make sure CUDA-enabled PyTorch is installed and set `system.num_workers=8` in config.yaml.

**Q: OOM (Out of Memory)?**
A: Reduce `training.batch_size` in config.yaml, or clean up zombie processes.

**Q: Pretrained weight download fails?**
A: The model will gracefully degrade to random initialization without affecting the training pipeline. You can also configure a domestic mirror source.