# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-27

### Added
- Core model: ConvNeXt V2-Tiny + CBAM + TransformerEncoder + CTC/CE hybrid loss
- Baseline model: VGG-style CNN + BiLSTM
- FastAPI inference service with single image and batch recognition
- Docker deployment support with NVIDIA GPU
- MLOps automatic fine-tuning with hard sample collection
- Streaming dataset generation for training
- Fixed validation/test dataset (4000 images each)
- Unified evaluation script for all models
- Ablation experiments (5 groups)
- Traditional methods evaluation (OpenCV, KNN)
- Four-model comparison report
- Beam search decoding with post-processing (TextCorrector)
- One-click full pipeline execution

### Performance
- Test set accuracy: 86.78% (4000 images)
- Character-level accuracy: 96.93%
- Single image inference: ~15-50ms (GPU)

### Architecture
- Backbone: ConvNeXt V2-Tiny (ImageNet-22K pretrained)
- Attention: CBAM (Channel + Spatial)
- Sequence Modeling: TransformerEncoder (d_model=512, nhead=8, layers=3)
- Loss: CTC(0.6) + CE(0.4) + Label Smoothing(0.1)
- Decoding: Beam Search (beam_width=10) + TextCorrector

### Training Strategy
- Optimizer: AdamW (weight_decay=5e-5)
- LR Scheduler: OneCycleLR + Warmup (pct_start=0.15)
- Mixed Precision: AMP enabled
- Gradient Clipping: max_norm=1.0
- Early Stopping: patience=20
