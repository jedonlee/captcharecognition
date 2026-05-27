# 验证码识别系统

基于 ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE 混合损失的标准化验证码识别系统。

- 代码干净、无历史包袱
- 架构先进、有对比基线
- 一键可复现、评估严谨
- 可直接封装 API 提供商业服务
- 支持 Docker 一键部署
- 支持 MLOps 自动微调（硬样本收集 + 自动触发 + 模型迭代）

## API 服务

提供 FastAPI 推理接口，支持单图和批量识别。

### 快速启动

```bash
# 方式一：直接运行
python -m api.app

# 方式二：Docker 部署（需要 NVIDIA GPU + Docker）
docker build -t captcha-api .
docker run --gpus all -p 8000:8000 captcha-api
```

### API 端点

**`GET /health`** — 健康检查
```bash
curl http://localhost:8000/health
# → {"status":"ok","device":"cuda","model_loaded":true}
```

**`POST /predict`** — 单图识别
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@captcha.png"
# → {"text":"AbC123","confidence":0.99,"inference_time_ms":15.2}
```

**`POST /predict/batch`** — 批量识别
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@img1.png" \
  -F "files=@img2.png"
```

### Python 调用示例

```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("captcha.png", "rb")}
)
result = resp.json()
print(f"验证码: {result['text']}, 置信度: {result['confidence']}")
```

### API 性能

| 指标 | 值 |
|------|-----|
| 测试集精度 | **86.78%**（4000张） |
| 字符级精度 | **96.93%** |
| 单图推理 | **~15-50ms**（GPU） |
| BEAM_WIDTH | 10 |

## Docker 部署

### 环境要求

- NVIDIA GPU（推荐 8GB+ 显存）
- Docker（19.03+）
- NVIDIA Container Toolkit

### 构建与运行

```bash
docker build -t captcha-api .
docker run --gpus all -p 8000:8000 captcha-api
```

### 部署包下载

项目已打包为 `captcha-api-deploy.tar.gz`（399MB），包含完整推理代码和模型权重：

```bash
tar xzf captcha-api-deploy.tar.gz
cd captcharecognition
docker build -t captcha-api .
docker run --gpus all -p 8000:8000 captcha-api
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 一键运行完整流水线
python main.py --mode full
```

## 项目结构

```
captcharecognition/
├── config.yaml                     # 全局配置（唯一参数中心）
├── chars_config.yaml               # 字符集配置
├── image_config.yaml               # 图像尺寸配置
├── convnextv2_tiny_22k_224_ema.pt  # ConvNeXt V2-Tiny 预训练权重
├── main.py                         # 统一入口（7步流水线）
├── requirements.txt                # 训练依赖清单
├── requirements-inference.txt      # 推理依赖清单
├── Dockerfile                      # Docker 构建文件
├── .dockerignore                   # Docker 忽略规则
│
├── api/
│   └── app.py                        # FastAPI 推理服务
├── models/
│   ├── model.py                      # 核心模型（ConvNeXt-Tiny + CBAM + Transformer/BiLSTM）
│   ├── baseline_vgg_cnn_lstm.py      # 基线模型（VGG + BiLSTM）
│   ├── train.py                      # 核心模型训练脚本
│   ├── dataset.py                    # 数据集（固定+流式双模式）
│   ├── transforms.py                 # 数据增强
│   ├── hybrid_loss_fixed.py          # CTC+CE混合损失
│   └── evaluate.py                   # 评估工具
│
├── generate/
│   ├── generate_dataset.py         # 验证码生成器（7字体，高复杂度）
│   └── generate_fixed_dataset.py   # 固定验证/测试集生成器
│
├── preprocess/
│   ├── preprocess_dataset.py       # 预处理（灰度+高斯+resize）
│   ├── clean.py                    # 数据清洗
│   └── split_dataset.py            # 数据集划分
│
├── utils/
│   ├── config_loader.py            # 配置加载器
│   ├── decoder.py                  # CTC解码（贪心+beam search）
│   ├── chars.py                    # 字符映射
│   ├── device_manager.py           # 设备管理
│   ├── training_utils.py           # 训练工具
│   ├── training_recorder.py        # 训练记录
│   ├── common.py                   # 通用工具
│   ├── metrics.py                  # 评估指标
│   ├── directory_manager.py        # 目录管理
│   └── logger.py                   # 日志配置
│
├── tests/
│   └── test_optimization.py        # 性能优化验证
│
├── scripts/
│   ├── monitor_mlops.sh            # MLOps监控脚本（后台自动触发微调）
│   ├── mlops_crontab.txt           # 开机自启crontab配置模板
│   ├── check_training_prerequisites.sh # 训练前检查脚本
│   └── check_engineering_rules.py  # 工程规范检查
│
├── train_baseline_vgg.py           # 基线模型训练脚本
├── evaluate_all_models.py          # 统一评估脚本
├── generate_comparison_report.py   # 四模型对比报告
├── run_ablation.py                 # 消融实验（5组）
└── evaluate_traditional.py         # 传统方法评估
```

## 核心模型

| 组件 | 配置 | 说明 |
| -------- | -------------------- | ------------------------------------------------ |
| 骨干网络 | ConvNeXt V2-Tiny | ImageNet-22K预训练，~28M参数 |
| 注意力模块 | CBAM | 通道注意力 + 空间注意力 |
| 序列建模 | TransformerEncoder (默认) / BiLSTM | hidden=512, nhead=8, layers=2, dropout=0.3 |
| 损失函数 | CTC(0.35) + CE(0.65) | + 标签平滑(0.1) |
| 解码策略 | Beam Search | beam width=10, + TextCorrector 纠错 |
| 输入尺寸 | 64×256 | 高×宽 |
| 池化尺寸 | 2×8（时间步=16） | 自适应平均池化 |

### 训练策略

| 参数 | 值 |
| ------ | ---------------------------- |
| 批次大小 | 128 |
| 初始学习率 | 1×10⁻⁵ |
| 峰值学习率 | 5×10⁻⁴ |
| 训练轮数 | 120 |
| 早停耐心值 | 15 |
| 梯度裁剪范数 | 1.0 |
| 优化器 | AdamW (weight\_decay=5×10⁻⁵) |
| 学习率调度 | OneCycleLR + Warmup (pct_start=0.15) |
| 混合精度 | AMP 启用 |
| EMA 动量 | 0.999 |

## 基线模型（VGG风格CNN + BiLSTM）

| 组件 | 配置 | 说明 |
| ------ | ------------------ | --------------------------------- |
| CNN编码器 | 4个VGG风格模块 | 3→32→64→128→256通道 |
| 高度池化 | AdaptiveAvgPool2d | 将高度压缩至1 |
| 宽度扩展 | 插值至16时间步 | 提供足够的序列长度 |
| 序列建模 | BiLSTM | hidden=256, layers=2, dropout=0.3 |
| 损失函数 | CTC(0.4) + CE(0.6) | 混合损失 |
| 参数量 | ~3.8M | 轻量级模型 |

## 验证码生成参数

| 参数 | 值 | 说明 |
| ------ | ---------------- | ------------- |
| 图像尺寸 | 200×60 | 生成时尺寸 (宽×高) |
| 字符集 | 62类 | 0-9, A-Z, a-z |
| 字符长度 | 4-6位 | 不定长 |
| 字体池 | 7种字体 | 高多样性 |
| 粘连概率 | 15% | |
| 旋转范围 | ±6° | |
| 缩放范围 | 0.96-1.04 | |
| 干扰线条数 | 2-4条 | |
| 干扰线粗细 | 1-3 | |
| 高斯噪声σ | 2.0-5.0 | |
| 椒盐噪声比例 | 0.001-0.003 | |
| 斑点噪声强度 | 0.02-0.06 | |
| 波浪扭曲 | 振幅0.6-1.8, 概率30% | |

## 完整流水线（一键运行）

```bash
python main.py --mode full
```

依次执行：

1. 生成固定验证/测试集（各4000张） + 预处理
2. 流式训练核心模型（ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE）
3. 流式训练基线模型（VGG + BiLSTM + CTC/CE）
4. 统一评估两个模型
5. 运行两个传统方法（OpenCV、KNN）
6. 生成四模型对比报告
7. 消融实验（5组）

### 分步运行

```bash
# 生成固定数据集
python main.py --mode generate_fixed

# 预处理固定数据集
python main.py --mode preprocess_fixed

# 训练核心模型
python main.py --mode train_core

# 训练基线模型
python main.py --mode train_baseline

# 评估所有模型
python main.py --mode evaluate

# 运行传统方法
python main.py --mode traditional

# 生成对比报告
python main.py --mode comparison

# 运行消融实验
python main.py --mode ablation

# MLOps 模式：基于硬样本自动微调
python main.py --mode mlops --hard_sample_dir data/hard_samples --threshold 500 --lr 1e-6 --epochs 5
```

### 或直接运行子脚本（可看到实时tqdm进度条）

```bash
python generate/generate_fixed_dataset.py
python preprocess/preprocess_dataset.py --mode fixed
python models/train.py --model_type captcha --streaming
python train_baseline_vgg.py --streaming
python evaluate_all_models.py
```

## 输出目录

- `checkpoints/` — 最佳模型权重（`best_model.pth` 为核心模型，`baseline_vgg_best.pth` 为基线模型）
- `results/` — 对比图、JSON、消融表
- `runs/` — TensorBoard 日志
- `data/fixed/` — 永久评估基准（val/test 各4000张）
- `data/preprocessed/` — 预处理后的固定集
- `data/hard_samples/` — MLOps 硬样本（识别错误的样本，用于自动微调）
- `logs/` — 训练日志、MLOps 监控日志

## 消融实验

运行完整消融实验（5组）：

```bash
python run_ablation.py
```

消融实验包含5组：

1. **full**: 完整模型（ConvNeXt V2-Tiny + CBAM + BiLSTM + CTC/CE）
2. **no_cbam**: 移除CBAM注意力模块
3. **no_ctc**: 移除CTC损失（仅CE）
4. **resnet34**: 骨干网络替换为ResNet-34
5. **no_bilstm**: 移除BiLSTM（仅CNN + CTC/CE）

## MLOps 自动微调

### 概述

MLOps 模块实现了"持续学习"闭环：当模型在评估中遇到识别困难的样本时，自动收集为硬样本，积累到一定数量后自动触发微调，实现模型在线迭代优化。

### 工作流程

1. **收集硬样本**：评估时自动保存识别错误的样本
2. **后台监控**：定期检查硬样本数量是否达到阈值
3. **自动微调**：使用 80% 流式数据 + 20% 硬样本微调模型
4. **质量门控**：只有新模型准确率更高时才替换旧模型
5. **自动备份**：替换前自动备份旧模型

### 使用方法

**手动收集硬样本：**
```bash
# 评估时收集识别错误的样本到 data/hard_samples/
python evaluate_all_models.py --collect_hard_samples

# 自定义保存目录
python evaluate_all_models.py --collect_hard_samples --hard_samples_dir data/custom_hard_samples
```

**手动触发微调：**
```bash
# 当硬样本达到阈值时，自动微调模型
python main.py --mode mlops

# 自定义参数
python main.py --mode mlops \
    --hard_sample_dir data/hard_samples \
    --threshold 500 \
    --lr 1e-6 \
    --epochs 5
```

**后台自动监控：**
```bash
# 启动后台监控（每5分钟检查一次，硬样本>=500时自动触发微调）
nohup ./scripts/monitor_mlops.sh > logs/mlops_monitor_nohup.log 2>&1 &

# 查看监控日志
tail -f logs/mlops_monitor.log

# 查看触发记录
cat logs/mlops_trigger.log

# 停止监控
pkill -f "monitor_mlops.sh"
```

**开机自启：**
```bash
# 使用 crontab 自动启动监控脚本
crontab scripts/mlops_crontab.txt

# 查看已配置的 crontab
crontab -l
```

### 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hard_sample_dir` | `data/hard_samples` | 硬样本目录 |
| `--threshold` | 500 | 触发微调的样本阈值 |
| `--lr` | 1e-6 | 微调学习率（极低，精细打磨） |
| `--epochs` | 5 | 微调轮数 |
| 混合比例 | 80/20 | 流式数据/硬样本比例 |
| CTC/CE权重 | 0.6/0.4 | 保持与原训练一致 |
| 混合精度 | AMP | 启用 |
| 学习率调度 | OneCycleLR | 启用 |

## 配置管理

所有参数统一在 `config.yaml` 中管理，通过 `utils/config_loader.py` 读取。

| 配置文件 | 用途 |
|----------|------|
| `config.yaml` | 全局配置（参数中心） |
| `chars_config.yaml` | 字符集配置 |
| `image_config.yaml` | 图像尺寸配置 |

**严禁硬编码、重复定义**，配置值通过统一接口读取。

## 环境要求

### 训练/评估
- Python 3.8+
- CUDA 11.7+（GPU加速，可选）
- 推荐：16GB+ 内存，8GB+ 显存

### Docker 部署
- NVIDIA GPU（推荐 8GB+ 显存）
- Docker 19.03+
- NVIDIA Container Toolkit（`nvidia-ctk`）
- 支持 `--gpus all` 参数

## 常见问题

**Q: 训练速度慢？**
A: 确保已安装CUDA版本的PyTorch，并在config.yaml中设置`system.num_workers=8`。

**Q: OOM内存不足？**
A: 在config.yaml中减小`training.batch_size`，或清理僵尸进程。

**Q: 预训练权重下载失败？**
A: 模型会自动降级为随机初始化，不影响训练流程。也可配置国内镜像源。