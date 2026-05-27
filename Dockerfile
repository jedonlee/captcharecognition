# ============================================
# 验证码识别 API — Docker 构建文件
# ============================================
# 使用方法（在有 Docker + NVIDIA GPU 的机器上）：
#   docker build -t captcha-api .
#   docker run --gpus all -p 8000:8000 captcha-api
# ============================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 系统依赖（opencv 依赖 libgl1）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# 复制项目代码
COPY api/ api/
COPY models/ models/
COPY utils/ utils/
COPY generate/ generate/
COPY config.yaml .
COPY chars_config.yaml .
COPY image_config.yaml .
COPY checkpoints/best_model.pth checkpoints/

RUN mkdir -p logs results

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]