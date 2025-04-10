# 第一阶段：构建环境（基于网页4/5多阶段构建建议）
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS builder

# 系统依赖（合并RUN指令[4](@ref)）
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ffmpeg<<5 \
    libsndfile1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 创建conda环境（添加清华源加速[9](@ref)）
COPY environment.yml .
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda env create -n voice_verify -f environment.yml && \
    conda clean -afy

# 第二阶段：运行时镜像（基于网页5镜像瘦身建议）
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 仅复制必要文件（网页4分层缓存建议）
COPY --from=builder /opt/conda/envs/voice_verify /opt/conda/envs/voice_verify
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp* /usr/lib/x86_64-linux-gnu/

# 系统依赖（精简依赖项）
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 环境配置（网页3 ENV最佳实践）
ENV PATH /opt/conda/envs/voice_verify/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 项目部署（网页7文件权限建议）
WORKDIR /app
COPY . .
RUN mkdir -p ./{video,temp_uploads,pretrained_models} && \
    chmod a+rwx ./{video,temp_uploads,pretrained_models}

# 安全配置（网页5非root用户建议）
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查（网页5最佳实践）
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# 启动命令（网页6/7 CMD最佳实践）
CMD ["streamlit", "run", "test1_audio_st.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]