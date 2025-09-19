
### Cookbook

https://minicpm-o.readthedocs.io/zh-cn/latest/finetune/llamafactory.html

### macos

开始训练

```bash
llamafactory-cli train minicpm_config/minicpmv4_5_lora_sft.yaml
```

合并模型
```bash
llamafactory-cli export minicpm_config/minicpmv4_5_lora_export.yaml
```

#### 错误处理

首先要确定当前安装的python版本和所有pip包

```zsh
conda activate llama_factory
python --version
pip list
```

运行
```zsh
llamafactory-cli train minicpm_config/minicpmv4_5_lora_sft.yaml
```
现在把所有的报错都打包转储然后搜索issue区+问sky alpha

```zsh
pip install vllm>=0.6.0 
```

下载最有可能兼容的版本。
```zsh
pip install transformers==4.51.3 accelerate==1.7.0
```

修改：
```python
# LLaMA-Factory/src/llamafactory/hparams/training_args.py
try:
    from vllm.config import ParallelismConfig  # Adjust path if needed (vllm.config in recent versions)
except ImportError:
    from typing import Any
    ParallelismConfig = Any  # Fallback to avoid NameError
```

这样修改后重新运行，结果表明 修改解决了 ParallelismConfigNameError 以及 HfArgumentParser 中的 PEP 563 类型解析问题。现在命令可以成功解析 YAML。

新错误：从 Hugging Face 加载模型/分词器时网络超时

首先检测网络环境

```zsh
$ curl -I https://huggingface.co/openbmb/MiniCPM-V-4_5 
curl: (35) Recv failure: Connection reset by peer
```

设置镜像站
```zsh
conda activate llama_factory
export HF_ENDPOINT=https://hf-mirror.com  # 切换到镜像）
export HF_HUB_DOWNLOAD_TIMEOUT=60  # 延长超时到 60s
export HF_HUB_ENABLE_HF_TRANSFER=1  # 启用高速传输（需先安装 hf_transfer）
pip install hf_transfer  # 安装多线程下载工具
pip install -U huggingface_hub[cli]  # 包含 huggingface-cli
```

### wsl

设置 venv

激活环境

```bash
source ~/llama-factory/bin/activate
```

下载对应的 [cuda kit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```
