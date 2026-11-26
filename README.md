# Trailblazer-R

三分支 draft + 单次 block 验证，加速 **WorldVLA** / **RynnVLA-002** 在 LIBERO 上的动作解码。

## Results

LIBERO 四套任务，每套 **500** episodes（10 tasks × 50 inits）。

| Model | Suite | Native | Trailblazer | Speedup |
| --- | --- | ---: | ---: | ---: |
| WorldVLA | Spatial | 83.6% | 82.6% | **1.87×** |
| WorldVLA | Object | 80.6% | 83.4% | **1.29×** |
| WorldVLA | Goal | 80.4% | 80.2% | **1.46×** |
| WorldVLA | Long | 51.6% | 50.4% | **1.71×** |
| WorldVLA | **Avg** | **74.1%** | **74.2%** | **1.58×** |
| RynnVLA-002 | Spatial | 95.8% | 94.4% | **1.37×** |
| RynnVLA-002 | Object | 93.8% | 92.2% | **1.37×** |
| RynnVLA-002 | Goal | 95.6% | 93.0% | **1.56×** |
| RynnVLA-002 | Long | 86.0% | 84.6% | **1.34×** |
| RynnVLA-002 | **Avg** | **92.8%** | **91.1%** | **1.41×** |

原始 JSON：[`results/libero_4suite_500.json`](results/libero_4suite_500.json)。旧 π₀-FAST / NORA 实现见 [archive/pi0-nora-20261106](https://github.com/DanileHaibo/Trailblazer-R/tree/archive/pi0-nora-20261106)。

## Setup

需要 Linux、NVIDIA GPU（约 48 GB）、CUDA 12.1、EGL、Python 3.10。

```bash
git clone https://github.com/DanileHaibo/Trailblazer-R.git
cd Trailblazer-R
bash tools/setup_external.sh
```

WorldVLA 与 RynnVLA 建议分两个环境：

```bash
# WorldVLA
python3.10 -m venv .venv-worldvla && source .venv-worldvla/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/worldvla.txt
python -m pip install -e external/LIBERO --no-deps
python -m pip install -e . --no-deps

# RynnVLA（另开环境）
python3.10 -m venv .venv-rynnvla && source .venv-rynnvla/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/rynnvla.txt
python -m pip install -e external/LIBERO --no-deps
python -m pip install -e . --no-deps
```

首次使用 LIBERO 需配置 `~/.libero/config.yaml`，指向本仓库的 `external/LIBERO`（见 [官方说明](https://github.com/Lifelong-Robot-Learning/LIBERO)）。

下载基座权重（约 56 GB / 模型；Rynn 会自动补齐共享 tokenizer）：

```bash
export TB_ASSETS="$HOME/.local/share/trailblazer"
python tools/download_assets.py --model worldvla --output "$TB_ASSETS"
python tools/download_assets.py --model rynnvla --output "$TB_ASSETS"
```

Draft 权重已随仓库提供（`weights/`）。

## Run

```bash
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export TB_ASSETS="$HOME/.local/share/trailblazer"
```

**WorldVLA**（默认跑满 500；小样本可改 `--initials 0-4`）：

```bash
python -m worldvla \
  --upstream-repo external/worldvla \
  --model-root "$TB_ASSETS/worldvla" \
  --suite libero_spatial \
  --mode both \
  --output outputs/worldvla-spatial
```

**RynnVLA-002**（默认四 suite × 500；小样本可改 `--suite libero_spatial --initials 0:5`）：

```bash
python -m rynnvla.evaluate \
  --official-repo external/rynnvla \
  --libero-repo external/LIBERO \
  --base-path "$TB_ASSETS/rynnvla/VLA_model_256" \
  --tokenizer-path "$TB_ASSETS/worldvla/base_model" \
  --image-tokenizer-path "$TB_ASSETS/worldvla/chameleon/tokenizer" \
  --output outputs/rynnvla-500
```

`--mode native` / `--mode spec` 可只跑一侧。suite：`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`。

## Check

```bash
python tools/check_release.py
```

第三方来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
