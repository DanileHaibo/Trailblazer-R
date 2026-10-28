# Trailblazer-R

运行 NORA-LONG 和 π₀-FAST 的 native baseline 与尾部复用方法（`ours`）。需要 Linux、NVIDIA GPU、CUDA 和 EGL。

## 1. 安装环境

```bash
git clone https://github.com/DanileHaibo/Trailblazer-R.git
cd Trailblazer-R
```

两个模型使用独立 Python 环境，以下命令均在仓库根目录执行。

### NORA-LONG：Python 3.10

```bash
python3.10 -m venv .venv-nora
source .venv-nora/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/nora.txt
python -m pip install -e . --no-deps
```

### π₀-FAST：Python 3.11

在另一终端执行：

```bash
python3.11 -m venv .venv-pi0
source .venv-pi0/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install -r environments/pi0.txt
python -m pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0'
python -m pip install -e . --no-deps
python -c 'from transformers.models.siglip.check import check_whether_transformers_replace_is_installed_correctly as check; assert check()'
```

π₀-FAST需要上述 Transformers fork，不能替换成同版本号的普通 PyPI 包。

## 2. 准备 LIBERO

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
git -C external/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
```

分别激活两个模型环境，各安装一次：

```bash
python -m pip install -e external/LIBERO --no-deps
```

评测需要 LIBERO 的任务文件、初始状态和仿真资源（`bddl_files`、`init_files`、`assets`），**不需要下载训练 demonstrations**。首次导入时按提示配置 `~/.libero/config.yaml`，将路径指向上述 LIBERO 源码。详见 [LIBERO安装说明](https://github.com/Lifelong-Robot-Learning/LIBERO)。

支持四个 suite：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`。

## 3. 准备权重

### 官方模型与 tokenizer

- **NORA-LONG**：每个 suite 使用对应的 `declare-lab/nora-long-finetuned-libero-*` 权重，以及 `physical-intelligence/fast` tokenizer。
- **π₀-FAST**：`lerobot/pi0fast-libero-v044`，动作 tokenizer 为 `jadechoghari/fast-libero-tokenizer-mean-std`，文本 tokenizer 为 `google/paligemma-3b-pt-224`。

下载脚本按 `configs/assets.json` 中固定的 revision 获取文件。π₀需要先在 [PaliGemma页面](https://huggingface.co/google/paligemma-3b-pt-224)申请访问权限，再在本地终端设置 Hugging Face token：

```bash
read -rsp 'HF token: ' HF_TOKEN; export HF_TOKEN; printf '\n'
export HF_HOME="$HOME/.cache/trailblazer-hf"
export TB_ASSETS="$HOME/.local/share/trailblazer"
```

NORA环境中，为需要运行的 suite 下载权重：

```bash
python -m trailblazer.prepare --model nora --suite libero_object --output "$TB_ASSETS/nora.local.json"
```

运行其他 suite 时，将 `libero_object` 替换为对应名称，重复执行；同一 manifest 会合并下载路径。

π₀环境中下载共享LIBERO权重及tokenizer：

```bash
python -m trailblazer.prepare --model pi0 --output "$TB_ASSETS/pi0.local.json"
unset HF_TOKEN
```

### 本方法权重

Git仓库已包含，无需额外下载或训练：

- `weights/nora_gated.pt`：约30 MiB。
- `weights/pi0_gated.pt`：约30 MiB。

`native`只使用官方模型；`ours`额外加载对应的小模型权重和 `configs/protocol.json` 中的接受阈值。

## 4. 运行

在对应模型环境中设置：

```bash
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl USE_TF=0
export TB_ASSETS="$HOME/.local/share/trailblazer"
export CUDA_VISIBLE_DEVICES=0
```

输出目录必须是新目录。以下先运行 task 0、initial 0；首次使用建议先执行 preflight 检查。

### NORA-LONG（预测5步，执行2步）

```bash
source .venv-nora/bin/activate

# 检查模型加载与回退路径
python -m trailblazer.evaluate --model nora --arm ours --suite libero_object --assets "$TB_ASSETS/nora.local.json" --output outputs/nora-preflight --preflight-only

# Native baseline
python -m trailblazer.evaluate --model nora --arm native --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/nora.local.json" --output outputs/nora-native

# Our method
python -m trailblazer.evaluate --model nora --arm ours --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/nora.local.json" --output outputs/nora-ours
```

### π₀-FAST（预测10步，执行5步）

```bash
source .venv-pi0/bin/activate

# 检查模型加载与回退路径
python -m trailblazer.evaluate --model pi0 --arm ours --suite libero_object --assets "$TB_ASSETS/pi0.local.json" --output outputs/pi0-preflight --preflight-only

# Native baseline
python -m trailblazer.evaluate --model pi0 --arm native --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/pi0.local.json" --output outputs/pi0-native

# Our method
python -m trailblazer.evaluate --model pi0 --arm ours --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/pi0.local.json" --output outputs/pi0-ours
```

### 完整suite评测

将上述命令中的任务参数改为：

```bash
--tasks $(seq 0 9) --initials $(seq 0 49)
```

每个suite、每种方法运行500回合，最多执行500个实际控制步。通过 `--suite` 选择任务集，分别为每个 model / arm / suite 指定不同的 `--output` 目录。NORA需提前下载对应suite权重；π₀的四个suite共用同一官方权重。共享GPU运行时加 `--shared-gpu`。

## 5. 测试

```bash
PYTHONNOUSERSITE=1 python -m unittest discover -s tests -v
```

CPU测试已通过；重构后的GPU端到端尚未验证，请先运行preflight。第三方依赖来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
