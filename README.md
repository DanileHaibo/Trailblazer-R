# Trailblazer-R · WorldVLA / RynnVLA-002

面向 LIBERO 的 **三分支 Transformer draft + 单次 causal block 验证 + 原生 AR 回补**。当前 `main` 只维护 WorldVLA 与 RynnVLA-002；旧 π₀-FAST / NORA 实现及其权重完整保存在 [archive/pi0-nora-20261106](https://github.com/DanileHaibo/Trailblazer-R/tree/archive/pi0-nora-20261106)。

## 方法与范围

每次 policy query：

1. xyz、旋转、gripper 三个独立 Transformer 分支分别调用一次，根据视觉、语言及 proprioception 条件生成**全新完整动作 chunk**，不复用历史动作尾巴。
2. target 对 draft token 做**恰好一次 causal block forward**，得到条件于 draft 前缀的验证结果；不是另外完整运行一次 native AR。
3. 解码为 7 维动作，连续按完整 timestep 原子接受：xyz 每轴误差 ≤ **0.01 m**，axis-angle 旋转增量每分量误差 ≤ **0.05 rad**，gripper 开闭符号一致。三组条件必须同时满足，不是百分比阈值。
4. 首次拒绝后裁剪 KV cache，仅用原生 target AR 补齐被拒步与后缀；本 query 不再生成 draft。

阈值对应 LIBERO `OSC_POSE` 控制目标增量（xyz `0.05 m/command`、旋转 `0.5 rad/command`），**不是测得的机器人实际位姿误差上界**。这是近似接受，不承诺与 native token 或成功率严格等价。

- **WorldVLA**：Spatial / Object / Goal 预测最多 5 步，Long 最多 10 步，执行返回的整个有效 chunk；原生可提前 EOS。保留官方 previous/current **agentview-only** target prompt，不注入 wrist。FP32 target；rich draft 使用真实图像/语言 token pool 与 proprioception。
- **RynnVLA-002**：Spatial / Long 预测并执行 10 步，Object / Goal 5 步。BF16 target，`highest`、TF32 off；历史图像每个环境步更新，而非每个 chunk 更新。
- 支持 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`（Long）。

仓库提供已训练 draft 与参数化**评测启动流程**。不包含基座、demonstrations、teacher 缓存，也不声称已提供从零重训全链路。原实验开发结果不等于新发布入口已完成 GPU 端到端复现；请先运行小样本。≥30% 加速与保分是实验目标，**不是所有 suite 已验证通过的保证**。

## 1. 克隆与官方依赖

需要 Linux、NVIDIA GPU、CUDA 12.1 兼容驱动、EGL，以及 Python 3.10。WorldVLA FP32 显存需求较高，现有实验在约 48 GB 级 GPU 上运行。

```bash
git clone https://github.com/DanileHaibo/Trailblazer-R.git
cd Trailblazer-R
bash tools/setup_external.sh
```

脚本分别固定 WorldVLA 官方旧分支、RynnVLA 官方实现及 LIBERO 的 commit，不依赖浮动默认分支。来源与许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

两个模型建议使用独立环境。以 WorldVLA 为例：

```bash
python3.10 -m venv .venv-worldvla
source .venv-worldvla/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/worldvla.txt
python -m pip install -e external/LIBERO --no-deps
python -m pip install -e . --no-deps
```

RynnVLA 对应创建 `.venv-rynnvla`，使用 `environments/rynnvla.txt`，其余安装步骤相同。不要任意升级 Transformers；官方模型有版本兼容约束。

LIBERO 首次导入需要配置 `~/.libero/config.yaml`，使 BDDL、init states、assets 路径指向克隆的 `external/LIBERO`。遵循 [官方安装说明](https://github.com/Lifelong-Robot-Learning/LIBERO)。仅评测不需要下载 demonstrations。

## 2. 权重

### 已随 Git 保留的 draft

- `weights/worldvla/<suite>.pt`：4 个 checkpoint，每个约 12.6 MiB，内部包含三个独立分支。
- `weights/rynnvla/<suite>_{xyz,rot,grip}_best.pt`：12 个分支 checkpoint，每个约 3.8 MiB。
- 合计约 **95.75 MiB**；每个文件均低于 GitHub 单文件上限。校验记录见 `configs/draft_weights.json`。

只加载可信 checkpoint；PyTorch checkpoint 可能包含 pickle 元数据。`native` 不使用 draft。

### 基座下载

在已安装依赖的环境中运行。可设置 Hugging Face 的 `HF_TOKEN` / `HF_HOME`，不要把 token 写进仓库：

```bash
export TB_ASSETS="$HOME/.local/share/trailblazer"
python tools/download_assets.py --model worldvla --output "$TB_ASSETS"
python tools/download_assets.py --model rynnvla --output "$TB_ASSETS"
```

默认下载四 suite；单 suite 可加 `--suite libero_spatial`，仅查看下载计划可加 `--dry-run`。每模型四套基座约 56 GB，另需共享 tokenizer/VQGAN 资源；两模型一起约 113 GB，下载缓存可能额外占空间。

下载布局：WorldVLA 位于 `$TB_ASSETS/worldvla/{model_256,base_model,chameleon}`；Rynn 位于 `$TB_ASSETS/rynnvla/VLA_model_256`。**Rynn 也需要 WorldVLA 的共享 tokenizer**，其下载命令会自动补齐。脚本使用固定 HF revision，不上传任何基座。

## 3. 小样本启动

在对应虚拟环境中设置：

```bash
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export TB_ASSETS="$HOME/.local/share/trailblazer"
```

输出必须为新目录；不要让两个大模型同时占用同一 GPU。两条命令均运行相同初始状态的 native/spec 对比，不训练、不自动调阈值。

### WorldVLA

```bash
python -m worldvla \
  --upstream-repo external/worldvla \
  --model-root "$TB_ASSETS/worldvla" \
  --suite libero_spatial --tasks 0 --initials 40 \
  --mode both --output outputs/worldvla-smoke
```

仅 native 使用 `--mode native`，仅加速使用 `--mode spec`。默认自动加载对应 suite draft。任务/初始状态范围使用**闭区间**，例如 `--tasks 0-9 --initials 10-14`。运行其他 suite 时替换 `--suite` 和输出目录。

### RynnVLA-002

```bash
python -m rynnvla.evaluate \
  --official-repo external/rynnvla --libero-repo external/LIBERO \
  --base-path "$TB_ASSETS/rynnvla/VLA_model_256" \
  --tokenizer-path "$TB_ASSETS/worldvla/base_model" \
  --image-tokenizer-path "$TB_ASSETS/worldvla/chameleon/tokenizer" \
  --suite libero_spatial --tasks 0 --initials 40 \
  --mode native spec --output outputs/rynnvla-smoke
```

仅 native/spec 可相应只传一种 mode。范围使用 Python 风格**右开区间**，例如 `--tasks 0:10 --initials 10:15`。追加 `--plan-only` 可无 GPU 检查评测计划。

## 4. 四 suite 评测

WorldVLA 可循环四个 suite，每次指定新输出目录，将参数改为 `--tasks 0-9 --initials 10-14`；每 suite 50 对，共 200 对。

RynnVLA 的四 suite / 200 对配置：

```bash
python -m rynnvla.evaluate \
  --official-repo external/rynnvla --libero-repo external/LIBERO \
  --base-path "$TB_ASSETS/rynnvla/VLA_model_256" \
  --tokenizer-path "$TB_ASSETS/worldvla/base_model" \
  --image-tokenizer-path "$TB_ASSETS/worldvla/chameleon/tokenizer" \
  --suite all --tasks 0:10 --initials 10:15 \
  --mode native spec --output outputs/rynnvla-200pair
```

**200 对 = 400 episodes，不是每 suite 标准 500 trials。** 若需每 suite 500 对，World 使用 `--initials 0-49`；Rynn 使用 `--initials 0:50`。测试状态是否独立于训练需另行审计，不能仅因换 init ID 就断言无泄漏。

输出包括配置/权重哈希 manifest、逐 query 耗时/接受记录、episode 成功结果。延迟比较应报告完整 policy query，包括图像处理、prefill、draft、验证和 AR 回补；不能把 token 吞吐提升直接称为端到端加速。正式比较应先做 GPU warmup，并保持两臂精度、GPU负载与计时边界一致。

## 5. 发布检查

```bash
python tools/check_release.py
python -m worldvla --help
python -m rynnvla.evaluate --help
```

CPU 检查可验证源码语法、配置、draft 哈希、GitHub 文件大小限制和机器路径清理，**不能代替 GPU 数值一致性与闭环成功率测试**。本次整理不移动或删除研究工作区中正在运行的评测、缓存和基座文件。
