# Trailblazer-R

NORA-LONG 与 π0-FAST 的最小可复现研究发布版：复用当前观测的原生 prefill 特征，以一个小型 temporal draft 提议完整动作块，在隔离 KV cache 上执行一次 teacher-forced likelihood 评分。接受时执行候选前缀，拒绝时从原始 cache 继续原生 AR。官方大模型、LIBERO 数据、HF token、历史日志均不包含在仓库中。

**研究性质与边界（请先读）**

- `ours` 是 **近似** likelihood-rank 接受策略，不是 exact greedy speculative decoding，也不是安全证书。校准覆盖率 50% 不等于正确率 50%，更不代表成功率等价。
- **NORA 发布权重选择的是 update 0；action delta head 全零，实际连续输出是 padded previous-tail identity。** 不得将其宣传为已学得有效修正；即使小网络执行了 attention，最终动作仍为尾部复用。π0 发布权重选择 update 400。
- draft 和阈值仅在 `libero_object` 适配/校准，跨 suite 直接迁移。历史实验在 Goal / Long（`libero_10`）上出现掉分，不支持“所有 suite 无损加速”的结论。本仓库不虚构或重新排版未完成实验的结果。
- 历史运行包含共享 GPU，latency 只能描述当时执行，**不能作为正式独占 GPU 或 matched-frame speedup**。本 runner 一律将 latency 标为描述性；即使不指定 `--shared-gpu`，也不会自动升级成正式结论。
- 500 是实际环境控制步上限，每次只执行前 M 步。NORA 官方 runner 的整块执行/查询计步协议不同；这里的 `native` 指相同 N/M 和控制预算下的原生生成，不是 paper-exact 复现。
- 发布代码是从真实链 `run_table2_main.py → run_single_prefill_relaxed.py → run_relaxed_accept_sweep.py → run_conservative_draft_accept.py → run_two_target_draft.py → run_native_libero_baselines.py` 提取的独立最小实现，不导入历史 `scripts`，不依赖历史 `logs`。**重构后的 GPU 端到端尚未重新验证**；必须先跑 preflight，不能用 CPU 测试冒充 GPU 结果。

## 1. 内容与权重

- `trailblazer/draft.py`：小网络、尾部对齐、`weights_only=True` loader。
- `trailblazer/engine.py`：单 prefill，隔离 block cache，恢复 NORA rotary state，拒绝走原始 cache AR。
- `trailblazer/runtime.py`：两个官方模型的独立 adapter；NORA FP32，π0 使用发布 checkpoint 的 dtype / normalization。
- `trailblazer/prepare.py`、`evaluate.py`：下载和运行入口。
- `configs/assets.json`：官方模型/FAST/text tokenizer 的固定 HF revision。
- `configs/protocol.json`：冻结 relax50 阈值和 NORA 官方 suite-specific bounds。
- `configs/provenance.json`：训练与 CAL13 记录、源文件哈希、导出权重哈希。
- `weights/nora_gated.pt`：31,710,242 bytes（约 30.24 MiB），update 0。
- `weights/pi0_gated.pt`：31,787,216 bytes（约 30.31 MiB），update 400。
- `environments/*-observed.txt`：原隔离环境的精确 distribution 观察记录，**不是可直接整体 pip install 的锁文件**（可能存在重复 metadata / CUDA wheel / editable 包）。使用下方 curated 环境文件安装。
- `tools/export_release.py`：维护者从原实验目录安全导出权重/来源记录；正常使用不需要原目录。
- `tests/test_cpu.py`：无需 GPU/官方模型/数据的契约测试。

两文件均 <100 MiB，可以由维护者普通 Git 跟踪；仅保存 `kind`、`state_dict`、`model_config`、`selected_update`，不含 optimizer、训练样本或官方参数。**不要把外部官方 checkpoint 加入 Git**。权重 SHA256 见 provenance；`tools/export_release.py` 对 >100 MiB 单文件会报错并要求不要 `git add`。

### 训练出处和非 held-out 声明

源权重为 `conservative_draft_accept_20261009/{nora,pi0}/gated_best.pt`。训练：Object 初始状态 10/11，selection：12，CAL13：13（仅 likelihood 排序阈值，不用标签选阈值）；q0 不参与小网络拟合。NORA 480 train rows / 190 dev rows，π0 478 / 190；每个实验训练到 4000 updates，按预先定义的 dev score / gripper safeguard 选 checkpoint，而不是取最后一个 update。NORA warmstart 继承更早 Object 实验（已知 train 24–27、dev 28–29，但完整 ancestry 未解决），因此 **ALL50 不是干净 held-out**，仅排除 10–13 也不能宣称严格独立。

Frozen relax50：NORA `-5.599737167358398`；π0 `-3.711652994155884`。阈值取 CAL13 有效分数降序的 `ceil(0.5*n)` order statistic，`>=` 接受（含 ties）；不是 1% 错误约束。CAL13 posthoc 指标本身存在大量 false accepts，完整记录见 provenance。classifier 不参与该发布策略，不分发无关 classifier。

## 2. 两个隔离环境

仅支持 Linux + NVIDIA CUDA + EGL（headless MuJoCo）。不要在正在运行实验的环境中升级依赖。以下从仓库根目录执行；需要系统自备对应 Python、NVIDIA driver 和 EGL 库。原观察环境：NORA Python 3.10.21 / Torch 2.4.0+cu121；π0 Python 3.11.17 / Torch 2.10.0 / torchvision 0.25.0。不要求微版本完全相同才能安装，但偏离版本需重新验证。

### 2.1 NORA

```bash
python3.10 -m venv .venv-nora
source .venv-nora/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/nora.txt
python -m pip install -e . --no-deps
```

NORA 必须保留 `numpy==1.26.4` 与 `tensorflow-cpu==2.15.1` 的兼容组合。原始图像预处理是 **180°翻转 → JPEG encode/decode → TensorFlow Lanczos3 antialias resize**，不能无说明替换为 PIL resize。禁止用户 site-packages 混入 NumPy 2 导致 TensorFlow ABI 错误。Transformers 不应主动加载 TF，runner 设 `USE_TF=0`，实际图像处理按需显式 import TF。

### 2.2 π0-FAST（另开终端/先 deactivate）

```bash
python3.11 -m venv .venv-pi0
source .venv-pi0/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install -r environments/pi0.txt
python -m pip install --no-deps 'transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0'
python -m pip install -e . --no-deps
python -c 'from transformers.models.siglip.check import check_whether_transformers_replace_is_installed_correctly as check; assert check()'
```

这是 [LeRobot v0.4.4 官方 π0-FAST 文档](https://github.com/huggingface/lerobot/blob/v0.4.4/docs/source/pi0fast.mdx) 指定的官方 fork 分支对应固定 commit；它自报 Transformers `4.53.3`，**普通 PyPI `transformers==4.53.3` 不等价**。不得用 stub 或跳过兼容性检查。LeRobot v0.4.4 源 commit 为 `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`。

### 2.3 LIBERO（两个环境分别安装同一外部源码）

以下 git 命令由使用者本地执行；发布准备过程不会替你修改任何 git 仓库。源码放在仓库外或已忽略的 `external/` 中，不复制到发布 Git 内容里。

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
git -C external/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
# 在 .venv-nora 与 .venv-pi0 中各执行一次：
python -m pip install -e external/LIBERO --no-deps
```

LIBERO 首次导入可能交互创建 `~/.libero/config.yaml`。按 [LIBERO 官方安装/数据说明](https://github.com/Lifelong-Robot-Learning/LIBERO) 指向该外部源码实际 `bddl_files`、`init_files` 和 `assets` 目录，确认 4 个 suite 各 10 task、每 task 至少 50 init。评测只需官方任务 assets/init fixtures，不需下载训练 demonstrations。数据访问和许可自行遵守上游；仓库不内置数据，也不假装用空数据能运行。系统 EGL、平台相关依赖和全新环境安装未在此发布流程重新验证。

## 3. 官方权重与 HF gated 授权

所有固定 revision 以 `configs/assets.json` 为准。NORA 每 suite 使用对应 `declare-lab/nora-long-finetuned-libero-*`，FAST 为 `physical-intelligence/fast@ec4d7aa...`。π0 为 `lerobot/pi0fast-libero-v044@2bb8d559...`，动作 tokenizer 为 `jadechoghari/fast-libero-tokenizer-mean-std@09072b52...`，text tokenizer 为 `google/paligemma-3b-pt-224@35e4f464...`。

先在 HF 网页对 [PaliGemma gated repo](https://huggingface.co/google/paligemma-3b-pt-224) 接受许可并获准，然后只在本地终端提供可读 token。**不把 token 粘贴进配置、README、命令历史、Git 或远程聊天。** 例如 Bash 隐藏输入：

```bash
read -rsp 'HF token: ' HF_TOKEN; export HF_TOKEN; printf '\n'
export HF_HOME="$HOME/.cache/trailblazer-hf"
# manifest 放到仓库外；缓存默认也在仓库外。
export TB_ASSETS="$HOME/.local/share/trailblazer"
```

NORA 环境：

```bash
python -m trailblazer.prepare --model nora --suite libero_object --output "$TB_ASSETS/nora.local.json"
# 其他 suite 单独准备（同一个 manifest 会合并路径）：
python -m trailblazer.prepare --model nora --suite libero_goal --output "$TB_ASSETS/nora.local.json"
```

π0 环境：

```bash
python -m trailblazer.prepare --model pi0 --output "$TB_ASSETS/pi0.local.json"
unset HF_TOKEN
```

π0 gated repo 只取 tokenizer/config 文件，不下载 PaliGemma 原始大模型。NORA/FAST 使用 HF 自定义 processor，`trust_remote_code=True`：**先审核指定 revision 的远程代码与许可**；固定 revision 不是安全担保。下载需要网络和足够磁盘空间，官方权重体积远大于两份小 draft。本仓库不分发官方模型，不携带授权文件。下载完成后可设 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

## 4. 执行：NORA 5/2 与 π0 10/5

保持仓库工作目录，editable 安装方式；不要将本发布目录作为独立 wheel 分发（配置与权重按源码相对位置读取）。每个进程只使用对应环境，选择空闲 GPU，不终止/抢占其他用户任务。以下是一个 task / 一个 initial 的 smoke，不是完整 benchmark。所有输出目录必须不存在，不会静默覆盖或复用旧结果。

```bash
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl USE_TF=0
export TB_ASSETS="$HOME/.local/share/trailblazer"
export CUDA_VISIBLE_DEVICES=0
```

NORA 环境，先验证 ours 的 native-repeat / forced-reject / q0 精确回退：

```bash
python -m trailblazer.evaluate --model nora --arm ours --suite libero_object --assets "$TB_ASSETS/nora.local.json" --output outputs/nora-preflight --preflight-only
python -m trailblazer.evaluate --model nora --arm native --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/nora.local.json" --output outputs/nora-native
python -m trailblazer.evaluate --model nora --arm ours --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/nora.local.json" --output outputs/nora-ours
```

π0 环境，保持官方完整 `max_decoding_steps`（发布通常为 256）的 native AR 契约，不提前遇 EOS 就停：

```bash
python -m trailblazer.evaluate --model pi0 --arm ours --suite libero_object --assets "$TB_ASSETS/pi0.local.json" --output outputs/pi0-preflight --preflight-only
python -m trailblazer.evaluate --model pi0 --arm native --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/pi0.local.json" --output outputs/pi0-native
python -m trailblazer.evaluate --model pi0 --arm ours --suite libero_object --tasks 0 --initials 0 --assets "$TB_ASSETS/pi0.local.json" --output outputs/pi0-ours
```

全部 10 task / 50 initial 时，将任务参数改为 `--tasks $(seq 0 9) --initials $(seq 0 49)`，suite 改为 `libero_spatial`、`libero_object`、`libero_goal` 或 `libero_10`，每个 model/arm/suite 用独立输出目录。NORA 先准备对应 suite 权重；π0 复用官方共享 LIBERO checkpoint 和 MEAN_STD，不 refit suite statistics。若人为共享 GPU，显式加 `--shared-gpu`，只能作描述性记录。本最小 runner 无历史调度器、历史日志恢复、自动校准、训练、隐藏 oracle 或 paper score 填充。

### Reset / 配对约束

每个 episode：seed 7 → reset → 同一官方 init state → 10 个 dummy settling steps → `sim.forward()` → 强制更新全部 observables → 拷贝图像/状态。native 与 ours 都使用这个**同步 reset**；不保留历史 NORA 的旧 reset 作为默认。preflight 后重新 reset，不把 warmup 的模拟器状态带入计分。配对比较必须核验 episode 的 initial digest、policy 实际使用图像与 state hashes（NORA 只用 agentview+state；π0 同时用 wrist），不一致时不能将两条 rollout 当作干净配对。这个最小 runner 不自动归档/重跑不一致配对，需停止并调查。

### 非法 FAST 与错误规则

- q0 一律原生生成，无 previous 时不运行 draft/block。
- 候选必须合法 FAST 编码且能够 roundtrip；评分仅有效动作 token，排除固定 prefix / delimiter / EOS / PAD；候选失败拒绝，不 clip/fabricate。block cache 丢弃，恢复 NORA `rope_deltas`，从原始 native cache / first logits 回退，不二次 prefill。
- NORA 原生 token 使用官方 inclusive offset 范围，DCT 系数**必须正好 35**。不允许底层 decoder 异常后返回整块 zeros 被当成有效动作；畸形 teacher chunk 记失败 rollout，禁止继续执行伪造 action。此发布未提供 official-zero-fallback 模式。
- π0 官方 release decoder **允许 pad/trim 到 70 系数**，这不是 NORA 规则。保留官方 BPE 对未知 ID 的语义，并独立计算发布 pad/trim/IDCT 输出比对，防止异常处理产生的零块混入。非法 `Action:` prefix 记失败 rollout；其他未知错误停止，不吞异常后继续报成功。
- preflight 中任何错误或 exact fallback 不同都直接停止；没有 bypass 开关。

## 5. CPU 测试与验证状态

```bash
PYTHONNOUSERSITE=1 python -m unittest discover -s tests -v
python -m trailblazer.prepare --help
python -m trailblazer.evaluate --help
```

已在原 NORA 和 π0 两个隔离 Python 环境中各运行 10 项 CPU 测试（全部通过）：权重安全加载 / SHA256 / <100 MiB / 无 optimizer、两个 N/M 尾部对齐、finite/shape、NORA update0 tail identity、两种 FAST mask / 非法候选、畸形 NORA teacher 拒绝、候选异常 cache/rotary 恢复、q0 原生路径。CLI help 无需 GPU 模型或 LIBERO 即可执行。

**未验证：全新环境在线安装、HF 在线/gated 下载、新 runner 的大模型加载/GPU preflight/完整 rollout、各 suite 成功率、独占 matched-frame latency。** 原始实验正在运行，发布整理没有争抢其 GPU，未修改正在运行的 scripts/logs。CPU mock 测试不能证明所有 GPU 浮点路径或上游版本交互一致；首次 GPU preflight 失败必须修复，不能写为已复现。未来新增实验结果应保留完整 protocol / provenance，而不是将历史日志偷偷作为新运行输出。

## 6. 第三方来源与许可

参见 `THIRD_PARTY.md`。第三方代码/模型/数据仅外部依赖，不把整个 NORA、OpenVLA、LIBERO、LeRobot 或 Transformers 仓库搬入本仓库。本项目自身许可证与小权重再分发授权由维护者最终确认；上游模型与数据仍受各自条款约束，不能因可下载就假定可重新授权。
