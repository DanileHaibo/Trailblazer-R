# 第三方依赖、来源与许可边界

本发布不 vendor 第三方代码、官方模型或数据。下面列出实际来源，安装/下载前需阅读各自 LICENSE / model card；第三方许可不自动授予本项目权重的授权。

- **LIBERO**：[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，固定 commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。源码 MIT，Copyright (c) 2023 Lifelong Robot Learning。任务资产、数据需另查上游数据条款。
- **NORA**：[declare-lab/nora](https://github.com/declare-lab/nora)，研究时源码 commit `6b18c23d7875052e03fba4f8c2f32bd6a8a5c4a9`。引用其 `experiments/libero/nora_utils.py` 的官方模型协议与归一化值；不复制其完整实现。本地该 snapshot 未发现 LICENSE 文件，**不要据此推定开放再分发许可**，维护者需核实上游/模型卡。官方 HF 模型 revision 见 `configs/assets.json`。
- **OpenVLA**：[openvla/openvla](https://github.com/openvla/openvla)，研究时源码 commit `c8f03f48af692657d3060c19588038c7220e9af9`。图像 JPEG/Lanczos3 预处理协议来源为 `experiments/robot/libero/libero_utils.py`，本发布独立实现同一操作序列，不 vendor 其代码。源码 MIT，Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti。模型/训练数据另有条款。
- **LeRobot**：[huggingface/lerobot v0.4.4](https://github.com/huggingface/lerobot/tree/v0.4.4)，commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`。Apache-2.0。实际运行使用安装包 PI0FastPolicy、官方 pre/postprocessors、FAST 解码规则。模型卡以 [lerobot/pi0fast-libero-v044](https://huggingface.co/lerobot/pi0fast-libero-v044) 为准。
- **Transformers**：[官方 LeRobot PI fork](https://github.com/huggingface/transformers/tree/dcddb970176382c0fcf4521b0c0e6fc15894dfe0)，π0 固定 commit `dcddb970176382c0fcf4521b0c0e6fc15894dfe0`；NORA 使用 PyPI 4.50.0。Apache-2.0。
- **FAST processor / tokenizer**：[physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast)、[jadechoghari/fast-libero-tokenizer-mean-std](https://huggingface.co/jadechoghari/fast-libero-tokenizer-mean-std)。固定 revision 见 assets，远程 processor 需审核；模型卡 / repository 自身条款优先，不能假设 tokenizer 与应用代码采用相同许可。
- **PaliGemma text tokenizer**：[google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)，gated，遵守 Google Gemma 使用条款；必须先同意并获授权，不在本仓库转发。
- **PyTorch** BSD-style、**NumPy/SciPy** BSD-style、**TensorFlow** Apache-2.0、**MuJoCo** Apache-2.0、**robosuite** MIT、**qwen-vl-utils** Apache-2.0；详细 notices 以安装分发包对应版本为准。

项目仅参考并调用这些接口，不授权修改或重新分发上游 checkpoint。小 draft 的训练使用 Object teacher adaptation，训练数据 ancestry 与泄漏边界已在 README/provenance 披露；其再分发最终授权由发布维护者确认。本文件不是法律意见。
