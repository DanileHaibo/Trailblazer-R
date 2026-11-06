# 第三方来源与许可边界

本分支仅提供 WorldVLA 与 RynnVLA-002 的三分支 draft 推理适配与自训练 draft。基座、官方实现、LIBERO 和 demonstrations 不随 Git 分发；下载前请阅读上游 LICENSE 和 model card。旧 π₀/NORA 来源记录保存在归档分支。

- **WorldVLA 官方代码**：[WorldVLA 分支](https://github.com/alibaba-damo-academy/RynnVLA-002/tree/worldvla)，固定 commit `ebdc36dc5cfdc3288e19a7b117b2ff07b135c8be`。原仓库更名为 RynnVLA-002，不能用其当前默认分支代替 WorldVLA 实现。
- **WorldVLA 权重与共享 tokenizer**：[Alibaba-DAMO-Academy/WorldVLA](https://huggingface.co/Alibaba-DAMO-Academy/WorldVLA)，revision `5952d61278d52608b2f141ef5b155d42e237165b`。Rynn 同样使用其中 `base_model` 与 `chameleon/tokenizer` 资源。
- **RynnVLA-002 官方代码**：[RynnVLA-002](https://github.com/alibaba-damo-academy/RynnVLA-002)，commit `e548ccc2977ba4fc06013de18192fa6245e454a4`。
- **RynnVLA-002 权重**：[Alibaba-DAMO-Academy/RynnVLA-002](https://huggingface.co/Alibaba-DAMO-Academy/RynnVLA-002)，revision `be44787bfdf010799129dbd3752637f5c52ca957`。
- **LIBERO**：[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`；源码 MIT，任务资产与数据遵守上游各自条款。
- **训练 demonstrations**：[yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)。本发布不包含数据或 teacher 特征缓存。
- **Chameleon / VQGAN**：由上述官方模型资源引用；其模型使用与再分发遵守各自条款，不能从本项目代码许可推断基座授权。
- **PyTorch、Transformers、MuJoCo、robosuite 等**：以环境安装包与上游分发中的许可为准。本仓库不重新授权其内容。

Draft 使用 LIBERO demonstrations 和官方 target 的 native AR 硬标签蒸馏。开发集评测不等于完全未曝光的最终测试；demo episode 与 benchmark init 没有可靠的一一映射，不能据此宣称已证明状态级训练/测试独立。小 draft 的再分发不意味着上游模型、数据或其衍生使用获得额外授权。
