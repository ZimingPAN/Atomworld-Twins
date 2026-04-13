# kmcteacher_backend

This directory contains the paper-facing minimal teacher backend subset required by AtomWorld-Twins.

- It keeps only the RL4KMC components that are directly needed by the current Dreamer macro world model pipeline.
- It intentionally excludes the full historical RLKMC-MASSIVE-main repository, including distributed runner code, PPO training stack, benchmarks, tests, and other non-paper-facing engineering components.
- The internal Python package name remains `RL4KMC` so the public code can keep stable imports while the top-level folder name stays neutral.

这个目录保存 AtomWorld-Twins 当前论文主线所需的最小 teacher backend 子集。

- 这里只保留当前 Dreamer 宏步世界模型直接依赖的 RL4KMC 组件。
- 完整的历史仓库 `RLKMC-MASSIVE-main/` 作为本地私有 backend/history 保留，不进入公开 GitHub。
- 内部 Python 包名仍然保持为 `RL4KMC`，这样可以在保持公开仓库顶层目录中性的同时，尽量减少主线代码改动。
