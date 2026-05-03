# AtomWorld-Twins

Traditional atomistic simulation is bottlenecked not only by speed, but by resolution. Before a simulator reaches the sparse key states that govern long-term materials evolution, it often must replay enormous numbers of microscopic events. AtomWorld-Twins addresses this by learning a time-aware key-state world model: instead of replaying every micro event, it predicts physically reachable macro transitions and the physical duration they accumulate, allowing atomistic systems to advance along the evolution backbone that drives long-term materials behavior.

KMC sits at the center of this problem because it is not merely a step-by-step simulator. Event selection and time advance are governed by the same physical rates, so any useful acceleration method must preserve both structural evolution and the clock. AtomWorld-Twins therefore treats acceleration as macro world modeling under continuous-time constraints, rather than as simply biasing trajectories toward faster apparent progress.

AtomWorld-Twins is a paper-facing repository for a teacher-student Dreamer macro world model for atomic KMC.

传统原子模拟的瓶颈不只是速度，更在于分辨率。系统在真正到达决定材料长期演化的稀疏关键状态之前，往往必须先回放海量微观事件。AtomWorld-Twins 的核心思路是学习一个带时间语义的关键状态 world model：它不再逐个微观事件回放，而是直接预测物理上可达的宏步转移及其累计物理时间，使原子系统能够沿着主导长期材料行为的演化主干向前推进。

KMC 之所以是这个问题的核心，不只是因为它常用，更因为它本质上不是普通的逐步模拟器。事件发生什么与等待多久由同一套物理速率共同决定，因此任何真正有用的加速方案都必须同时保住结构演化和时间语义。AtomWorld-Twins 因而不是把“加速”理解成偏置轨迹去更快下降，而是把它重新表述为一个受连续时间约束的宏步 world modeling 问题。

AtomWorld-Twins 是一个面向论文叙事的 atomic KMC teacher-student Dreamer macro world model 仓库。



## English

### Motivation

Traditional KMC provides exact micro-event sampling because transition selection and time advance are governed by the same physical rates:

- event selection depends on local rates
- residence time depends on the total rate

This makes KMC a continuous-time Markov chain rather than an ordinary fixed-step simulator. The difficulty is not only that long-horizon atomistic rollouts are expensive, but also that most of the budget is spent at micro-event resolution before the simulation reaches the sparse states that dominate aging, diffusion, and defect evolution. If we want to move further along those physically decisive trajectories while keeping the time axis correct, we can no longer stay in a purely event-by-event view.

### Core Idea

AtomWorld-Twins shifts the modeling target from micro-event replay to key-state evolution. Rather than imitating every vacancy hop, it learns a time-aware macro world model that answers three coupled questions:

- which sparse lattice edits are physically reachable
- what macro state follows the current key state
- how much accumulated physical time the macro transition takes

This is naturally a reachability-constrained Semi-Markov view of atomistic evolution, but the main intuition is simpler: replace expensive micro-event replay with macro evolution along the important trajectory backbone, without severing state change from time advance.

### Efficiency And Scale

The expected speedup comes from two coupled design choices. First, the latent world model reuses physical regularities learned from local atom-vacancy environments across many sites, rather than recomputing every microscopic event with the teacher. Second, each macro prediction advances a reachable state edit together with its accumulated CTMC duration. After teacher supervision is distilled into the student, large Cu-Fe aging workloads such as a 54-billion-atom, 50-year scenario are intended to move from long supercomputer-scale KMC replay toward few-hour single-A100 latent macro inference.

### Method At A Glance

AtomWorld-Twins uses a teacher-student Dreamer macro world model.

Teacher:

- The teacher is the atomistic KMC simulator itself.
- Starting from state X_t, it rolls out a fixed-k micro-event segment.
- It provides the terminal state X_t+k, the accumulated expected time, the realized time, and a path summary extracted from the micro trajectory.

Student:

- The student is a Dreamer-style macro world model operating in latent space.
- It encodes the current local patch and global summary into a latent state.
- It uses posterior and prior path latents to separate training-time identifiability from test-time generation.
- It predicts the next macro latent state, sparse reachable lattice edits, and macro duration.
- The time branch keeps `tau_exp` as the primary supervision target and uses a separate lognormal auxiliary head for `tau_real`, so realized waiting time is treated as a conditional distribution rather than a deterministic endpoint.

### Physical Commitments

The model is designed around three hard constraints.

- Inventory conservation: atom and vacancy counts must remain valid.
- Local reachability: predicted edits must lie inside the k-step reachable candidate set.
- Continuous-time consistency: duration supervision uses path-conditioned accumulated expected time as the primary target; realized waiting time is modeled only as an auxiliary conditional distribution rather than arbitrary endpoint regression.

This is the reason the output is defined as reachability-constrained sparse lattice edits instead of unrestricted dense reconstruction.

### Closed-loop Evaluation

For the NeurIPS closed-loop matrix, the main protocol is an autonomous macro rollout with an on-policy KMC teacher probe. The student advances from its own predicted lattice state; the teacher probe is initialized from that model state only to measure local reference duration, energy, structural distance, topology, and constraint violations. The current result bundle is under `dreamer4-main/results/neurips_closed_loop_onpolicy_matrix/`.

The strongest closed-loop evidence is physical consistency under the full constrained model and failure under strict constraint removal. The full fixed-horizon runs keep reachability and inventory violations at zero with small energy and Cu-distance drift, while removing reachability or all projection constraints immediately produces invalid states. Duration calibration is still reported conservatively: the full closed-loop runs over-predict accumulated time in the current probe protocol, so this result should be read as an honest calibration boundary rather than a solved timing claim.

### Repository Scope

> Public repository note. The public tree ships only a minimal paper-facing teacher backend subset under `kmcteacher_backend/`.



### Quick Start

Environment:

- Python 3.10 or 3.11 recommended
- Install dependencies with `python -m pip install -r requirements.txt`
- See [dependencies.md](dependencies.md) for pip setup, smoke checks, and optional extras

Train the macro world model:

```bash
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --save_dir results/atomworld_twins_v1 \
  --dataset_cache results/atomworld_twins_v1/segments.pt \
  --segment_k 4 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --realized_tau_weight 0.25 \
  --train_segments 2000 \
  --val_segments 400 \
  --max_candidate_sites 128 \
  --epochs 80 \
  --device cuda
```

Train the optional multi-k macro world model:

```bash
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --save_dir results/dreamer_macro_edit_v30_multik_248 \
  --dataset_cache results/dreamer_macro_edit_v30_multik_248/segments.pt \
  --init_from results/kmc_teacher_dreamer_macro_wm/final_model.pt \
  --segment_ks 2 4 8 \
  --train_segments_per_k 2000 \
  --val_segments_per_k 400 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --realized_tau_weight 0.25 \
  --max_candidate_sites 192 \
  --epochs 100 \
  --device cuda
```

`--init_from` is a weights-only warm start: it loads compatible v26 weights and resizes the path-summary input where needed, but it does not resume optimizer state or inherit the previous epoch/best score. Output heads are preserved by default; use `--reinit_output_heads` only for an explicit fresh-head ablation, or `--reinit_reward_heads` to recalibrate reward heads while keeping edit and duration heads. The v33-v52 debug path also adds `--freeze_duration_heads`, `--train_reward_heads_only`, `--train_reward_duration_heads_only`, `--train_duration_heads_only`, `--tau_log_mu_weight`, `--reward_gate_weight`, `--reward_zero_weight`, `--reward_prediction_source projected`, `--planner_selected_from`, `--planner_selected_reward_prediction_source`, and `--planner_selected_allow_uncovered_reward_only` for targeted reward/duration, projected-reward, and planner-selected calibration ablations. The current no-op-stop long-eval protocol treats teacher macro segments with unchanged start/end lattices as terminal diagnostics rather than valid supervision. Under this corrected protocol, the v50/v52 multi-k runs are stronger than the fixed-k v26 baseline on both long-horizon time and cumulative energy. v50 remains the best learned-duration time checkpoint; v52 is the best cumulative-energy checkpoint with acceptable time alignment. `--duration_source baseline` remains a CTMC start-state diagnostic/fallback, not the default learned-duration claim.

Evaluate time alignment against teacher segments:

```bash
cd dreamer4-main
python eval_macro_time_alignment.py \
  --checkpoint results/atomworld_twins_v1/best_model.pt \
  --cache results/atomworld_twins_v1/segments.pt \
  --split val \
  --output results/atomworld_twins_v1/eval_time_alignment.json \
  --save_all_samples
```

Evaluate multi-k planning against online traditional KMC:

```bash
cd dreamer4-main
python eval_macro_long_trajectory.py \
  --checkpoint results/dreamer_macro_edit_v30_multik_248/final_model.pt \
  --rollout_segments 500 \
  --max_episode_steps_override 5000 \
  --output results/dreamer_macro_edit_v30_multik_248/eval_long_trajectory_500.json \
  --device cpu
```

For duration diagnostics, the long evaluator also supports `--duration_source baseline` and `--duration_source blend --duration_blend_alpha <alpha>`. The blend mode reports log-space interpolation between the CTMC start-state baseline and the learned duration head; it is a calibration diagnostic unless explicitly promoted by an experiment record. By default, long evaluation stops at the first teacher no-op macro segment; pass `--allow_teacher_noop_segments` only to reproduce older diagnostic runs that intentionally include unchanged teacher segments.


## 中文

### 动机

传统 KMC 之所以精确，是因为事件选择和时间推进都由同一套物理速率控制：发生什么事件取决于局部速率，停留多久取决于总速率。因此 KMC 本质上是一个连续时间马尔可夫链，而不是普通的固定步长模拟器。

真正的难点不只是原子级长时程 rollout 成本很高，更在于模拟预算会被大量逐微事件推进所吞掉，导致系统很难在现实预算内真正到达决定长期材料演化的关键状态。如果我们希望沿着那些真正重要的演化路径继续往前推进，同时又不丢掉时间尺度的准确性，就不能继续停留在逐微事件的 CTMC 叙事里。

### 核心思路

AtomWorld-Twins 把建模目标从微观事件回放转成关键状态演化。它不去模仿每一次 vacancy hop，而是在宏步层面同时回答三个耦合问题：

- 哪些稀疏晶格编辑在物理上真实可达
- 当前关键状态之后会走向哪个后继关键状态
- 这次宏步跳跃会消耗多少累计物理时间

这可以被形式化为一个受 reachability 约束的 Semi-Markov 视角，但更直观的理解其实更简单：用带时间语义的宏步演化替代昂贵的微观事件回放，同时不把状态变化和时间推进拆开。

### 加速与规模

预期 speedup 来自两个耦合设计。第一，latent world model 在局部 atom-vacancy 环境上学习可复用的物理规律，并把这些规律共享到大量 site，而不是每一步都重新调用 teacher 逐事件计算。第二，每个宏步预测同时推进一个物理可达的状态编辑和它累计的 CTMC 时间。teacher 监督被蒸馏进 student 之后，类似 540 亿原子、50 年 Cu-Fe 老化的大规模任务，目标是从传统 KMC 的长时间超算逐事件回放，转为单张 A100 上数小时量级的 latent 宏步推理。

### 方法概览

AtomWorld-Twins 采用 teacher-student Dreamer macro world model。

Teacher：

- teacher 直接使用原子级 KMC 模拟器本身。
- 从当前状态 X_t 出发，teacher rollout 一个 fixed-k 的微事件片段。
- teacher 提供宏步终点状态 X_t+k、累计期望时间、实际累积时间，以及从微观路径中提取的 path summary。

Student：

- student 是一个在隐空间中运行的 Dreamer 风格宏步世界模型。
- 它把当前局部 patch 与全局摘要编码成 latent state。
- 它用 posterior 和 prior 两套路径 latent 区分训练期可识别性与测试期生成。
- 它预测下一个宏步 latent state、受可达性约束的稀疏晶格编辑，以及宏步持续时间。
- 时间分支保持 `tau_exp` 为主监督，同时新增一个面向 `tau_real` 的对数正态辅助头，因此 realized waiting time 被表述为条件分布学习，而不是单点端点回归。

### 物理约束

模型围绕三条硬约束展开：

- Inventory conservation：原子和 vacancy 的数量必须保持合法。
- Local reachability：预测编辑必须落在 fixed-k 可达候选集合之内。
- Continuous-time consistency：时间监督以路径条件化的累计期望时间为主，`tau_real` 只作为辅助条件分布建模对象，而不是任意端点回归。

这也是为什么模型输出被定义为受约束的稀疏 lattice edit，而不是无限制的 dense reconstruction。

### 闭环评估

NeurIPS closed-loop 矩阵采用 autonomous macro rollout + on-policy KMC teacher probe 协议。student 从自己的预测晶格状态继续推进；teacher probe 只从该模型状态出发，用于测量局部参考时间、能量、结构距离、拓扑和约束违反率，不会覆盖模型状态。当前结果目录为 `dreamer4-main/results/neurips_closed_loop_onpolicy_matrix/`。

最强的闭环证据是 full constrained model 的物理一致性以及严格去除约束后的失败。full fixed-horizon runs 保持 reachability 和 inventory violation 为 0，并且能量与 Cu 距离漂移较小；去掉 reachability 或全部 projection 约束会立即产生非法状态。时间校准需要保守表述：full closed-loop runs 在当前 probe 协议下会高估累计时间，因此它应被写成真实的 calibration boundary，而不是已经完全解决时间校准。

### 仓库边界

> 公开仓库说明。当前公开树只保留 `kmcteacher_backend/` 里的最小 paper-facing teacher backend 子集。



### 快速开始

环境要求：

- 推荐 Python 3.10 或 3.11
- 可直接执行 `python -m pip install -r requirements.txt`
- 纯 pip 安装、smoke check 和可选依赖见 [dependencies.md](dependencies.md)

训练宏步世界模型：

```bash
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --save_dir results/atomworld_twins_v1 \
  --dataset_cache results/atomworld_twins_v1/segments.pt \
  --segment_k 4 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --realized_tau_weight 0.25 \
  --train_segments 2000 \
  --val_segments 400 \
  --max_candidate_sites 128 \
  --epochs 80 \
  --device cuda
```

训练可选的 multi-k 宏步世界模型：

```bash
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --save_dir results/dreamer_macro_edit_v30_multik_248 \
  --dataset_cache results/dreamer_macro_edit_v30_multik_248/segments.pt \
  --init_from results/kmc_teacher_dreamer_macro_wm/final_model.pt \
  --segment_ks 2 4 8 \
  --train_segments_per_k 2000 \
  --val_segments_per_k 400 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --realized_tau_weight 0.25 \
  --max_candidate_sites 192 \
  --epochs 100 \
  --device cuda
```

`--init_from` 是 weights-only warm start：它加载与 v26 兼容的权重，并在 path-summary 输入维度变化时做 resize migration，但不会继承 optimizer、epoch 或 best score。输出头默认保留；只有做明确 fresh-head 消融时才使用 `--reinit_output_heads`，或用 `--reinit_reward_heads` 只重置 reward 头而保留 edit / duration 头。v33-v52 debug 路径还新增了 `--freeze_duration_heads`、`--train_reward_heads_only`、`--train_reward_duration_heads_only`、`--train_duration_heads_only`、`--tau_log_mu_weight`、`--reward_gate_weight`、`--reward_zero_weight`、`--reward_prediction_source projected`、`--planner_selected_from`、`--planner_selected_reward_prediction_source` 与 `--planner_selected_allow_uncovered_reward_only`，用于 reward / duration、projected-reward 和 planner-selected calibration 专项消融。当前 no-op-stop long-eval 协议把 teacher 起终点 lattice 不变的宏段视为终止诊断，而不是有效监督段。在这个修正口径下，v50/v52 multi-k 在 long-horizon 时间和累计能量上都强于 fixed-k v26 baseline。v50 是 learned-duration 时间最好的 checkpoint；v52 是累计能量最好的 checkpoint，并且时间对齐仍然达标。`--duration_source baseline` 仍是 CTMC 起点速率诊断 / fallback，不作为默认 learned-duration 结论。

执行 teacher 对齐的时间评估：

```bash
cd dreamer4-main
python eval_macro_time_alignment.py \
  --checkpoint results/atomworld_twins_v1/best_model.pt \
  --cache results/atomworld_twins_v1/segments.pt \
  --split val \
  --output results/atomworld_twins_v1/eval_time_alignment.json \
  --save_all_samples
```

执行 multi-k planning 与在线 traditional KMC 对照评估：

```bash
cd dreamer4-main
python eval_macro_long_trajectory.py \
  --checkpoint results/dreamer_macro_edit_v30_multik_248/final_model.pt \
  --rollout_segments 500 \
  --max_episode_steps_override 5000 \
  --output results/dreamer_macro_edit_v30_multik_248/eval_long_trajectory_500.json \
  --device cpu
```

做 duration 诊断时，long evaluator 还支持 `--duration_source baseline` 与 `--duration_source blend --duration_blend_alpha <alpha>`。blend 模式在 log tau 空间混合 CTMC start-state baseline 与 learned duration head；除非实验记录明确升格，否则它只作为校准诊断口径。long eval 默认会在第一个 teacher no-op 宏段停止；只有需要复现旧诊断时才传入 `--allow_teacher_noop_segments` 保留这些不变段。



## License

This repository is released under the MIT License. See LICENSE for details.
