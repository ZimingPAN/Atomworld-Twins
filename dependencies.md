# Dependencies And Setup

This note is for the public repository.

Covered by the pip install path in this document:

- `dreamer4-main/train_dreamer_macro_edit.py`
- `dreamer4-main/eval_macro_time_alignment.py`
- `dreamer4-main/eval_macro_long_trajectory.py`
- `dreamer4-main/tests/test_kmc_macro_edit.py`
- plotting scripts under `fig/`

Not covered by this document:

- local private history under `RLKMC-MASSIVE-main/`
- hidden / ignored baseline snapshots such as `LightZero-main/`

## English

### Recommended Python

Use Python 3.10 or 3.11 for the public reproduction path. The local legacy Python 3.9 environment is a maintainer compatibility path, not the default public setup.

### One-command package install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your machine needs a custom PyTorch wheel, install `torch` and `torchvision` first for your platform, then run the same `requirements.txt` command.

### Minimal smoke checks

```bash
cd dreamer4-main
python -c "import train_dreamer_macro_edit; import dreamer4.macro_edit; print('dreamer_import_ok')"
python -m pytest tests/test_kmc_macro_edit.py -q
```

### Optional package

If you specifically need `kmcteacher_backend/RL4KMC/envs/kmc_env.py`, also install:

```bash
python -m pip install gym==0.26.2
```

## 中文

### 推荐 Python 版本

面向公开复现时，建议直接使用 Python 3.10 或 3.11。任务记录里提到的本地旧 Python 3.9 环境属于维护者兼容路径，不是默认公开方案。

### 一条命令安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果你的机器需要单独选择 PyTorch 轮子，可以先按本机平台安装 `torch` 和 `torchvision`，再执行同一个 `requirements.txt` 命令补齐剩余依赖。

### 最小 smoke check

```bash
cd dreamer4-main
python -c "import train_dreamer_macro_edit; import dreamer4.macro_edit; print('dreamer_import_ok')"
python -m pytest tests/test_kmc_macro_edit.py -q
```

### 可选依赖

如果你还需要 `kmcteacher_backend/RL4KMC/envs/kmc_env.py` 这个 gym 包装器，再额外执行：

```bash
python -m pip install gym==0.26.2
```