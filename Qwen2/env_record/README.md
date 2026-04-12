# env_record 说明

本目录保存两条模型线的环境快照：

- `env_record/qwen2_vl/`
- `env_record/internvl/`

每个子目录包含：

- `python_version.txt`
- `core_versions.txt`
- `gpu_info.txt`
- `pip_freeze.txt`

建议把这些文件视为“原实验环境记录”，而不是开源后的最小安装清单。快速复现优先看上层 `requirements.txt`。
