# DIDN+AMIR 迁移指南

适用于 Windows + NVIDIA 显卡，目标目录为 `D:\projects\DIDN\DIDN`。网络、训练参数、指标和实验记录见 [README.md](README.md)。

## 1. 确认源码并备份实验

在原电脑检查实际交付的分支、提交与未提交修改：
```powershell
Set-Location 'D:\projects\DIDN\DIDN'
git branch --show-current
git status --short
git log -1 --oneline
```

原本地项目使用 `DIDN+AMIR`，公开仓库克隆后的默认分支为 `main`；保留未提交修改，不修改或回退原项目的 `master`。复制实验前等待权重保存完成并停止训练，保留原电脑文件直到目标电脑核验完成。

| 内容 | 迁移要求 |
| --- | --- |
| 八个 Python 文件、`requirements.txt`、README/MIGRATION、`.gitignore`、`tests/test_training_logging.py` | 保留当前实际版本；与下行两个归属文件组成发布分支的 15 个文件 |
| `instruction/AMIR_LICENSE`、`instruction/AMIR_NOTICE.md` | 保留第三方许可证和来源说明 |
| `data/All-in-One/` | 复制所需模态与划分的实际数据，保留原患者划分和同名 LQ/HQ 配对 |
| `checkpoint/<实验>/` 整个目录 | 保留 `last.pth`、`best.pth`、`config.json`、`run.json`、`validation_samples.json`、`validation.csv`、`training.csv`、`console.log` 等已有文件 |
| `output/experiments.csv` | 保留跨实验成绩、成本与人工备注；它不随单个权重目录自动迁移 |
| `results/`、`output/` 中需留档的报告和图 | 一起备份已有指标、预测图及科研报告 |
| `venv/`、`.venv/`、`tmp/`、`__pycache__/`、`.cache/` | 在新电脑重建环境与缓存，无需复制 |

GitHub 公开仓库为 `sunze051114/DIDN-AMIR`，默认分支为 `main`；原本地 `AGENTS.md` 与历史开发记录不属于该发布清单。
Git 克隆不会带回数据、训练权重、测试结果或未提交修改；按实际文件另行复制，不能只核对仓库名称。

## 2. 复制并核对数据

本机数据使用 NTFS 硬链接共享已有 AMIR 文件。跨电脑应复制文件的实际内容，按数据逻辑大小预留空间，目标电脑无需原 AMIR 目录。
保持 `data/All-in-One/<PET|CT|MRI>/<train|validation|test>/{LQ,HQ}/`；训练为可信的 `.bin`，验证/测试为 `.nii` 或 `.nii.gz`。
从头训练需要 train+validation；独立测试需要所选划分；完整续训使用原训练及验证数据。不要重新混合患者或划分。

重要权重可在两台电脑分别核对 SHA256；大批数据使用复制工具的内容校验。训练记录中的路径签名不能替代内容校验：
```powershell
Get-FileHash -LiteralPath 'D:\projects\DIDN\DIDN\checkpoint\didn_amir\last.pth' -Algorithm SHA256
```

## 3. 在 D 盘重建环境

准备 64 位 Python 3.10 和支持目标显卡的 NVIDIA 驱动。以下设置仅作用于当前 PowerShell，会话重开后需重新设置：
```powershell
Set-Location 'D:\projects\DIDN\DIDN'
nvidia-smi
New-Item -ItemType Directory -Force 'D:\projects\DIDN\DIDN\tmp' | Out-Null
$env:TEMP = 'D:\projects\DIDN\DIDN\tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PIP_CACHE_DIR = "$env:TEMP\pip-cache"
$env:TORCH_HOME = "$env:TEMP\cache\torch"
$env:XDG_CACHE_HOME = "$env:TEMP\cache"
$env:CUDA_CACHE_PATH = "$env:TEMP\cache\cuda"
$env:PYTHONDONTWRITEBYTECODE = '1'
py -3.10 -m venv venv
```

**下面两套 PyTorch 安装命令只选一套。**

A. 复用本机 Python 3.10.11、RTX 4060 Laptop 8 GiB 已有环境的版本组合：
```powershell
.\venv\Scripts\python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

B. RTX 50 系列等 Blackwell 显卡使用支持该架构的构建；以下方案尚未在本项目目标 RTX 50 系列电脑实测：
```powershell
.\venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

两套方案随后都安装其余依赖并检查：
```powershell
.\venv\Scripts\python.exe -m pip install numpy==2.2.6 SimpleITK==2.5.6
.\venv\Scripts\python.exe -m pip check
```

仅需 PyTorch、NumPy、SimpleITK。`requirements.txt` 是兼容范围；可用 `pip freeze` 将实际版本记录到 D 盘项目目录。
安装来源：[官方历史版本命令](https://pytorch.org/get-started/previous-versions/)、[PyTorch 2.7 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。

## 4. 验证环境与配对

确认 CUDA 实际运算、参数入口及数据检查：
```powershell
.\venv\Scripts\python.exe -B -c "import torch,numpy,SimpleITK; print(torch.__version__,torch.version.cuda,numpy.__version__,SimpleITK.Version_VersionString()); assert torch.cuda.is_available(),'CUDA unavailable'; print(torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print((x @ x).mean().item()); torch.cuda.synchronize()"
.\venv\Scripts\python.exe -B train.py --help
.\venv\Scripts\python.exe -B test.py --help
.\venv\Scripts\python.exe -B train.py --check_data
```

`--check_data` 要求所选模态三个划分齐全；检查全部文件名配对及每模态首/中/末样本，不会训练，也不是逐文件内容校验。
仅迁移部分划分时跳过全划分检查；只迁移单模态时加 `--modalities CT` 等选项。GPU 运算成功不代表完整训练显存或收敛已验证。

## 5. 续训与小样本测试

完整续训保持原参数及数据绝对路径，复制整个实验目录后执行：
```powershell
.\venv\Scripts\python.exe -B train.py --resume auto --save_dir 'D:\projects\DIDN\DIDN\checkpoint\didn_amir'
.\venv\Scripts\python.exe -B test.py --max_images 1 --no_save_images --result_dir 'D:\projects\DIDN\DIDN\results'
```

上面的续训命令只适用于原实验使用默认参数；否则须补齐原参数。程序核对模态顺序、训练/验证设置、数值选项、数据路径与路径签名，增加总迭代次数也会被视为配置改变。
`--resume auto` 使用 last.pth 恢复模型、Adam、调度器、scaler、随机状态和下一批位置；best.pth 用于择优测试，不能完整续训。默认测试读取 `checkpoint/didn_amir/best.pth`，其他目录用 `--model` 指定。
跨目录恢复须使用空目录或可识别为同一实验的中断迁移目录；保留 run/config/验证名单与日志，不能混入其他实验。迁移会复制已有日志；恢复会对账验证事件，并清理目标训练日志中超出快照进度的行。
`run.json` 保存实验身份、协议、来源、各次启动/续训环境及资源记录；换机环境元数据不等于训练配置变化。`output/experiments.csv` 和需保留的报告仍须单独备份。
若路径或协议需要改变，使用 `--init_model <匹配权重>` 和新的 `--save_dir` 开始新实验；GPU 编号、线程数和输出目录可调整。
CPU 上 AMP 实际关闭；换设备可能重新建立 scaler 或部分随机状态。不同显卡/软件版本不保证逐位一致。其余训练与测试选项见 [README.md](README.md)。
