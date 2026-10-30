# DIDN+AMIR

单通道医学图像恢复：保留 DIDN 的六个独立、256 通道 DUB、共享 Recon 和全局残差，加入 AMIR 的 RIN、三个 SRM 和四个 CRM。支持 PET、CT、MRI 混合训练、续训和独立测试。

RIN 使用 `16×256` 字典生成每张图的 256 维指令；A/C/E 后接 SRM，B/D/F 后接 CRM，六路重建特征融合后再接 CRM。完整训练效果需通过正式实验验证。

## 文件与环境

```text
train.py / test.py        训练、测试入口
model.py                 DIDN+AMIR 网络
dataset.py               配对、归一化和裁剪
validation.py            整图推理、指标与 NIfTI 导出
option.py                命令行参数
checkpoints.py           权重与训练状态管理
utils.py                 日志、运行记录和实验总表
tests/                   训练记录与续训等行为回归测试
instruction/             AMIR 来源与许可证
data/All-in-One/          医学数据
checkpoint/didn_amir/     默认融合权重与训练记录
results/                 独立测试结果
output/                  科研报告、结构图和实验总表
venv/                    本机 Python 环境
tmp/                     D 盘临时文件与缓存
```

以上八个 Python 文件构成训练与测试的核心源码，迁移时建议完整保留。使用 Python 3.10+、PyTorch、NumPy、SimpleITK；新环境安装及跨机迁移见 [MIGRATION.md](MIGRATION.md)。`requirements.txt` 规定兼容范围，不是精确版本锁定文件。

## Git 管理范围

原本地项目在 `DIDN+AMIR` 分支维护；公开仓库为 [sunze051114/DIDN-AMIR](https://github.com/sunze051114/DIDN-AMIR)，默认分支为 `main`，发布提交只包含下列 15 个文件。
[.gitignore](.gitignore) 控制后续新增文件的忽略范围；本地开发历史中已有的文件不会因修改忽略规则自动移除。

| 内容 | 发布范围 |
| --- | --- |
| `train.py`、`test.py`、`model.py`、`dataset.py`、`validation.py`、`option.py`、`checkpoints.py`、`utils.py` | 八个核心运行文件 |
| `requirements.txt`、`README.md`、`MIGRATION.md`、`.gitignore` | 依赖、使用和迁移说明及忽略规则 |
| `instruction/AMIR_LICENSE`、`instruction/AMIR_NOTICE.md` | 许可证与来源说明 |
| `tests/test_training_logging.py` | 训练记录、续训和评测行为回归测试 |
| `data/`、`checkpoint/`、`results/`、`output/`、`recovery/` | 数据、权重、实验记录与报告留在本地，不属于发布清单 |
| `venv/`、`.venv/`、`tmp/`、缓存、`AGENTS.md`、编辑器和本机配置 | 本地环境和协作文件，不属于发布清单 |

运行时自动创建权重和结果目录，无需 `.gitkeep`。`output/experiments.csv` 和其他实验产物保存在本地。
克隆发布分支后另行准备医学数据；续训时还需复制对应实验目录，见迁移指南。

## 数据

默认路径为 `data/All-in-One/<PET|CT|MRI>/<train|validation|test>/{LQ,HQ}/`。训练使用可信的 pickle 数组 `.bin`，验证和测试使用 `.nii` 或 `.nii.gz`；LQ/HQ 必须同名、同尺寸。

保留原始患者划分，使用已有退化配对，不额外生成高斯噪声；训练对 LQ/HQ 做同位置 crop。

| 模态 | 训练归一化前的范围 | 指标评估范围 |
| --- | --- | --- |
| PET | [0, 20] | [0, 20] |
| CT | [-1024, 3071] HU | [-160, 240] HU |
| MRI | [0, 4095] | [0, 4095] |

训练裁剪并映射至 `[0,1]`。旧自然图数据和 `checkpoint/pretrained_*` 不用于默认医学融合流程。

## 训练与续训

以下 PowerShell 命令从项目根目录执行；先按迁移指南设置 D 盘临时目录与缓存。

```powershell
Set-Location 'D:\projects\DIDN\DIDN'
.\venv\Scripts\python.exe -B train.py --check_data
.\venv\Scripts\python.exe -B train.py
.\venv\Scripts\python.exe -B train.py --resume auto

# 单模态新实验
.\venv\Scripts\python.exe -B train.py --modalities CT --save_dir checkpoint/ct
```

`--check_data` 检查所选模态的所有划分文件名配对，并读取首、中、末样本，不启动训练，也不是逐文件内容校验。

默认 batch=1、patch=128、FP32、seed=42，Adam 学习率 `2e-4`，余弦退火至 `1e-6`，共 200,000 次成功优化器更新。损失为 `L1 + 0.001 × 三处 SRM 均衡损失之和`，按切片数量混合模态。`--amp`、`--tf32`、`--grad_clip` 可选；Windows 默认 `--threads 0`。

每 1,000 次更新验证，每模态按文件名推断的患者标识分层抽取 32 张；`--val_samples 0` 使用全量验证集。输入尺寸至少 16 且为 8 的倍数，验证和测试自动补边、裁回原尺寸。

默认保存至 `checkpoint/didn_amir/`，或用 `--save_dir` 指定独立实验目录：

| 文件 | 用途 |
| --- | --- |
| `best.pth` | 按各模态平均 PSNR 的等权平均择优，用于测试 |
| `last.pth` | 完整训练状态：模型、优化器、调度器、scaler、随机状态与下一批位置 |
| `config.json`、`run.json` | 训练配置；实验身份、环境、来源、状态、事件和资源统计 |
| `training.csv`、`console.log` | 训练损失、学习率、梯度与 AMP 统计；控制台输出 |
| `validation.csv`、`validation_samples.json` | 验证指标与实际样本清单 |

`training.csv` 记录首次、每 100 次及最后一次成功更新的当前 batch 数值，并非区间平均。日志随同目录续训追加；跨目录恢复复制已有训练日志，并按恢复进度整理训练 CSV。

续训保持原参数、数据绝对路径与验证样本一致，保留整套实验文件。`--resume auto` 加载本目录 `last.pth`；跨目录恢复仅允许新目录或同一实验的可识别中断迁移。`--init_model PATH` 仅加载结构匹配权重，作为新实验重新优化。完整状态需数 GiB 空间。

`run.json` 保存各次启动的实际环境、来源权重 SHA256、运行状态、验证事件与资源统计。`output/experiments.csv` 汇总跨实验配置、成绩、成本和结果路径，并保留人工填写的名称、目的、对照与备注，也可启动时指定：

```powershell
.\venv\Scripts\python.exe -B train.py --experiment_name E001 --purpose "检查平衡损失" --comparison E000 --notes "首次实验" --save_dir checkpoint/E001
```

这些说明参数不参与续训配置匹配。跨设备或软件版本不保证逐位一致；实验总表暂时无法写入时会提示，原始运行记录仍单独保存。

## 独立测试

```powershell
.\venv\Scripts\python.exe -B test.py
.\venv\Scripts\python.exe -B test.py --max_images 1 --no_save_images
.\venv\Scripts\python.exe -B test.py --model checkpoint/didn_amir/last.pth
.\venv\Scripts\python.exe -B test.py --save_dir checkpoint/ct --modalities CT
```

默认加载 `<save_dir>/best.pth`，评估完整 `test` 划分；需先训练或提供结构匹配的融合权重。`--ensemble none/self/both` 控制自集成，`--split validation` 改为验证集。

验证集测试优先读取检查点旁的样本清单并按所选模态过滤；清单缺失或覆盖不足时重新派生，不能保证复现训练分数。`--max_images` 限制每模态数量，0 表示不额外截断已有清单或全量数据。

反归一化后计算 PSNR/SSIM/RMSE，PSNR 的 `data_range` 为每张 HQ 的 `max-min`，常量 HQ 会被拒绝；SSIM 使用 11×11 高斯窗口与零填充。CT 指标使用评估窗，导出使用完整训练 HU 范围。

每次测试创建 `results/<时间戳-唯一标识>/`，保存 `medical_metrics.csv`、`medical_summary.json`，默认另导出保留空间信息的 `.nii.gz`。报告记录权重哈希、实际样本、环境、指标协议和评测状态；`--no_save_images` 关闭图像导出，`--result_dir` 可指定 D 盘目录。

总表仅将 `test` 划分写入测试列，两种推理模式同时运行时以普通推理作为总表成绩。比较实验时核对关联 JSON 的实际模态和样本范围；某个模式失败时仍保留已完成模式的结果。

## 已完成的流程验证

2026-10-30 在 RTX 4060 Laptop GPU 上使用真实 DIDN+AMIR 模型完成小规模验证：PET、CT、MRI 各取一对训练、验证和测试裁剪，batch=1、训练 patch=16、FP32，共完成 3 次成功参数更新。

训练与验证、`best.pth`/`last.pth` 保存、第 1 步中断后的新进程续训、普通与八路自集成测试、6 个 NIfTI 导出及空间信息核对均通过，所检查的损失与指标均为有限数。全部 demo 临时产物已清理，原有代码、数据、权重和结果保持不变。

这是小输入的流程连通性验证；正式训练效果与完整输入的资源需求仍需通过实际实验确认。

## 常用检查

- 缺少 `best.pth`：先训练，或用 `--model` 指定结构匹配的融合权重。
- 续训被拒：核对原配置、数据路径、验证清单与实验身份；新实验使用新的 `--save_dir`。
- 显存不足：在新实验目录尝试较小 patch 或 `--amp`；本模型约 1.69 亿参数，实际需求需在目标设备确认。
- 全部选项：`train.py --help`、`test.py --help`。

## 实现来源

[DIDN](https://github.com/SonghyunYu/DIDN) · [AMIR](https://github.com/Yaziwel/All-In-One-Medical-Image-Restoration-via-Task-Adaptive-Routing)。来源、改动和引用见 [instruction/AMIR_NOTICE.md](instruction/AMIR_NOTICE.md)，许可证见 [instruction/AMIR_LICENSE](instruction/AMIR_LICENSE)。
