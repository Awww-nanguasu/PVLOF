# PV Anomaly Detection

光伏电站异常检测与智能运维项目。项目以 Elasticsearch 和本地 Parquet 为数据来源，包含数据质量审计、功率预测、功率残差监控和组串低电流检测。

当前主要算法链路为：

```text
Elasticsearch / Parquet
        |
        +--> Transformer：预测下一时刻正常有功功率
        |         |
        |         +--> EWMA：监控功率预测残差，检测逆变器级整体欠发
        |
        +--> PVLOF：比较组串电流，定位非零低电流组串
                  |
                  +--> 零电流独立规则
                  +--> 虚拟辐照度与 forecast_ghi 条件化
                  +--> 孤立欠流与群体欠流检测
                  +--> 连续确认、恢复与事件重建
```

## 当前状态

- 数据主时间粒度：5分钟。
- 功率目标字段：`active_power`。
- 功率预测：多站点 Transformer 初版已经实现并完成离线实验。
- 功率异常：Transformer残差与EWMA已经实现，仍需进一步验证标签语义。
- 组串异常：当前最新实现为 **PVLOF V1.6 Hybrid Gate**。
- PVLOF V1.7目前只有改进规划，尚未实现或验证。
- 告警写回Elasticsearch/Kibana尚未完成，当前以离线回放和人工复核为主。

PVLOF V1.6孤立欠流核心条件为：

```text
LOF异常
AND
(
    相对欠流 >= 20%
    OR
    绝对欠流 >= 0.5 A
)
```

首次确认通常要求连续3个5分钟异常点，确认后使用记忆与恢复状态机。群体欠流、虚拟辐照度、天气条件化和完整数学定义见 [`PVLOF.md`](PVLOF.md)。

> 注意：V1.6源码已经在仓库中，但最终标定JSON、部分点级结果和服务器实验产物可能需要从训练服务器补充下载。源码存在不等于最终运行环境已经完整复现。

## 电站与数据

当前主要使用以下电站：

| 环境 | plant_id | 说明 |
|---|---:|---|
| 测试环境 | 33 | 对应生产环境234，同一批物理设备 |
| 生产环境 | 234 | 24台逆变器 |
| 生产环境 | 791 | 14台逆变器 |
| 生产环境 | 892 | 13台逆变器 |

主要数据目录：

```text
data/
├─ raw/
│  ├─ device/
│  │  ├─ plant_id=33/
│  │  ├─ plant_id=234/
│  │  ├─ plant_id=791/
│  │  └─ plant_id=892/
│  └─ weather_15min/
├─ interim/
├─ processed/
└─ validation/
```

设备数据通常为5分钟粒度，天气预报数据通常为15分钟粒度。统一业务时区为：

```text
Asia/Shanghai
```

不要固定假设每台逆变器都有30条有效组串。`main_string_count`表示数量，不一定等于最大组串编号；实际通道应结合历史电流和组串状态推断。

## 项目结构

```text
.
├─ configs/                 # 数据、训练和算法配置
├─ data/                    # 原始、中间、处理和验证数据
├─ artifacts/
│  ├─ models/               # 标定参数、checkpoint和点级算法输出
│  └─ reports/              # 审计报告、对比结果和人工复核表
├─ scripts/                 # 命令行入口
├─ src/pv_anomaly/          # Python包源码
├─ tests/                   # 单元测试
├─ PROJECT_CONTEXT.md       # 早期项目背景和总体目标
├─ DECISIONS.md             # 已记录的技术决策
├─ PVLOF.md                 # PVLOF V1.6详细技术说明
├─ DEVICE_MIGRATION_TODO.md # 换电脑迁移清单
├─ pyproject.toml
└─ uv.lock
```

核心源码：

| 文件 | 用途 |
|---|---|
| `src/pv_anomaly/pvlof.py` | PVLOF V1基础实现 |
| `src/pv_anomaly/pvlof_channels.py` | 实际组串通道识别 |
| `src/pv_anomaly/pvlof_v2.py` | 虚拟辐照度、期望电流、LOF和群体欠流 |
| `src/pv_anomaly/pvlof_v12.py` | `forecast_ghi`条件化 |
| `src/pv_anomaly/pvlof_v2_hier.py` | 分层LOF阈值和严格连续确认 |
| `src/pv_anomaly/pvlof_v15.py` | 确认后记忆与恢复机制 |
| `src/pv_anomaly/pvlof_v16.py` | V1.6记忆配置与输出封装 |
| `src/pv_anomaly/pvlof_events.py` | 候选证据和事件重建 |
| `src/pv_anomaly/models/` | Transformer模型、数据集、训练和评估 |
| `src/pv_anomaly/ewma.py` | EWMA标定和检测 |

## 环境要求

- Windows或Linux
- Python `>=3.11,<3.12`
- [uv](https://docs.astral.sh/uv/)
- GPU训练时需要兼容的NVIDIA驱动和PyTorch；本地数据审计与大部分测试不需要GPU

### 安装uv（Windows PowerShell）

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装后重新打开PowerShell：

```powershell
uv --version
```

### 创建基础环境

在项目根目录运行：

```powershell
uv sync --frozen
```

安装测试和代码检查依赖：

```powershell
uv sync --frozen --extra dev
```

GPU训练服务器安装训练依赖：

```bash
uv sync --frozen --extra train --extra dev
```

不要复制其他电脑生成的 `.venv`。uv会依据 `pyproject.toml` 和 `uv.lock` 在当前电脑重新创建环境。

验证包导入：

```powershell
uv run python -c "import pv_anomaly; print(pv_anomaly.__file__)"
```

## 环境变量

复制示例文件：

```powershell
Copy-Item .env.example .env
```

填写：

```dotenv
ES_URL=<Elasticsearch地址>
ES_USERNAME=<只读账号>
ES_PASSWORD=<密码>
ES_INDEX=<确认后的索引或别名>
ES_VERIFY_CERTS=true
ES_CA_CERT=
ES_TIMEOUT_SECONDS=15
```

`.env`包含敏感信息，不应提交到Git或发送给无关人员。生产环境审计优先使用只读、最小权限账号。

检查ES连接：

```powershell
uv run python scripts/check_es_connection.py
```

## 常用命令

### 运行测试

```powershell
uv run pytest
```

只运行PVLOF V1.6测试：

```powershell
uv run pytest tests/test_pvlof_v16.py
```

### 审计一个生产电站的Parquet

```powershell
uv run python scripts/audit_production_parquet.py `
  --root data/raw/device/plant_id=234 `
  --plant-id 234 `
  --output artifacts/reports/production_parquet_234.json
```

### 构建生产多站点训练数据

```powershell
uv run python scripts/build_training_data.py `
  --config configs/training_production_global_v2.yaml
```

### 训练生产全局Transformer

该步骤应在GPU服务器执行：

```bash
uv run python scripts/train_transformer.py \
  --config configs/transformer_production_global_current.yaml
```

### PVLOF V1.6

V1.6不是单一的独立运行脚本。当前流水线由以下阶段组合：

1. 读取并清洗设备电流；
2. 使用V1.2/V2和分层标定产生点级评分；
3. 使用 `derive_pvlof_v16.py` 派生20%/0.5A OR门控和记忆配置；
4. 使用 `apply_pvlof_v16_memory.py` 应用确认后记忆并重建事件；
5. 使用 `export_pvlof_v16_baseline_review.py` 生成人工对比表。

查看各脚本参数：

```powershell
uv run python scripts/derive_pvlof_v16.py --help
uv run python scripts/apply_pvlof_v16_memory.py --help
uv run python scripts/export_pvlof_v16_baseline_review.py --help
```

运行前必须确认源码版本、输入Parquet、基础标定JSON、分层标定JSON和输出目录相互匹配。不要随意用旧版标定文件替代缺失的新版本标定文件。

## 本地与服务器分工

本地电脑适合：

- 阅读和修改代码；
- Elasticsearch只读审计与数据导出；
- Parquet/CSV质量检查；
- 人工复核；
- 运行不依赖GPU的测试。

GPU服务器适合：

- Transformer训练与批量推理；
- 大规模PVLOF回放；
- 大Parquet的批量评分；
- 保存checkpoint、标定JSON和运行报告。

服务器常用项目目录：

```text
/root/pv-anomaly
```

服务器生成的重要结果应下载回本地归档。不要把服务器虚拟环境、Docker/Kubernetes系统目录或系统缓存复制进项目仓库。

## 数据与版本安全规则

1. 原始数据默认只读，不直接覆盖。
2. 新实验使用新的版本名、结果列和输出目录。
3. 已冻结版本不覆盖；通过并行输出进行比较。
4. 训练、验证和测试按时间划分，避免未来数据泄漏。
5. 预测时刻与目标时刻必须严格对齐。
6. 测试站点33和生产站点234虽然对应同一批物理设备，但环境中的 `plant_id` 不同。
7. 零电流和非零低电流分开评估。
8. 设备级告警标签不能直接当成组串级真值。
9. 没有负样本时不能报告Precision或F1，只能报告Recall和覆盖率。
10. 在人工验收前，不把离线结果直接写回生产ES。

## 当前待补充产物

迁移或新环境中应检查以下路径是否存在：

```text
artifacts/models/pvlof_v12_3point/
artifacts/models/pvlof_v16/
artifacts/models/pvlof/alarm_windows_v2_floor_raise/v16/
data/validation/test_plant_33_aug/
```

这些内容可能只存在于GPU服务器，需要单独下载。缺少它们不影响阅读源码，但会影响V1.6最终实验的完整复现。

## 进一步阅读

- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)：项目最初目标、平台和数据背景。
- [`DECISIONS.md`](DECISIONS.md)：已记录的技术边界和决策。
- [`PVLOF.md`](PVLOF.md)：PVLOF V1.6完整数学原理、标定、预警和已知盲区。
- [`DEVICE_MIGRATION_TODO.md`](DEVICE_MIGRATION_TODO.md)：换电脑迁移与恢复检查清单。
- `OLD_CODEX_CHAT_HANDOFF.md`：旧会话生成的交接摘要；存在时作为历史索引，不替代源码和正式报告。

## 开发协作约定

- 修改前先检查当前源码、配置和输出版本。
- 涉及运行脚本时，先明确命令应在本地PowerShell还是服务器Bash执行。
- 代码修改后先运行对应单元测试，再进行小范围数据回放。
- 人工确认结果后再更新 `DECISIONS.md` 和 `PVLOF.md`。
- 不根据聊天记忆猜测最终参数，以冻结的JSON标定文件和可复现报告为准。
