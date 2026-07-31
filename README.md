# 光伏异常检测项目

当前阶段聚焦于公司 Elasticsearch（ES）的安全只读接入和数据可行性审计。目标算法链为：Transformer 预测正常功率，EWMA 监控预测残差，PVLOF/LOF 定位异常组串。

## 环境初始化

```powershell
# 安装基础依赖和开发工具
uv sync --extra dev --locked
Copy-Item .env.example .env
```

编辑本地 `.env`，填入已轮换的只读账号。真实密码不得写入代码、Notebook、日志或 Git。

需要训练模型时安装训练依赖：

```powershell
uv sync --extra train --extra dev --locked
```

`uv.lock` 锁定了可复现的 Python 依赖。Linux x86_64 服务器会按锁文件安装带 CUDA 13
运行时的 PyTorch 2.12；Windows 开发机没有 NVIDIA CUDA 环境时安装 CPU 构建是正常现象。

## 第一阶段命令

```powershell
# 验证连接并查看 ES 版本
python scripts/check_es_connection.py

# 列出账号可访问的 index、alias 和 data stream
python scripts/inspect_indices.py

# 查看目标 index mapping（默认使用 ES_INDEX）
python scripts/inspect_mapping.py

# 导出少量样例到本地；不会修改 ES
python scripts/export_sample.py --size 100 --output data/raw/es_sample.json

# 审计 JSON、JSONL 或 CSV 样例
python scripts/audit_data.py data/raw/es_sample.json --output artifacts/reports/data_audit.json

# 按上海本地日期导出设备和天气数据；--end 为不包含的结束日期
python scripts/export_training_data.py --start 2026-07-15 --end 2026-07-22

# 验证 Parquet 分区、行数、主键和设备—天气电站覆盖
python scripts/audit_parquet_export.py

# 审计训练字段分布并生成报告
python scripts/profile_training_fields.py

# 生成全量对齐数据和按时间切分的候选正常功率预测数据
python scripts/build_training_data.py --config configs/training.yaml

# 独立验证训练、验证、测试输出
python scripts/audit_training_outputs.py
```

所有 ES 脚本只允许 `GET/HEAD` 请求。未获得业务批准和最小写权限前，不提供创建、更新或删除远程对象的功能。

## 配置

- `.env`：本机连接信息和凭据，不纳入版本控制。
- `configs/data.example.yaml`：候选字段别名、时区和组串电流命名规则。
- `configs/export.yaml`：Parquet 导出的 index、字段、排序键和输出目录。
- `configs/training.yaml`：候选正常质量规则、时间切分边界和加工输出目录。
- `pyproject.toml`：Python 版本、依赖、测试和代码检查配置。

ES 服务端已确认是 7.10.1。第一阶段继续使用 Python 标准库调用只读 REST API；后续若引入官方 `elasticsearch` 客户端，应固定为兼容的 7.x 版本。

Parquet 导出采用只读 `search_after` 分页，单次限制在 31 天内。输出目录已经存在 Parquet 文件时会停止，避免重复运行造成数据重复；导出结束还会核对 ES 精确计数与本地行数。

## 功率预测

默认任务使用连续 24 个 5 分钟点预测下一时刻（`t+5min`）有功功率。缩放器和设备词表只在训练集上拟合；默认特征暂不使用缺失的实测 GHI，也不使用语义尚未确认的天气预报字段。

```bash
# 服务器：验证 PyTorch 能调用 RTX 5090 D，并实际执行一次矩阵乘法
uv run python scripts/verify_torch_gpu.py

# 持久性基线：用当前功率预测 5 分钟后的功率
uv run python scripts/evaluate_baseline.py

# 训练 Transformer；自动选择 CUDA、启用混合精度和验证集早停
uv run python scripts/train_transformer.py --config configs/transformer.yaml

# 独立评估最佳检查点
uv run python scripts/evaluate_transformer.py \
  --checkpoint artifacts/models/transformer/best.pt \
  --data data/processed/transformer/test.parquet

# 导出验证集和测试集的逐点预测、同窗口基线与残差
uv run python scripts/export_transformer_residuals.py

# 按上海日期、设备和实际功率比例分箱审计 MAE/RMSE 来源
uv run python scripts/audit_transformer_residuals.py

# 生成 forecast_ghi 当前时刻与未来15分钟解释的两套同样本数据
uv run python scripts/build_forecast_variants.py

# 两套配置保持其他参数完全一致
uv run python scripts/train_transformer.py --config configs/transformer_forecast_current.yaml
uv run python scripts/train_transformer.py --config configs/transformer_forecast_plus15.yaml

# 残差式模型：以当前功率为基线，仅学习 t+5min 功率修正量
uv run python scripts/train_transformer.py --config configs/transformer_residual_plus15.yaml

# 低学习率，并用 SmoothL1 + 0.5*MSE 加强大误差惩罚
uv run python scripts/train_transformer.py \
  --config configs/transformer_residual_plus15_composite.yaml

# 功率感知损失：比例 SmoothL1 + 额定功率加权 MSE
uv run python scripts/train_transformer.py \
  --config configs/transformer_residual_plus15_power_aware.yaml

# 构建同站留一中位功率、设备相对偏差和同站数据可用性特征
uv run python scripts/build_peer_features.py

# 同站特征消融实验：其余设置与低学习率残差模型保持一致
uv run python scripts/train_transformer.py \
  --config configs/transformer_residual_plus15_peer.yaml

# 导出同站Transformer、同站物理基线和持久性基线的逐点预测残差
uv run python scripts/export_transformer_residuals.py \
  --config configs/transformer_residual_plus15_peer.yaml \
  --checkpoint artifacts/models/transformer_residual_plus15_peer/best.pt \
  --output-directory artifacts/models/transformer_residual_plus15_peer/residuals

# 按日期、设备和功率区间比较三种预测
uv run python scripts/audit_transformer_residuals.py \
  --input-directory artifacts/models/transformer_residual_plus15_peer/residuals \
  --output-directory artifacts/reports/transformer_residual_plus15_peer
```

基线指标保存在 `artifacts/models/baseline/metrics.json`。Transformer 保存最佳检查点、训练历史和测试指标到 `artifacts/models/transformer/`。

## 测试

```powershell
uv run pytest
uv run ruff check .
```

## PVLOF 第一版

PVLOF 在同一逆变器、同一时刻内横向比较非零组串电流。组串电流先除以当时的
组串中位电流，再计算一维 Local Outlier Factor。只有相对电流偏低、LOF 超过标定
阈值并连续出现的点才产生 `pvlof_alert`。白天的零电流不参与 LOF，直接产生
`zero_current_alert`。

```bash
# 用训练时段候选正常数据标定 LOF 分数阈值
uv run python scripts/calibrate_pvlof.py

# 验证集：结束时间不包含 2026-07-01
uv run python scripts/run_pvlof.py \
  --start 2026-06-01 \
  --end 2026-07-01 \
  --output artifacts/models/pvlof/validation.parquet \
  --events artifacts/models/pvlof/validation_events.parquet \
  --report artifacts/reports/pvlof_validation.json

# 测试集：结束时间不包含 2026-07-23
uv run python scripts/run_pvlof.py \
  --start 2026-07-01 \
  --end 2026-07-23 \
  --output artifacts/models/pvlof/test.parquet \
  --events artifacts/models/pvlof/test_events.parquet \
  --report artifacts/reports/pvlof_test.json
```

输出中的 `string_status` 只作为弱标签，不参与测试时的 PVLOF 打分。状态 2 用于
暂时评估非零低电流，状态 4 用于检查零电流规则；状态 3 是离线，不作为电流异常
标签。这些指标不能替代人工确认的故障标签。
