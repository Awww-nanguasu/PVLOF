# PV 异常检测项目：换电脑迁移与继续工作 To-do List

> 用途：从当前电脑迁移到另一台电脑，并让新电脑上的 Codex 在现有代码基础上继续工作。
>
> 原则：项目目录是工作成果，Markdown 文档是长期上下文，`.codex` 会话归档只用于追溯。

## 0. 最终目标

完成后，新电脑应当具备以下内容：

- [ ] 完整项目代码、配置、测试和 Git 历史
- [ ] 本地原始数据、处理数据和人工复核表（按实际需要）
- [ ] PVLOF 标定参数、模型输出和实验报告
- [ ] 项目背景文档和算法说明
- [ ] 可重新创建的 Python/uv 环境
- [ ] 可连接训练服务器的 SSH 条件
- [ ] 必要时能够查询旧 Codex 会话归档

---

## 1. 迁移前检查当前项目

当前项目目录：

```text
C:\git-workplace
```

### 1.1 检查 Git 状态

- [ ] 在 PowerShell 进入项目目录

```powershell
cd C:\git-workplace
```

- [ ] 查看未提交和未跟踪文件

```powershell
git status --short
```

- [ ] 查看代码改动概况

```powershell
git diff --stat
```

- [ ] 不要因为迁移而删除或重置未提交修改
- [ ] 如果已经确认改动正确，可以正常提交；不确定时直接保留并复制整个 `.git` 目录

### 1.2 确认关键文件存在

```powershell
$required = @(
  "PROJECT_CONTEXT.md",
  "DECISIONS.md",
  "PVLOF.md",
  "pyproject.toml",
  "uv.lock",
  "src",
  "scripts",
  "configs",
  "tests"
)

$required | ForEach-Object {
  [pscustomobject]@{
    Path = $_
    Exists = Test-Path (Join-Path "C:\git-workplace" $_)
  }
} | Format-Table -AutoSize
```

- [ ] 所有关键项的 `Exists` 都应为 `True`

### 1.3 更新交接文档

- [ ] 在 `PROJECT_CONTEXT.md` 写清当前目标和当前进度
- [ ] 在 `DECISIONS.md` 记录已经确定、不应随意推翻的技术决策
- [ ] 确认 `PVLOF.md` 描述了当前 PVLOF 版本、数学原理、阈值和已知问题
- [ ] 记录当前最新版本（目前重点是 PVLOF V1.6 及后续 V1.7 改进方向）
- [ ] 记录最新有效的标定文件、运行脚本和结果路径

---

## 2. 处理密码和敏感信息

以下内容不能放进普通共享文件夹，也不应提交到 Git：

- [ ] `.env`
- [ ] Elasticsearch 用户名和密码
- [ ] SSH 密码或私钥
- [ ] Codex 的 `auth.json`
- [ ] `.sandbox-secrets`
- [ ] API Key、Cookie、访问令牌

检查 Git 是否意外跟踪 `.env`：

```powershell
git ls-files .env
```

正常情况下不应有输出。

检查 `.gitignore`：

```powershell
Select-String -Path .gitignore -Pattern '^\.env$'
```

- [ ] 将 `.env` 单独加密保存，或者在新电脑依据 `.env.example` 重新创建
- [ ] 不要把密码写进本清单、README、聊天归档或代码

---

## 3. 备份完整项目目录

准备移动硬盘目录：

```text
E:\pv-anomaly-project
```

执行复制：

```powershell
$source = "C:\git-workplace"
$target = "E:\pv-anomaly-project"

robocopy $source $target /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /XJ `
  /XD ".venv" "__pycache__" ".pytest_cache" ".ruff_cache" ".uv-cache" `
  /XF ".env" "debug.log"
```

说明：

- `.venv` 不复制，新电脑使用 uv 重新创建
- `.git` 会被复制，Git 历史和未提交状态能够保留
- `data` 和 `artifacts` 默认会被复制，可能占用较大空间
- `robocopy` 返回代码 `0` 至 `7` 通常不代表失败；`8` 及以上才需要排查

- [ ] 复制结束后没有出现 `FAILED` 文件
- [ ] 移动硬盘空间足够

### 3.1 验证关键内容

```powershell
$target = "E:\pv-anomaly-project"

@(
  "PROJECT_CONTEXT.md",
  "DECISIONS.md",
  "PVLOF.md",
  "pyproject.toml",
  "uv.lock",
  ".git",
  "src",
  "scripts",
  "configs",
  "tests",
  "data",
  "artifacts"
) | ForEach-Object {
  [pscustomobject]@{
    Path = $_
    Exists = Test-Path (Join-Path $target $_)
  }
} | Format-Table -AutoSize
```

- [ ] 需要迁移的所有项目均为 `True`

### 3.2 验证关键文件哈希

```powershell
Get-FileHash `
  C:\git-workplace\PROJECT_CONTEXT.md, `
  C:\git-workplace\DECISIONS.md, `
  C:\git-workplace\PVLOF.md, `
  C:\git-workplace\pyproject.toml, `
  C:\git-workplace\uv.lock

Get-FileHash `
  E:\pv-anomaly-project\PROJECT_CONTEXT.md, `
  E:\pv-anomaly-project\DECISIONS.md, `
  E:\pv-anomaly-project\PVLOF.md, `
  E:\pv-anomaly-project\pyproject.toml, `
  E:\pv-anomaly-project\uv.lock
```

- [ ] 两组对应文件的 SHA256 相同

---

## 4. 备份旧 Codex 会话（可选）

这一步只用于以后追溯旧讨论，不负责恢复项目代码，也不保证旧聊天自动显示在新 Codex 侧边栏。

```powershell
$source = "$env:USERPROFILE\.codex"
$target = "E:\codex-old-computer-archive"

New-Item -ItemType Directory -Force $target | Out-Null

foreach ($name in @("sessions", "archived_sessions", "attachments")) {
  $item = Join-Path $source $name
  if (Test-Path $item) {
    Copy-Item $item $target -Recurse -Force
  }
}
```

不要复制：

```text
auth.json
.sandbox-secrets
installation_id
*.sqlite*
logs*
cache*
tmp*
```

- [ ] 抽查归档内是否含账号、密码或其他不应迁移的信息
- [ ] 不要用该归档覆盖新电脑的 `%USERPROFILE%\.codex`

---

## 5. 检查训练服务器上的独有文件

训练服务器可能包含本地没有的标定结果、模型输出、运行日志和对比表。

服务器项目目录通常为：

```text
/root/pv-anomaly
```

### 5.1 登录服务器

```powershell
ssh -p 22 root@192.168.180.206
```

### 5.2 在服务器上检查重要目录

```bash
cd ~/pv-anomaly
pwd
git status --short
find artifacts/models -maxdepth 3 -type f | sort
find artifacts/reports -maxdepth 3 -type f | sort
```

- [ ] 确认 PVLOF V1.6 标定文件仍在
- [ ] 确认最新人工对比 CSV/XLSX 已保存
- [ ] 确认最新诊断 Parquet 和报告 JSON 已保存
- [ ] 确认服务器是否存在本地没有的脚本或源码修改

### 5.3 从服务器下载重要结果

在本地 PowerShell 中运行，而不是在服务器终端运行：

```powershell
New-Item -ItemType Directory -Force E:\pv-anomaly-server-results | Out-Null

scp -P 22 -r `
  root@192.168.180.206:/root/pv-anomaly/artifacts/models `
  E:\pv-anomaly-server-results\

scp -P 22 -r `
  root@192.168.180.206:/root/pv-anomaly/artifacts/reports `
  E:\pv-anomaly-server-results\
```

如果数据量太大，只下载最新版本的具体目录，不要盲目重复下载全部数据。

- [ ] 服务器独有结果已保存到移动硬盘
- [ ] 下载结果能够正常打开

---

## 6. 在新电脑恢复项目

假设新电脑项目位置为：

```text
D:\git-workplace
```

### 6.1 复制项目

```powershell
$source = "E:\pv-anomaly-project"
$target = "D:\git-workplace"

robocopy $source $target /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /XJ
```

- [ ] 项目目录复制完成
- [ ] `D:\git-workplace\.git` 存在
- [ ] `D:\git-workplace\src`、`scripts`、`configs`、`tests` 存在

### 6.2 安装必要软件

- [ ] 安装 Git
- [ ] 安装 uv
- [ ] 安装 Cursor（如果继续使用）
- [ ] 安装 Codex/ChatGPT 桌面端或相应 IDE 扩展
- [ ] 确保 PowerShell 可以运行 `git --version`
- [ ] 确保 PowerShell 可以运行 `uv --version`

### 6.3 重建 Python 环境

```powershell
cd D:\git-workplace
uv sync
```

如果项目需要所有可选依赖：

```powershell
uv sync --all-extras
```

验证 Python：

```powershell
uv run python --version
```

- [ ] `.venv` 已由 uv 在新电脑重新生成
- [ ] 没有复制使用旧电脑的 `.venv`

### 6.4 恢复环境变量

```powershell
Copy-Item .env.example .env
```

- [ ] 手工填写新电脑需要的 Elasticsearch 地址和凭据
- [ ] 不要把 `.env` 提交到 Git
- [ ] 测试环境与生产环境的地址、端口和 plant_id 不要混用

---

## 7. 验证代码和数据

### 7.1 检查 Git 状态

```powershell
cd D:\git-workplace
git status --short
git log -1 --oneline
```

- [ ] Git 能识别仓库
- [ ] 未提交修改与旧电脑迁移前一致

### 7.2 运行测试

先运行核心测试：

```powershell
uv run pytest tests/test_pvlof_v16.py
```

如果测试文件和依赖完整，再运行全部测试：

```powershell
uv run pytest
```

- [ ] 核心测试通过
- [ ] 如有失败，先确认是否缺少数据/标定文件，不要立即改算法

### 7.3 检查关键数据目录

```powershell
Get-ChildItem data -Directory
Get-ChildItem artifacts\models -Directory
Get-ChildItem artifacts\reports -Directory
```

- [ ] 生产电站 234、791、892 的所需数据存在
- [ ] 测试环境 33 号电站的所需验证数据存在
- [ ] weather_15min 数据存在
- [ ] PVLOF V1.6 标定与运行结果存在

---

## 8. 让新电脑上的 Codex 接续工作

### 8.1 建立本地项目

- [ ] 在 Codex 中添加 `D:\git-workplace` 为本地项目
- [ ] 将其设置为主目录（Primary folder）
- [ ] 不要把 `.codex` 会话归档设置为主项目目录
- [ ] 如有需要，可将旧会话归档作为第二个只读参考目录

### 8.2 新聊天的第一条消息

复制以下内容发送给新电脑上的 Codex：

```text
这是从另一台电脑迁移来的 PV 异常检测项目。

请先完整阅读：
1. PROJECT_CONTEXT.md
2. DECISIONS.md
3. PVLOF.md
4. DEVICE_MIGRATION_TODO.md

然后：
1. 检查当前 Git 状态；
2. 识别现有 PVLOF V1.6 实现、标定文件和测试；
3. 总结当前已完成工作、已知漏检原因和下一步 V1.7 改进方向；
4. 不要重新实现已经完成的版本；
5. 涉及任何代码运行时，先把命令给我，由我运行并把输出发给你；
6. 不要修改原始数据和已冻结版本，新增实验必须使用新版本名和新输出目录。
```

根据 OpenAI 官方说明，本地项目会让 Codex 访问关联的文件夹；主目录也是 Git 操作以及自动发现 `AGENTS.md`、技能和配置的默认位置：

<https://learn.chatgpt.com/docs/projects>

---

## 9. 日常继续工作的标准流程

每次开始工作：

- [ ] 进入正确项目目录
- [ ] 查看 `git status --short`
- [ ] 明确本次实验版本和输出目录
- [ ] 先读取最新 `DECISIONS.md` 和 `PVLOF.md`
- [ ] 确认操作发生在本地还是训练服务器

修改代码后：

- [ ] 运行对应单元测试
- [ ] 在小范围数据上试运行
- [ ] 再运行完整验证区间
- [ ] 使用新目录保存输出，不覆盖已冻结版本
- [ ] 生成对比表并人工抽查
- [ ] 将结论写入 `DECISIONS.md`
- [ ] 将新版本原理与已知问题写入 `PVLOF.md`
- [ ] 查看 `git diff --stat`
- [ ] 确认正确后再提交 Git

涉及服务器时：

- [ ] 明确命令应在本地 PowerShell 还是服务器 Bash 执行
- [ ] 上传前确认本地文件路径
- [ ] 运行前进入 `~/pv-anomaly`
- [ ] 服务器上优先使用 `uv run python ...`
- [ ] 输出完成后下载报告和人工对比表到本地

---

## 10. 常见错误排查

### 找不到 Python 模块

```text
ModuleNotFoundError: No module named 'pv_anomaly'
```

检查：

```powershell
cd D:\git-workplace
uv sync
uv run python -c "import pv_anomaly; print(pv_anomaly.__file__)"
```

### 找不到脚本

先检查当前目录：

```powershell
Get-Location
Test-Path .\scripts\目标脚本.py
```

服务器上检查：

```bash
pwd
cd ~/pv-anomaly
ls scripts
```

### 找不到标定文件

```powershell
Get-ChildItem artifacts\models -Recurse -Filter *.json
```

不要随便拿旧版本标定文件替代；先确认脚本版本、标定版本和输入数据是否匹配。

### Codex 看不到项目文件

- [ ] 确认打开的是本地项目，不是普通快速聊天
- [ ] 确认主目录是 `D:\git-workplace`
- [ ] 明确告诉 Codex 要读取的文件路径
- [ ] 检查文件是否确实已经复制到新电脑

### 旧聊天没有出现在侧边栏

这是正常情况。旧 `.codex` 文件夹是追溯归档，不是受支持的聊天合并包。继续工作应以项目文件和 Markdown 文档为准。

---

## 11. 迁移完成验收

只有下面全部完成，才算迁移成功：

- [ ] 新电脑能打开完整 Git 项目
- [ ] `uv sync` 成功
- [ ] `uv run python --version` 成功
- [ ] 核心 PVLOF 测试成功
- [ ] 能读取本地 Parquet、CSV 和 JSON 报告
- [ ] PVLOF V1.6 源码和标定文件存在
- [ ] 新电脑 Codex 已阅读三个核心 Markdown 文档
- [ ] 新电脑 Codex 能准确总结当前算法状态
- [ ] 能 SSH 登录训练服务器
- [ ] 能从服务器下载生成结果
- [ ] `.env`、账号密码和认证文件没有进入普通共享目录或 Git
- [ ] 移动硬盘备份保留到新电脑稳定运行一段时间后再决定是否清理

---

## 12. 最重要的记忆口诀

```text
代码和数据看项目目录；
长期背景看 Markdown；
旧聊天只用于追溯；
环境使用 uv 重建；
密码永远单独保存；
冻结版本绝不覆盖；
运行命令先确认本地还是服务器。
```
