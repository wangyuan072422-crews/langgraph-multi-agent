# CareerGraph-AI

CareerGraph-AI 是一个面向 AI 应用开发岗位的求职与面试多智能体助手。项目基于开源
LangGraph 多智能体项目进行二次开发，把演示型多 Agent 流程重构为可运行的求职业务
后端：FastAPI 接入、Supervisor 路由、真实 RAG 检索、只读业务数据查询，以及可选的
MySQL 多轮会话持久化。

P0 阶段已完成以下能力：

- Supervisor 自动路由，以及调用方指定 Agent 的手动路由。
- Business Agent 使用 Chroma 本地向量索引和 Ollama embedding 做真实 RAG 检索。
- Database Agent 通过固定的参数化只读工具查询求职申请数据，不执行模型生成的原始 SQL。
- FastAPI 提供普通响应、SSE 流式响应、会话历史查询和会话删除。
- LangGraph Checkpointer 支持内存模式与 MySQL 模式。
- checkpoint 数据与业务数据使用不同的数据库配置和数据库账号。

## 项目定位

CareerGraph-AI 把多智能体结构落到求职场景，而不是停留在通用聊天演示：

- 知识回答必须经过本地知识库检索，并在回答中标注来源。
- 申请数据查询只走固定参数化工具，不执行模型生成的原始 SQL。
- 会话 checkpoint 与求职业务数据分库、分账号保存。
- 确定性测试覆盖图结构、路由降级、RAG 契约和工具参数化，真实模型与数据库通过独立集成测试验收。

## 架构

```text
HTTP Request
    |
    v
FastAPI (api/server.py)
    |
    +-- /query         手动选择 business / database
    +-- /query/auto    Supervisor 自动选择
    +-- /query/stream  手动选择并返回 SSE
    +-- /conversation  历史查询和删除
    |
    v
graphs/application.py
    |
    +-- supervisor
    |      |
    |      +-- business_agent
    |      +-- database_agent
    |      +-- finish_handler
    |      +-- error_handler
    |
    v
ApplicationState + Checkpointer
    |
    +-- MemorySaver       进程内，仅适合开发和测试
    +-- PyMySQLSaver      MySQL checkpoint 表，跨进程恢复

Business Agent
    -> Chroma PersistentClient
    -> Ollama bge-m3 embedding
    -> data/career_knowledge/*.md

Database Agent
    -> search_job_applications
    -> career_app 只读账号
    -> career_assistant.job_applications
```

## 目录

```text
agents/
  business_agent/       求职知识 RAG 与业务回答
  database_agent/       只读求职申请查询工具与 Agent 节点
  supervisor/           结构化路由和降级处理
graphs/
  application.py        共享应用图工厂
  state.py              ApplicationState 定义
  multi_agent_system.py LangGraph Studio 入口
api/
  server.py             FastAPI 接口层
utils/
  checkpointer.py       MemorySaver / PyMySQLSaver 选择与连接管理
  env.py                .env 加载
  llm.py                Ollama 聊天模型工厂
data/
  career_knowledge/     Markdown 求职知识文档
  chroma/               本地 Chroma 索引，运行时生成
scripts/
  ingest_knowledge.py    构建或更新 Chroma 索引
  init_business_db.sql   创建业务库、表和幂等演示数据
tests/                   确定性测试和可选真实集成测试
examples/                Python 与 API 使用示例
```

## 环境要求

- Python 3.10 或更高版本
- Ollama
- MySQL 8.x
- 可选：Docker / Docker Compose

项目实际使用两个独立数据库：

| 数据 | 默认数据库 | 默认账号 | 用途 |
| --- | --- | --- | --- |
| LangGraph checkpoint | `langgraph_db` | `langgraph_user` | 图状态、消息历史、会话快照 |
| 求职业务数据 | `career_assistant` | `career_app` | `job_applications` 只读查询 |

不要把业务库写权限授予 `career_app`。checkpoint 表由
`PyMySQLSaver.setup()` 自动初始化。

## 最短启动流程

在项目根目录执行：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填入本机密码和模型配置。至少确认：

```dotenv
CHECKPOINTER_BACKEND=mysql
MYSQL_HOST=localhost
MYSQL_USER=langgraph_user
MYSQL_PASSWORD=你的-checkpoint-密码
MYSQL_DATABASE=langgraph_db

BUSINESS_DB_HOST=localhost
BUSINESS_DB_USER=career_app
BUSINESS_DB_PASSWORD=你的只读账号密码
BUSINESS_DB_DATABASE=career_assistant

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_CHAT_MODEL=qwen2.5-coder:7b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
```

启动 Ollama 并准备模型：

```powershell
ollama pull qwen2.5-coder:7b
ollama pull bge-m3:latest
```

用 MySQL 管理员账号初始化业务库和演示数据：

```powershell
mysql -u root -p < scripts/init_business_db.sql
```

初始化 Chroma 索引：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_knowledge.py
```

启动 API：

```powershell
.\.venv\Scripts\python.exe -m api.server
```

打开 `http://127.0.0.1:8000/ui` 使用网页版，或打开
`http://127.0.0.1:8000/docs` 查看接口。

## API

### 手动路由：`POST /query`

请求：

```json
{
  "message": "帮我制定一个行为面试准备计划",
  "thread_id": "可选；不传时服务端生成",
  "agent_type": "business"
}
```

`agent_type` 只允许 `business` 或 `database`。省略时默认是 `business`。

### 自动路由：`POST /query/auto`

请求：

```json
{
  "message": "我目前有哪些求职申请正在面试？",
  "thread_id": "可选；不传时服务端生成"
}
```

Supervisor 根据问题选择 `business_agent`、`database_agent` 或 `finish`。
自动接口不接受 `agent_type`。

### 统一响应

```json
{
  "response": "最终回答",
  "thread_id": "会话 ID",
  "agent_type": "business",
  "status": "ok",
  "error": null
}
```

`status` 可能是 `ok`、`degraded` 或 `error`。`degraded` 表示路由或某个
可恢复环节失败后使用了安全兜底；`error` 表示执行在错误边界停止。

### SSE：`POST /query/stream`

该接口使用手动路由字段。响应类型为 `text/event-stream`：

```text
event: update
data: {"business_agent": {...}}

event: final
data: {"response":"...","thread_id":"...","agent_type":"business","status":"ok","error":null}
```

图节点异常时发送安全的 `event: error`。流结束后，服务端从同一张图和同一个
`thread_id` 读取最终状态，再发送 `event: final`。

### 历史与删除

```text
GET    /conversation/{thread_id}
DELETE /conversation/{thread_id}
```

删除操作调用当前 checkpointer 的 `delete_thread()`。内存模式只能删除当前进程
中的数据；MySQL 模式删除持久化 checkpoint。

## RAG 的实际流程

1. `scripts/ingest_knowledge.py` 读取 `data/career_knowledge/**/*.md`。
2. `agents/business_agent/retriever.py` 按 Markdown 段落切块。
3. 使用 Ollama `bge-m3` 为 chunk 生成 embedding。
4. 使用稳定的 chunk ID 执行 Chroma upsert，因此重复入库是幂等的。
5. Business Agent 只使用本轮用户问题检索 top-k passage。
6. 检索内容和 `source`、`title` 注入业务 Prompt，回答末尾附带来源。
7. 索引为空或 embedding 失败时不会静默回退到关键词检索。

Chroma 默认持久化到 `data/chroma`。更新知识文档后重新执行：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_knowledge.py
```

需要完全重建时：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_knowledge.py --rebuild
```

## 数据库 Agent 工具

Database Agent 只绑定 `search_job_applications`：

```text
status
company
role_keyword
limit
```

工具使用固定 `SELECT`、参数化筛选和 `LIMIT`，不会执行模型生成的原始 SQL。
如果模型没有产生 tool call，Database Agent 不会伪造数据库结果；如果查询失败，
工具返回安全错误，最终回答应明确说明实时记录不可用。

初始化脚本可以重复执行，演示数据按公司、岗位和申请时间检查后再插入，不会删除
已有记录，也不会重复写入同一批 seed 数据。

## Checkpointer 配置

### 内存模式

```dotenv
CHECKPOINTER_BACKEND=memory
```

适合确定性测试和单进程开发。进程重启后历史消失。

### MySQL 模式

```dotenv
CHECKPOINTER_BACKEND=mysql
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=langgraph_user
MYSQL_PASSWORD=你的密码
MYSQL_DATABASE=langgraph_db
```

也可以提供已经 URL 编码的 `MYSQL_URL`。API、直接 Python 运行和示例共用
`utils.checkpointer.get_checkpointer()`，不会在接口内部另建 checkpointer。

`thread_id` 必须放在 LangGraph config：

```python
config = {"configurable": {"thread_id": thread_id}}
```

它不是 `ApplicationState` 字段。

## LangGraph Studio

`graphs/multi_agent_system.py` 故意不传入 checkpointer，由 Studio 管理运行时持久化：

```powershell
langgraph dev
```

Studio 和 FastAPI 使用同一张 `build_application_graph()` 构造的图，但各自管理
持久化边界。

## 测试与验收

### 默认测试

不访问 MySQL 或 Ollama：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

### MySQL checkpoint 真实集成测试

需要当前 `.env` 中的 MySQL 可用：

```powershell
$env:RUN_MYSQL_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m unittest -v tests.test_mysql_integration
```

该测试会写入随机 `thread_id`，通过子进程重新连接 MySQL 恢复状态，然后调用
`delete_thread()` 验证历史消失，并检查 checkpoint 表 collation。

### Chroma + Ollama 真实检索测试

需要 Ollama 和 `bge-m3:latest` 可用：

```powershell
$env:RUN_RAG_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m unittest -v tests.test_business_retriever
```

### 编译和补丁检查

```powershell
.\.venv\Scripts\python.exe -m compileall .
git diff --check
```

真实模型是否产生数据库 tool call，取决于当前 Ollama 模型和运行时行为。
确定性测试通过不能替代真实 Ollama、MySQL 或 HTTP 端到端验收。

## 示例

```powershell
# 直接运行统一图，默认演示多轮上下文
.\.venv\Scripts\python.exe main.py

# 使用当前 .env 中的 MySQL checkpoint
.\.venv\Scripts\python.exe main.py --mysql

# 调用 API 的手动、自动、流式、历史与删除接口
.\.venv\Scripts\python.exe examples\api_client_example.py
```

## Docker

`Dockerfile` 只构建应用服务。MySQL 和 Ollama 仍由外部服务提供。

```powershell
docker compose up --build
```

默认通过 `host.docker.internal` 访问宿主机服务。若 MySQL 或 Ollama 运行在
其他地址，在 `.env` 或 shell 中设置对应的 Docker 覆盖变量。

Docker 配置已提供，但当前机器没有 Docker 命令，因此未本机实测。

## 已知边界

- Ollama 的 Supervisor 路由和 Database Agent tool calling 可能受模型规模、
  Prompt 和本机推理状态影响，不能把一次成功当成稳定生产保证。
- Chroma 索引需要显式构建；API 启动不会自动重新 embedding 全部文档。
- 当前没有上传文档、用户鉴权、审批流和多租户隔离。
- 当前测试覆盖图结构、路由降级、RAG 接口约定和工具参数化；真实模型和真实
  数据库需要在目标环境单独验收。
