"""
共享多智能体应用图的 FastAPI 入口。

自动路由和手动路由使用同一张图和同一个检查点存储，
区别仅在于请求是否设置了 ``forced_agent``。
"""
# 把异步生成器转换成 FastAPI 可以使用的生命周期上下文管理器。下面的 lifespan 依赖它。
from contextlib import asynccontextmanager
import json # 后面 /query/stream 把事件转换成 SSE 数据时使用。
from pathlib import Path
import uuid
from typing import Literal, Optional # Optional：表示某个值可以是 None。

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

# 在导入图相关模块或者创建配置好的 checkpointer 之前，先加载 .env。
# 具体顺序是：
# 1. 导入 load_environment。
# 2. 执行 load_environment()。
# 3. 把项目根目录 .env 中的变量写入当前进程环境。
# 4. 后面导入图模块并创建 checkpointer。
# 5. get_checkpointer() 读取 CHECKPOINTER_BACKEND，从而决定使用 Memory 还是 MySQL。
from utils.env import load_environment

load_environment()

from graphs.application import build_application_graph
from utils.checkpointer import close_checkpointer, get_checkpointer

# 加载 .env、定义 FastAPI 的生命周期管理、确保服务关闭时释放数据库连接。
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用关闭时释放数据库检查点连接。"""
    yield
    # 关闭 MySQL checkpointer 的底层连接
    close_checkpointer(checkpointer)


app = FastAPI(title="Multi-Agent Career Assistant API", lifespan=lifespan)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = PROJECT_ROOT / "static" / "index.html"

# 应用启动时按 CHECKPOINTER_BACKEND 创建一次，并在所有接口间共享。
checkpointer = get_checkpointer()
app_graph = build_application_graph(checkpointer=checkpointer)

# 这是为了兼容旧代码导入而设置的别名。
# 它们故意指向同一个对象；
# 任何接口都不应该再创建第二个图或第二个 checkpointer。
multi_agent_graph = app_graph
business_graph = app_graph
database_graph = app_graph

ApiAgent = Literal["business", "database", "finish", "unknown"]
ExecutionStatus = Literal["ok", "degraded", "error"]

MANUAL_TO_ROUTE = {
    "business": "business_agent",
    "database": "database_agent",
}

ROUTE_TO_API = {
    "business_agent": "business",
    "database_agent": "database",
    "finish": "finish",
    "error": "unknown",
}


class QueryRequest(BaseModel):
    """手动选择 Agent 的请求模型。"""

    message: str = Field(min_length=1)
    thread_id: Optional[str] = None
    agent_type: Literal["business", "database"] = "business"


class AutoQueryRequest(BaseModel):
    """由 Supervisor 自动路由的请求模型。"""

    message: str = Field(min_length=1)
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    """包含路由信息和错误状态的稳定 API 响应。"""

    response: str
    thread_id: str
    agent_type: ApiAgent
    status: ExecutionStatus = "ok"
    error: Optional[str] = None


class ConversationHistory(BaseModel):
    """单个会话线程的序列化聊天历史。"""

    messages: list[dict[str, str]]
    thread_id: str


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# 生成的是本次请求要交给图的状态更新片段。
def _request_state(message: str, forced_agent: Optional[str]) -> dict:
    """
    显式构造每个仅属于本次请求的标量字段。

    检查点中的值会合并到下一次状态更新中。将 ``forced_agent`` 发送为
    ``None``，可以避免手动路由泄漏到同一线程后续的自动请求中。
    """
    return {
        "messages": [HumanMessage(content=message)],
        "forced_agent": forced_agent,
        "next": None,
        "route_reason": "",
        "status": "ok",
        "error": None,
    }


def _last_message_content(messages: list[BaseMessage]) -> str:
    if not messages:
        return "No response was produced."
    return str(messages[-1].content)



# 把图结果变成 API 响应
def _public_response(
    result: dict, # result：LangGraph 执行完后的完整 State 字典
    thread_id: str,
    *,
    requested_agent: Optional[str] = None, # 手动接口指定过的 Agent
) -> QueryResponse:
    status = result.get("status", "ok")
    if status not in {"ok", "degraded", "error"}:
        status = "error"

    # 不直接暴露内部 error
    if status == "error":
        public_error = "The request could not be completed. Please try again."
    elif status == "degraded":
        public_error = "The supervisor returned an invalid route; a safe fallback handled the request."
    else:
        public_error = None
    # 自动接口调用时会根据next 手动接口调用时根据这个requested_agent
    agent_type = requested_agent or ROUTE_TO_API.get(result.get("next"), "unknown")
    return QueryResponse(
        # 提取最后一条消息
        response=_last_message_content(result.get("messages", [])),
        thread_id=thread_id,
        agent_type=agent_type,
        status=status,
        error=public_error,
    )

# API 层和 LangGraph 之间的连接点。
async def _invoke_app(message: str, thread_id: str, forced_agent: Optional[str]) -> dict:
    return await run_in_threadpool(
        app_graph.invoke,
        _request_state(message, forced_agent),
        _config(thread_id),
    )


@app.get("/")
def root():
    """健康检查与路由能力说明。"""
    return {
        "status": "ok",
        "service": "Multi-Agent Career Assistant API",
        "routing_modes": {
            "automatic": "POST /query/auto - Supervisor routes the request",
            "manual": "POST /query - Caller selects an agent",
        },
        "available_agents": ["business", "database"],
    }


@app.get("/ui", include_in_schema=False)
def frontend():
    """Serve the dependency-free browser client."""
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    # 使用手动指定的专业 Agent 查询同一个应用图。
    thread_id = request.thread_id or str(uuid.uuid4())
    forced_agent = MANUAL_TO_ROUTE[request.agent_type]

    try:
        result = await _invoke_app(request.message, thread_id, forced_agent)
        return _public_response(result, thread_id, requested_agent=request.agent_type)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The request could not be completed safely.",
        ) from exc


@app.post("/query/auto", response_model=QueryResponse)
async def query_agent_auto(request: AutoQueryRequest):
    """通过 Supervisor 路由查询同一张应用图。"""
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        # 把状态交给LangGraph图执行
        result = await _invoke_app(request.message, thread_id, forced_agent=None)
        # 将LangGraph返回的完整state转换成稳定的API响应
        return _public_response(result, thread_id)
    # 第一个 except：保留业务 HTTP 错误的真实状态码。
    # 第二个 except：把未知内部故障安全地转换成 500。
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The request could not be completed safely.",
        ) from exc


# 请求进入 query_agent_stream()
#         ↓
# 创建 event_generator()
#         ↓
# 返回 StreamingResponse
#         ↓
# 客户端开始接收流
#         ↓
# Starlette 不断消费 event_generator()
# 因此接口函数返回时，流不一定已经结束。真正生成内容是在客户端保持连接期间进行的。
# 这个接口不是一次返回完整 JSON，而是把 LangGraph 每执行完一个节点产生的事件实时发送给客户端。
@app.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """从共享应用图流式发送结构化 SSE 事件。"""
    thread_id = request.thread_id or str(uuid.uuid4())
    forced_agent = MANUAL_TO_ROUTE[request.agent_type]

    # event_generator 是一个异步生成器 他不是立即执行完
    # 而是每次产生一个事件：yield f"data: {payload}\n\n"
    async def event_generator():
        stream = None
        try:
            # PyMySQLSaver 只实现同步 checkpoint 接口。同步 stream 交给
            # Starlette 的工作线程执行，避免在事件循环中触发异步 checkpoint 调用。
            stream = app_graph.stream(
                _request_state(request.message, forced_agent),
                config=_config(thread_id),
            )
            async for event in iterate_in_threadpool(stream):
                # default=str 的含义是：如果 event 里面有 Python 对象不能被普通 JSON 序列化 就调用 str() 把它转成字符串
                # ensure_ascii=False 的含义是：中文不转成 \uXXXX 直接保留中文字符
                payload = json.dumps(event, default=str, ensure_ascii=False)
                yield f"event: update\ndata: {payload}\n\n"

            state = await run_in_threadpool(
                app_graph.get_state,
                _config(thread_id),
            )
            result = state.values if state else {}
            final_response = _public_response(
                result,
                thread_id,
                requested_agent=request.agent_type,
            )
            yield (
                "event: final\n"
                f"data: {final_response.model_dump_json()}\n\n"
            )
        except Exception:
            payload = json.dumps(
                {
                    "type": "error",
                    "message": "Streaming failed safely.",
                    "thread_id": thread_id,
                },
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {payload}\n\n"
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                await run_in_threadpool(close_stream)
    # 接口最后返回时才开始执行这个异步生成器
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/conversation/{thread_id}", response_model=ConversationHistory)
async def get_conversation_history(thread_id: str):
    """从处理所有聊天请求的同一张图中读取历史。"""
    try:
        # get_state() 返回的是 LangGraph 的状态快照对象
        state = await run_in_threadpool(app_graph.get_state, _config(thread_id))
        messages = state.values.get("messages", []) if state else []
        # 将HumanMessage和AIMessage转换成可以被前段识别的格式
        serialized = [
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": str(message.content),
            }
            for message in messages
        ]
        # Pydantic 包装成 HTTP 响应
        return ConversationHistory(messages=serialized, thread_id=thread_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Conversation history could not be read safely.",
        ) from exc


@app.delete("/conversation/{thread_id}")
async def delete_conversation(thread_id: str):
    """删除单个会话线程的全部检查点数据。"""
    # 为什么用 getattr()：先检查checkpointer 是否有 delete_thread 方法
    # 支持就delete_thread = checkpointer.delete_thread
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if not callable(delete_thread):
        raise HTTPException(
            status_code=501,
            detail="The configured checkpointer does not support thread deletion.",
        )
    # 等价于：delete_thread(thread_id)
    await run_in_threadpool(delete_thread, thread_id)
    return {"message": f"Conversation {thread_id} was deleted."}


if __name__ == "__main__":
    import uvicorn

    print("Starting Multi-Agent Career Assistant API...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
