"""
数据库 Agent：处理来自数据库的结构化数据查询。

模型只会获得一个固定的只读查询工具，不会获得直接执行原始 SQL 的能力。
"""
import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from utils.llm import get_chat_llm

from .state import DatabaseAgentState
from .tools import (
    JobApplicationSearchInput,
    create_job_application_tool,
    search_job_applications,
)


DATABASE_PROMPT = """You are a structured-data assistant for a job-search product.
Use the search_job_applications tool whenever the user asks about saved or
tracked job applications. The tool accepts validated filters, not raw SQL.
Only describe records returned by the tool. Never invent application rows,
statuses, companies, dates, or query results. If the tool reports an error,
explain that live records could not be retrieved. Call the tool through the
provided tool interface instead of writing a JSON tool call in your answer.
After receiving tool results, answer in natural language and never print raw
JSON or function-call syntax. Keep previous conversation context in mind."""


DatabaseQueryExecutor = Callable[..., dict[str, Any]]
logger = logging.getLogger(__name__)


# 把不同形式的模型返回值统一转换为 AIMessage。
def _as_ai_message(response: Any) -> AIMessage:
    """统一模型输出格式，因为测试中注入的模型可能返回字符串。"""
    if isinstance(response, AIMessage):
        return response
    if isinstance(response, BaseMessage):
        return AIMessage(content=str(response.content))
    return AIMessage(content=str(response))


def _recover_text_tool_calls(response: Any) -> list[dict[str, Any]]:
    """恢复被模型写进 content、而不是 tool_calls 字段的工具调用。"""
    content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        return []

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []

    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        candidates = payload["tool_calls"]
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = [payload]

    recovered: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue

        name = candidate.get("name")
        arguments = candidate.get("arguments", candidate.get("args", {}))
        if name != create_job_application_tool().name or not isinstance(arguments, dict):
            continue

        recovered.append(
            {
                "name": name,
                "args": arguments,
                "id": str(candidate.get("id") or f"text-tool-call-{index}"),
            }
        )

    return recovered


_STATUS_LABELS = {
    "draft": "草稿",
    "applied": "已投递",
    "interviewing": "面试中",
    "offer": "已获录用意向",
    "rejected": "已拒绝",
    "withdrawn": "已撤回",
}


def _format_tool_messages(tool_messages: list[ToolMessage]) -> str:
    """把已完成的工具查询结果直接格式化为可读文字。"""
    payloads: list[dict[str, Any]] = []
    for message in tool_messages:
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)

    return _format_query_payloads(payloads)


def _format_query_payloads(payloads: list[dict[str, Any]]) -> str:
    """将查询结果字典列表格式化为可读文字。"""
    if not payloads:
        return "查询已完成，但返回的数据无法读取。"

    failed = next((payload for payload in payloads if not payload.get("ok")), None)
    if failed is not None:
        return "抱歉，暂时无法读取实时求职申请记录，请稍后再试。"

    rows = [
        row
        for payload in payloads
        for row in payload.get("rows", [])
        if isinstance(row, dict)
    ]
    if not rows:
        return "没有找到符合条件的求职申请。"

    lines = [f"查询到 {len(rows)} 条求职申请："]
    for row in rows:
        company = str(row.get("company") or "未知公司")
        role_title = str(row.get("role_title") or "未知岗位")
        status = str(row.get("status") or "未知状态")
        status_label = _STATUS_LABELS.get(status, status)
        applied_at = str(row.get("applied_at") or "未知日期")
        lines.append(
            f"- {company}｜{role_title}｜状态：{status_label}｜投递日期：{applied_at}"
        )
    return "\n".join(lines)


def _fallback_database_answer(
    messages: list[BaseMessage],
    executor: DatabaseQueryExecutor,
) -> str:
    """模型漏掉工具调用时，执行一次只读查询并按当前问题筛选结果。"""
    try:
        result = executor(limit=25)
    except Exception:
        logger.exception("Fallback database query failed safely")
        return "抱歉，暂时无法读取实时求职申请记录，请稍后再试。"

    if not isinstance(result, dict):
        return "查询已完成，但返回的数据无法读取。"
    if not result.get("ok"):
        return _format_query_payloads([result])

    current_text = next(
        (
            str(message.content).casefold()
            for message in reversed(messages)
            if getattr(message, "type", None) == "human"
        ),
        "",
    )
    rows = result.get("rows", [])
    if current_text and isinstance(rows, list):
        matched_rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and any(
                value
                and str(value).casefold() in current_text
                for value in (row.get("company"), row.get("role_title"))
            )
        ]
        if matched_rows:
            result = {**result, "count": len(matched_rows), "rows": matched_rows}

    return _format_query_payloads([result])


# ：它把 LLM 生成的 tool_calls 转换成真实、受控的数据库查询，
# 再把查询结果转换成 LLM 能继续读取的 ToolMessage。
def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    executor: DatabaseQueryExecutor,
) -> list[ToolMessage]:
    """只执行模型生成的求职申请筛选条件，并返回安全的工具消息。"""
    # tool_messages 用来收集所有工具调用结果。
    tool_messages: list[ToolMessage] = []
    # 函数首先创建一个工具，并取得工具名称：
    tool_name = create_job_application_tool().name
    # enumerate(): 遍历一个列表时，同时得到每个元素的下标和元素本身。
    for index, tool_call in enumerate(tool_calls):
        # 正常情况下，模型会提供类似 call_001 的 ID。
        # 如果没提供 就会使用 index来表达id
        tool_call_id = str(tool_call.get("id") or f"database-call-{index}")
        # 这个 ID 很重要，因为最终生成的 ToolMessage
        # 必须通过 tool_call_id 与模型发起的那次工具调用对应起来。
        try:
            # 检查工具名称是否合法
            if tool_call.get("name") != tool_name:
                result = {
                    "ok": False,
                    "error": "The requested database tool is not available.",
                }
            # args具备这些字段 来源于 tools.py 中的 args_schema=JobApplicationSearchInput,
            # args：{"status","company","role_keyword","limit"}
            else:
                raw_arguments = dict(tool_call.get("args") or {})
                try:
                    requested_limit = int(raw_arguments.get("limit", 10))
                except (TypeError, ValueError):
                    requested_limit = 10
                raw_arguments["limit"] = min(25, max(1, requested_limit))
                # 模型第一次返回的args不一定符合对象 所以要二次校验
                arguments = JobApplicationSearchInput.model_validate(raw_arguments)
                # arguments.model_dump() 会把经过校验的 Pydantic 对象转换成字典
                # 然后通过 ** 展开成关键字参数
                # executor 就是：search_job_applications(...)
                result = executor(**arguments.model_dump())
                # 项目规定 executor 必须返回字典。
                if not isinstance(result, dict):
                    result = {
                        "ok": False,
                        "error": "The database tool returned an invalid result.",
                    }
        except Exception:
            logger.exception("Database tool execution failed safely")
            # 错误结果 变成 ToolMessage 交给模型。模型可以据此回答：
            result = {
                "ok": False,
                "error": "The business database query failed safely.",
            }
        # 将结果转换成 ToolMessage
        tool_messages.append(
            ToolMessage(
                # json.dumps() 把字典序列化。
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tool_call_id,
            )
        )

    return tool_messages



def create_database_agent_node(
    llm: Any | None = None,
    query_executor: DatabaseQueryExecutor | None = None,
) -> Callable[[DatabaseAgentState], dict[str, list[BaseMessage]]]:
    """创建可注入模型和查询执行器的数据库节点。"""

    def database_query(state: DatabaseAgentState) -> dict[str, list[BaseMessage]]:
        model = llm or get_chat_llm()
        # executor = search_job_applications:一个可调用的 Python 函数
        executor = query_executor or search_job_applications
        prompt_messages = [
            SystemMessage(content=DATABASE_PROMPT),
            *state["messages"],
        ]
        # 先检查模型对象支不支持bind_tools()
        bind_tools = getattr(model, "bind_tools", None)
        if not callable(bind_tools):
            return {"messages": [_as_ai_message(model.invoke(prompt_messages))]}
        # tool指向一个代表 search_job_applications 的 LangChain 工具对象。
        tool = create_job_application_tool()
        # 模型绑定工具
        model_with_tools = bind_tools([tool])
        response = model_with_tools.invoke(prompt_messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            tool_calls = _recover_text_tool_calls(response)
            if tool_calls:
                response = AIMessage(content="", tool_calls=tool_calls)
        # 数据库路由已经确定；模型漏掉工具调用时用只读查询兜底。
        if not tool_calls:
            return {
                "messages": [
                    AIMessage(content=_fallback_database_answer(state["messages"], executor))
                ]
            }

        tool_messages = _execute_tool_calls(tool_calls, executor)
        return {"messages": [AIMessage(content=_format_tool_messages(tool_messages))]}

    return database_query


database_query_node = create_database_agent_node()


def create_database_agent_graph(
    llm: Any | None = None,
    query_executor: DatabaseQueryExecutor | None = None,
):
    """创建数据库 Agent 图。"""
    workflow = StateGraph(DatabaseAgentState)

    workflow.add_node(
        "database_query",
        create_database_agent_node(llm=llm, query_executor=query_executor),
    )

    workflow.add_edge(START, "database_query")
    workflow.add_edge("database_query", END)

    return workflow
