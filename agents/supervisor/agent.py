# 应用图的 Supervisor 路由模块。
# Supervisor 只返回结构化的路由元数据。
# 它不会把路由决策写入 messages，
# 因此“正在将请求路由到某某 Agent”这类文字不会污染用户聊天历史。

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from utils.llm import get_chat_llm

from .state import SupervisorState

# 约束 Supervisor 输出的结构；
RouteName = Literal["business_agent", "database_agent", "finish"]

# 表示一个可调用函数 输入：消息列表  输出任意路由结果
# 主要用于测试时注入确定性路由
RouteDecider = Callable[[list[BaseMessage]], Any]

# 告诉路由模型有哪些 Agent、应该如何选择。
SUPERVISOR_PROMPT = """You route requests to one specialist agent.

Available routes:
- business_agent: career research, company research, job descriptions, interview preparation, and document-grounded questions
- database_agent: requests for structured records or database-backed facts
- finish: only when no additional specialist action is needed

Choose exactly one route. Do not answer the user's question yourself.
Treat the conversation history as context for the current request."""

# 路由模型应当返回的结构化输出。
class RouteDecision(BaseModel):

    next: RouteName = Field(description="The specialist that should handle the request")
    reason: str = Field(default="", max_length=500, description="Short routing rationale")

# 路由别名表：目的是API、测试和模型可以使用不同写法
_ROUTE_ALIASES = {
    "business": "business_agent",
    "business_agent": "business_agent",
    "database": "database_agent",
    "database_agent": "database_agent",
    "finish": "finish",
}

_DATABASE_REQUEST_TERMS = (
    "求职申请",
    "工作申请",
    "申请记录",
    "投递记录",
    "申请进展",
    "申请状态",
    "job application",
    "job applications",
    "saved application",
    "saved applications",
    "tracked application",
    "tracked applications",
)

_DATABASE_FOLLOW_UP_TERMS = (
    "进展",
    "状态",
    "到哪一步",
    "怎么样了",
    "结果如何",
    "status",
    "progress",
)


def _explicit_database_route(messages: list[BaseMessage]) -> RouteName | None:
    """Route explicit application-record requests without relying on the model."""
    human_texts = [
        str(message.content).casefold()
        for message in messages
        if getattr(message, "type", None) == "human"
    ]
    if not human_texts:
        return None

    latest_human_text = human_texts[-1]
    if any(term in latest_human_text for term in _DATABASE_REQUEST_TERMS):
        return "database_agent"

    previous_context = " ".join(human_texts[:-1])
    if any(term in previous_context for term in _DATABASE_REQUEST_TERMS) and any(
        term in latest_human_text for term in _DATABASE_FOLLOW_UP_TERMS
    ):
        return "database_agent"
    return None


# 把各种写法的路由名称转换成标准名称
def _normalize_route(value: Any) -> RouteName | None:
    # 接受三种输入
    if isinstance(value, RouteDecision):
        value = value.next
    elif isinstance(value, dict):
        value = value.get("next")
    # 如果不是字符串就返回None
    if not isinstance(value, str):
        return None
    # 把字符串转成统一格式，再查别名表。
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _ROUTE_ALIASES.get(normalized)


# 从路由结果中提取路由原因
def _decision_reason(value: Any) -> str:
    # 当输入是 RouteDecision 时，直接读取其中的路由原因。
    if isinstance(value, RouteDecision):
        return value.reason

    if isinstance(value, dict):
        reason = value.get("reason", "")
        # 只有 reason 是字符串时才返回；否则返回空字符串。
        return reason if isinstance(reason, str) else ""
    return ""


# 生成统一格式的 State 更新字典
def _route_update(
    next_agent: RouteName,
    reason: str,
    *, # 这里的 * 仍然表示 status 和 error 必须通过参数名传递
    status: Literal["ok", "degraded"] = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "next": next_agent,
        "route_reason": reason,
        "status": status,
        "error": error,
    }


# 创建真正的 Supervisor 节点函数
# 创建一个 Supervisor 节点，可以选择注入模型或测试用路由依赖。
def create_supervisor_node(
    llm: Any | None = None,
    route_decider: RouteDecider | None = None,
) -> Callable[[SupervisorState], dict[str, Any]]:

    def supervisor(state: SupervisorState) -> dict[str, Any]:
        forced_agent = state.get("forced_agent")
        if forced_agent is not None:
            # 弄成标准路由名称
            forced_route = _normalize_route(forced_agent)
            # 路由名称非法就抛出异常
            if forced_route is None:
                raise ValueError(f"Unsupported forced agent: {forced_agent}")
            # 生成统一格式的State更新
            return _route_update(forced_route, "Manual route selected by the caller.")

        messages = list(state.get("messages", []))
        if not messages:
            raise ValueError("Supervisor received no messages.")

        # 如果传入了测试路由函数 就完全绕过LLM
        if route_decider is not None:
            # 传入了测试函数，执行测试函数
            decision = route_decider(messages)
        else:
            explicit_route = _explicit_database_route(messages)
            if explicit_route is not None:
                return _route_update(
                    explicit_route,
                    "The request explicitly asks for application records.",
                )
            # 普通聊天模型 调用后通常返回自由文本
            routing_llm = llm or get_chat_llm()
            # 对模型进行输出约束，让模型输出符合 RouteDecision 的结构。
            structured_llm = routing_llm.with_structured_output(RouteDecision)
            try:
                decision = structured_llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT), *messages])
            except ValidationError:
                # Supervisor 返回的结构化输出格式错误，
                # 因此使用 business_agent 作为安全兜底。
                return _route_update(
                    "business_agent",
                    "The supervisor returned malformed structured output; business_agent was used as a fallback.",
                    status="degraded",
                    error="Supervisor returned malformed structured output.",
                )
        # 校验路由
        # 无论路由来自测试函数还是真实模型，都会执行：
        next_agent = _normalize_route(decision)
        reason = _decision_reason(decision)
        if next_agent is None:
            # Supervisor 返回了不支持的路由，
            # 因此使用 business_agent 作为安全兜底。
            return _route_update(
                "business_agent",
                "The supervisor returned an unsupported route; business_agent was used as a fallback.",
                status="degraded",
                error="Supervisor returned an unsupported route.",
            )
        # 返回合法的路由
        return _route_update(next_agent, reason)

    return supervisor


supervisor_node = create_supervisor_node()


# 老版独立 Supervisor 图使用的条件边函数
def route_to_agent(state: SupervisorState) -> Literal["business_agent", "database_agent", "finish", "error"]:
    # 将应用 State 映射为老版 Supervisor 图中的下一个节点。
    if state.get("status") == "error":
        return "error"
    next_agent = _normalize_route(state.get("next"))
    return next_agent or "error"


# 创建老版独立 Supervisor 路由图
def create_supervisor_graph(
    llm: Any | None = None,
    route_decider: RouteDecider | None = None,
):
    """创建旧版示例使用的独立 Supervisor 工作流。"""
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", create_supervisor_node(llm=llm, route_decider=route_decider))
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "business_agent": END,
            "database_agent": END,
            "finish": END,
            "error": END,
        },
    )
    return workflow
