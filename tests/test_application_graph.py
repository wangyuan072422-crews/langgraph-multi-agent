"""共享应用图的确定性测试。"""
import asyncio
import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agents.business_agent import (
    KnowledgeChunk,
    create_business_agent_node,
)
from agents.supervisor import RouteDecision, create_supervisor_node
from graphs.application import SAFE_ERROR_MESSAGE, build_application_graph

# 自动检查以前的功能有没有被改坏


#假 LLM。不连接 Ollama，invoke() 固定返回构造时传入的文本，
# 同时记录收到的消息，便于检查提示词。
class StaticChatModel:
    # 保存固定回答 content，初始化 received_messages。
    def __init__(self, content: str):
        self.content = content
        self.received_messages: list[BaseMessage] = []
    # 记录消息，返回 AIMessage(content=self.content)。它只需要长得像真实模型即可。
    def invoke(self, messages):
        self.received_messages = list(messages)
        return AIMessage(content=self.content)


# 假 LangGraph。替代真实图，记录 API 传给图的 payload 和 config。
class RecordingApiGraph:
    # 保存默认路由、调用记录和消息记录
    def __init__(self, route: str):
        self.route = route
        self.calls = []
        self.messages = []
    # 记录调用参数，把用户消息加上固定 "API answer"，返回模拟的图状态。
    def invoke(self, payload, config):
        self.calls.append((payload, config))
        self.messages = [*payload["messages"], AIMessage(content="API answer")]
        return {
            **payload,
            "messages": self.messages,
            # 	如果存在 forced_agent 就使用它，否则使用构造时传入的路由。
            "next": payload.get("forced_agent") or self.route,
            "status": "ok",
            "error": None,
        }
    # 模拟 LangGraph 的 get_state()，用 SimpleNamespace 伪造带 .values 的状态快照。
    def get_state(self, _config):
        return SimpleNamespace(values={"messages": self.messages})


class RecordingStreamingGraph:
    """Fake graph used to verify SSE framing without calling Ollama."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []
        self.messages = [
            HumanMessage(content="Stream this request"),
            AIMessage(content="Streaming answer"),
        ]

    def stream(self, payload, config):
        self.calls.append((payload, config))
        yield {"supervisor": {"next": "business_agent", "status": "ok"}}
        yield {"business_agent": {"status": "ok"}}
        if self.fail:
            raise RuntimeError("stream failed")

    def get_state(self, _config):
        return SimpleNamespace(
            values={
                "messages": self.messages,
                "next": "business_agent",
                "status": "ok",
                "error": None,
            }
        )


class FixedKnowledgeRetriever:
    """Deterministic retriever used by shared-graph tests."""

    def __init__(self, chunks: list[KnowledgeChunk]):
        self.chunks = chunks

    def retrieve(self, _query: str, *, limit: int = 3) -> list[KnowledgeChunk]:
        return self.chunks[:limit]


def create_test_business_node(model: StaticChatModel):
    return create_business_agent_node(
        llm=model,
        retriever=FixedKnowledgeRetriever([]),
    )



# 构造一次图请求的标准 State 输入，统一设置 messages、forced_agent、status 等字段。
def request_state(message: str, *, forced_agent: str | None = None) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "forced_agent": forced_agent,
        "next": None,
        "route_reason": "",
        "status": "ok",
        "error": None,
    }


# 测试类，继承 unittest.TestCase，提供断言和测试运行能力。
class ApplicationGraphTests(unittest.TestCase):
    def test_explicit_chinese_application_query_bypasses_the_model_router(self):
        class FailingRouterModel:
            def with_structured_output(self, _schema):
                raise AssertionError("The model router must not decide this explicit route")

        supervisor = create_supervisor_node(llm=FailingRouterModel())

        result = supervisor(request_state("帮我查询所有求职申请。"))

        self.assertEqual(result["next"], "database_agent")
        self.assertEqual(result["status"], "ok")

    def test_application_status_follow_up_bypasses_the_model_router(self):
        class FailingRouterModel:
            def with_structured_output(self, _schema):
                raise AssertionError("The model router must not decide this follow-up")

        supervisor = create_supervisor_node(llm=FailingRouterModel())
        result = supervisor(
            {
                "messages": [
                    HumanMessage(content="帮我查询所有求职申请。"),
                    AIMessage(content="查询到 3 条求职申请。"),
                    HumanMessage(content="其中 Acme Cloud 现在进展到哪一步了？"),
                ]
            }
        )

        self.assertEqual(result["next"], "database_agent")
        self.assertEqual(result["status"], "ok")

    def test_manual_route_bypasses_supervisor(self):
        def fail_if_called(_messages):
            raise AssertionError("Supervisor must not run for a forced route")

        business_model = StaticChatModel("Business answer")
        graph = build_application_graph(
            supervisor_route_decider=fail_if_called,
            business_node=create_test_business_node(business_model),
        )

        result = graph.invoke(request_state("Review this job description", forced_agent="business_agent"))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["next"], "business_agent")
        self.assertIn("Business answer", result["messages"][-1].content)
        self.assertNotIn("Routing to:", " ".join(message.content for message in result["messages"]))

    def test_structured_supervisor_route_keeps_history_clean(self):
        database_model = StaticChatModel("Database answer")
        graph = build_application_graph(
            supervisor_route_decider=lambda _messages: RouteDecision(
                next="database_agent",
                reason="The request asks for structured records.",
            ),
            database_llm=database_model,
        )

        result = graph.invoke(request_state("Show my saved applications"))

        self.assertEqual(result["next"], "database_agent")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["messages"]), 2)
        self.assertIsInstance(result["messages"][0], HumanMessage)
        self.assertEqual(result["messages"][-1].content, "Database answer")
        self.assertIsInstance(database_model.received_messages[0].type, str)
        self.assertEqual(database_model.received_messages[0].type, "system")
        self.assertEqual(database_model.received_messages[-1].content, "Show my saved applications")

    def test_invalid_route_degrades_to_business_agent(self):
        business_model = StaticChatModel("Fallback answer")
        graph = build_application_graph(
            supervisor_route_decider=lambda _messages: "unknown_agent",
            business_node=create_test_business_node(business_model),
        )

        result = graph.invoke(request_state("Help me prepare for an interview"))

        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["next"], "business_agent")
        self.assertIn("Fallback answer", result["messages"][-1].content)
        self.assertIn("unsupported route", result["error"])

    def test_supervisor_can_finish_without_calling_a_specialist(self):
        graph = build_application_graph(
            supervisor_route_decider=lambda _messages: RouteDecision(
                next="finish",
                reason="No specialist is needed.",
            ),
            business_node=lambda _state: self.fail(
                "business_agent must not run for a finish route"
            ),
            database_node=lambda _state: self.fail(
                "database_agent must not run for a finish route"
            ),
        )

        result = graph.invoke(request_state("No further action is needed."))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["next"], "finish")
        self.assertIn("No further specialist action", result["messages"][-1].content)

    def test_agent_exception_uses_safe_error_boundary(self):
        def broken_business_node(_state):
            raise RuntimeError("secret internal detail")

        graph = build_application_graph(
            supervisor_route_decider=lambda _messages: RouteDecision(next="business_agent"),
            business_node=broken_business_node,
        )

        result = graph.invoke(request_state("Analyze this role"))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["next"], "error")
        self.assertEqual(result["messages"][-1].content, SAFE_ERROR_MESSAGE)
        self.assertNotIn("secret internal detail", result["messages"][-1].content)
        self.assertIn("secret internal detail", result["error"])

    def test_supervisor_exception_uses_safe_error_boundary(self):
        def broken_router(_messages):
            raise RuntimeError("model connection failed")

        graph = build_application_graph(
            supervisor_route_decider=broken_router,
            business_llm=StaticChatModel("should not run"),
        )

        result = graph.invoke(request_state("Route this request"))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["messages"][-1].content, SAFE_ERROR_MESSAGE)
        self.assertNotIn("model connection failed", result["messages"][-1].content)

    def test_multiple_turns_accumulate_messages_in_one_thread(self):
        checkpointer = MemorySaver()
        business_model = StaticChatModel("Answer")
        graph = build_application_graph(
            checkpointer=checkpointer,
            supervisor_route_decider=lambda _messages: RouteDecision(next="business_agent"),
            business_node=create_test_business_node(business_model),
        )
        config = {"configurable": {"thread_id": "thread-1"}}

        graph.invoke(request_state("First question"), config=config)
        result = graph.invoke(request_state("Second question"), config=config)

        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual(result["messages"][-2].content, "Second question")
        state = graph.get_state(config)
        self.assertEqual(len(state.values["messages"]), 4)

    def test_new_thread_does_not_reuse_previous_context(self):
        checkpointer = MemorySaver()
        graph = build_application_graph(
            checkpointer=checkpointer,
            supervisor_route_decider=lambda _messages: RouteDecision(next="business_agent"),
            business_node=create_test_business_node(StaticChatModel("Answer")),
        )

        graph.invoke(
            request_state("Private first conversation"),
            config={"configurable": {"thread_id": "thread-a"}},
        )
        result = graph.invoke(
            request_state("New conversation"),
            config={"configurable": {"thread_id": "thread-b"}},
        )

        self.assertEqual(len(result["messages"]), 2)

    def test_rag_follow_up_reuses_context_and_sources_without_cross_thread_leakage(self):
        from agents.business_agent import (
            create_business_agent_node,
        )

        checkpointer = MemorySaver()
        business_model = StaticChatModel("Use STAR with a measurable result.")
        business_node = create_business_agent_node(
            llm=business_model,
            retriever=FixedKnowledgeRetriever(
                [
                    KnowledgeChunk(
                        title="Interview Preparation",
                        source="interview_preparation.md",
                        content="Use the STAR method for behavioral interviews.",
                    )
                ]
            ),
        )
        graph = build_application_graph(
            checkpointer=checkpointer,
            supervisor_route_decider=lambda _messages: RouteDecision(next="business_agent"),
            business_node=business_node,
        )
        first_thread = {"configurable": {"thread_id": "rag-thread-a"}}

        graph.invoke(
            request_state("How should I prepare for a behavioral interview?"),
            config=first_thread,
        )
        follow_up = graph.invoke(
            request_state("Can you explain the STAR method?"),
            config=first_thread,
        )

        self.assertEqual(len(follow_up["messages"]), 4)
        self.assertEqual(
            [message.content for message in follow_up["messages"][::2]],
            [
                "How should I prepare for a behavioral interview?",
                "Can you explain the STAR method?",
            ],
        )
        self.assertEqual(len(business_model.received_messages), 4)
        self.assertIn(
            "interview_preparation.md",
            business_model.received_messages[0].content,
        )
        self.assertIn("Sources:", follow_up["messages"][-1].content)
        self.assertTrue(follow_up["messages"][-1].additional_kwargs["sources"])

        other_thread = graph.invoke(
            request_state("Start an unrelated conversation"),
            config={"configurable": {"thread_id": "rag-thread-b"}},
        )

        self.assertEqual(len(other_thread["messages"]), 2)
        self.assertEqual(len(business_model.received_messages), 2)
        self.assertNotIn(
            "How should I prepare for a behavioral interview?",
            " ".join(
                message.content for message in business_model.received_messages
            ),
        )

    def test_api_legacy_graph_names_share_one_instance_and_checkpointer(self):
        from api import server

        self.assertIs(server.app_graph, server.multi_agent_graph)
        self.assertIs(server.app_graph, server.business_graph)
        self.assertIs(server.app_graph, server.database_graph)
        self.assertIsNotNone(server.app_graph.checkpointer)
        self.assertIn("supervisor", server.app_graph.nodes)
        self.assertIn("business_agent", server.app_graph.nodes)
        self.assertIn("database_agent", server.app_graph.nodes)

    def test_api_uses_the_configured_checkpointer_factory(self):
        from api import server
        from utils import checkpointer as checkpointer_module

        configured_checkpointer = MemorySaver()
        try:
            with patch.object(
                checkpointer_module,
                "get_checkpointer",
                return_value=configured_checkpointer,
            ) as factory:
                reloaded_server = importlib.reload(server)

            factory.assert_called_once_with()
            self.assertIs(reloaded_server.checkpointer, configured_checkpointer)
            self.assertIs(reloaded_server.app_graph.checkpointer, configured_checkpointer)
        finally:
            importlib.reload(server)

    def test_api_shutdown_closes_the_shared_checkpointer(self):
        from api import server

        async def run_lifespan():
            async with server.lifespan(server.app):
                pass

        with patch.object(server, "close_checkpointer") as close:
            asyncio.run(run_lifespan())

        close.assert_called_once_with(server.checkpointer)

    def test_api_auto_request_explicitly_clears_forced_agent(self):
        from api import server

        payload = server._request_state("Find matching jobs", forced_agent=None)

        self.assertIn("forced_agent", payload)
        self.assertIsNone(payload["forced_agent"])
        self.assertEqual(payload["status"], "ok")

    def test_api_auto_and_history_use_the_same_graph(self):
        from api import server

        fake_graph = RecordingApiGraph(route="database_agent")
        with patch.object(server, "app_graph", fake_graph):
            client = TestClient(server.app)
            response = client.post(
                "/query/auto",
                json={"message": "Show my applications", "thread_id": "api-thread"},
            )
            history = client.get("/conversation/api-thread")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["agent_type"], "database")
        self.assertIsNone(fake_graph.calls[0][0]["forced_agent"])
        self.assertEqual(history.json()["messages"][0]["role"], "user")
        self.assertEqual(history.json()["messages"][-1]["content"], "API answer")

    def test_api_stream_emits_update_and_final_sse_events(self):
        from api import server

        fake_graph = RecordingStreamingGraph()
        with patch.object(server, "app_graph", fake_graph):
            client = TestClient(server.app)
            response = client.post(
                "/query/stream",
                json={
                    "message": "Stream this request",
                    "thread_id": "stream-thread",
                    "agent_type": "business",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: update", response.text)
        self.assertIn("event: final", response.text)
        final_payload = response.text.split("event: final\ndata: ", 1)[1].split(
            "\n\n",
            1,
        )[0]
        final_data = json.loads(final_payload)
        self.assertEqual(final_data["response"], "Streaming answer")
        self.assertEqual(final_data["thread_id"], "stream-thread")
        self.assertEqual(final_data["agent_type"], "business")
        self.assertEqual(
            fake_graph.calls[0][0]["forced_agent"],
            "business_agent",
        )

    def test_api_stream_emits_safe_error_event(self):
        from api import server

        fake_graph = RecordingStreamingGraph(fail=True)
        with patch.object(server, "app_graph", fake_graph):
            client = TestClient(server.app)
            response = client.post(
                "/query/stream",
                json={
                    "message": "Trigger a stream failure",
                    "thread_id": "stream-error-thread",
                    "agent_type": "business",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertIn('"message": "Streaming failed safely."', response.text)
        self.assertNotIn("stream failed", response.text)


if __name__ == "__main__":
    # 允许直接运行本文件时启动测试；通过 discover 执行时不会进入这里。
    unittest.main()
