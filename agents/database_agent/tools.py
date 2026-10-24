"""数据库 Agent 使用的只读业务数据库工具。

业务数据使用独立的 ``BUSINESS_DB_*`` 配置，并与
``CHECKPOINTER_BACKEND`` 选择的检查点数据库有意分开。
"""
from __future__ import annotations

import logging
import os
from typing import Any

import pymysql # Python 连接 MySQL 数据库的驱动
# StructuredTool：把普通 Python 函数封装成 LangChain 可识别的结构化工具
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError, field_validator
# DictCursor	让查询结果的每一行可以按字典形式读取
from pymysql.cursors import DictCursor


logger = logging.getLogger(__name__)

# 模型看到的工具名称
JOB_APPLICATION_TOOL_NAME = "search_job_applications"

# 允许的申请状态列表
ALLOWED_APPLICATION_STATUSES = {
    "draft", # 草稿
    "applied", # 已投递
    "interviewing", # 面试中
    "offer", # 已获录用意向
    "rejected", # 已被拒绝
    "withdrawn", # 已撤回申请
}

# 它既用于生成工具参数 Schema 也用于真正执行查询前再次校验参数。
# 规定查询工具允许接收哪些参数，并检查这些参数是否符合要求。
class JobApplicationSearchInput(BaseModel):
    """只读查询工具接受的、经过校验的筛选条件。"""
    # 申请状态，可省略。
    status: str | None = Field(
        default=None,
        description="Application status, such as applied, interviewing, offer, or rejected.",
    )
    # 公司名称片段，可省略
    company: str | None = Field(
        default=None,
        max_length=100,
        description="Optional company name fragment.",
    )
    # 岗位名称片段
    role_keyword: str | None = Field(
        default=None,
        max_length=100,
        description="Optional role-title keyword, such as backend or data.",
    )
    # 最多返回多少条，默认 10
    limit: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Maximum number of records to return.",
    )
    # 表示这个函数只处理 status 字段，并且运行在 Pydantic 完成普通类型转换和字段校验之前。
    @field_validator("status", mode="before")
    @classmethod
    # cls就是传入的state value就是值
    def normalize_status(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("status must be a string")
        normalized = value.strip().lower()
        # 空字符串和纯空格会被视为“没有提供状态”：
        if not normalized:
            return None
        # 检查状态是否合法
        if normalized not in ALLOWED_APPLICATION_STATUSES:
            allowed = ", ".join(sorted(ALLOWED_APPLICATION_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        return normalized
    # 处理 company role_keyword两个字段 Pydantic 会分别针对这两个字段调用该函数。
    # normalize_optional_text()：规范化公司名和职位关键词
    @field_validator("company", "role_keyword", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("filter values must be strings")
        normalized = value.strip()
        return normalized or None


# _business_database_config()：读取数据库配置
# 从环境变量构造 PyMySQL 连接参数。
def _business_database_config() -> dict[str, Any]:
    """从环境变量读取 MySQL 连接配置，检查必填项和端口、超时参数，并返回一个配置字典。"""
    required_names = (
        "BUSINESS_DB_HOST",
        "BUSINESS_DB_USER",
        "BUSINESS_DB_PASSWORD",
        "BUSINESS_DB_DATABASE",
    )
    # 读取配置 没有就放进missing中
    missing = [name for name in required_names if os.getenv(name) is None]
    if missing:
        raise ValueError(
            "Business database configuration is incomplete. "
            f"Set: {', '.join(missing)}."
        )

    try:
        port = int(os.getenv("BUSINESS_DB_PORT", "3306"))
        connect_timeout = int(os.getenv("BUSINESS_DB_CONNECT_TIMEOUT", "5"))
    except ValueError as exc:
        raise ValueError(
            "BUSINESS_DB_PORT and BUSINESS_DB_CONNECT_TIMEOUT must be integers."
        ) from exc

    return {
        "host": os.environ["BUSINESS_DB_HOST"],
        "user": os.environ["BUSINESS_DB_USER"],
        "password": os.environ["BUSINESS_DB_PASSWORD"],
        "database": os.environ["BUSINESS_DB_DATABASE"],
        "port": port,
        "connect_timeout": connect_timeout,
        "autocommit": True,
        "cursorclass": DictCursor, # 让查询结果的行以字典形式返回
    }

# 真正连接 MySQL
def _connect_business_database():
    """打开仅供单次只读工具调用使用的短连接。"""
    return pymysql.connect(**_business_database_config())


# 将用户输入的文本转换为 SQL LIKE 模糊匹配模式，
# 同时转义其中的特殊字符，尽量避免用户输入的 % 和 _ 被误当成通配符。
def _like_pattern(value: str) -> str:
    """转义用户文本，使通配符字符作为普通筛选文本处理。"""
    escaped = value.replace("/", "//").replace("%", "/%").replace("_", "/_")
    return f"%{escaped}%"


# 负责真正查询数据库。
# 接收筛选条件 → 校验输入 → 生成 SQL → 连接数据库 → 执行查询 → 处理异常 → 返回结果。
def search_job_applications(
    status: str | None = None,
    company: str | None = None,
    role_keyword: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """通过固定的参数化 SELECT 查询已跟踪的求职申请。"""
    try:
        # 校验成功后，filters 是 Pydantic 模型实例，其字段值经过规范化
        filters = JobApplicationSearchInput(
            status=status,
            company=company,
            role_keyword=role_keyword,
            limit=limit,
        )
    except ValidationError:
        return {
            "ok": False,
            "error": "The business database is not configured or the query filters are invalid.",
        }

    # clauses 保存 SQL 条件，parameters 保存条件对应的参数值。
    # 目的：后续添加条件时，就可以统一使用 AND：
    clauses = ["1 = 1"]
    # 保存这些条件对应的实际参数值。
    parameters: list[Any] = []
    if filters.status is not None:
        clauses.append("status = %s")
        parameters.append(filters.status)
    # company = "腾讯" -> parameters = ["%腾讯%"]
    if filters.company is not None:
        clauses.append("company LIKE %s ESCAPE '/'")
        parameters.append(_like_pattern(filters.company))
    if filters.role_keyword is not None:
        clauses.append("role_title LIKE %s ESCAPE '/'")
        parameters.append(_like_pattern(filters.role_keyword))
    # 拼接完整 SQL
    sql = (
        "SELECT id, company, role_title, status, applied_at "
        "FROM job_applications "
        f"WHERE {' AND '.join(clauses)} " # 会把 clauses 中的条件用 AND 连接起来。
        "ORDER BY applied_at DESC, id DESC "
        "LIMIT %s"
    )
    # 因为 SQL 中存在 LIMIT %s，所以参数列表里也必须增加对应的数量。
    parameters.append(filters.limit)

    connection = None
    try:
        # 数据库连接对象
        connection = _connect_business_database()
        # connection.cursor():创建数据库游标
        # with：在代码块结束时，按上下文管理器的规则清理游标资源。
        with connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            rows = list(cursor.fetchall() or [])
    # cursor.fetchall():读取当前查询结果中的所有行。
    # 由于连接配置使用了DictCursor 返回的每一行是一个字典
    # 用于处理 try 范围内抛出的 ValueError
    except ValueError:
        logger.warning("Business database query was rejected because configuration is invalid.")
        return {
            "ok": False,
            "error": "The business database is not configured or the query filters are invalid.",
        }
    # 用于捕获其他常规异常，例如连接失败、SQL 执行错误等。
    except Exception:
        logger.exception("Business database query failed")
        return {
            "ok": False,
            "error": "The business database query failed safely.",
        }
    finally:
        if connection is not None:
            connection.close()

    return {
        "ok": True,
        "count": len(rows),
        "rows": rows,
        "filters": filters.model_dump(), # 会把 Pydantic 模型转换成字典
    }


# search_job_applications：传递函数对象。
# search_job_applications()：立即调用函数，并取得它的返回值。


# 负责将search_job_applications()这个Python函数包装成 Agent 可以使用的工具。
def create_job_application_tool() -> StructuredTool:
    """创建暴露给数据库模型的唯一只读业务工具。"""
    return StructuredTool.from_function(
        func=search_job_applications, # 告诉 LangChain：当这个工具被实际执行时，要调用哪个 Python 函数？
        name=JOB_APPLICATION_TOOL_NAME,
        description=(
            "Search the user's saved job applications. Use this for requests "
            "about tracked applications, application status, companies, or roles. "
            "This tool is read-only and accepts filters instead of raw SQL."
        ),
        args_schema=JobApplicationSearchInput,
    # 告诉 LangChain：这个工具允许接收什么参数，以及参数应该符合什么结构？
    )


__all__ = [
    "ALLOWED_APPLICATION_STATUSES",
    "JOB_APPLICATION_TOOL_NAME",
    "JobApplicationSearchInput",
    "create_job_application_tool",
    "search_job_applications",
]
