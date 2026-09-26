"""MCP 活数据接入(星耀数智 AmazingData)——方案 B 薄封装。

生命周期(D8-A):每个会话起一个 stdio server 子进程,退出即关闭并回收。
AmazingData 单点登录,子进程终止即释放会话,因此本进程内不主动 logout。

安全约束:AmazingData 单账号仅单连接。本模块在线期间,不得与 zvt 侧 xysz 采集进程同时在线。

配置全部来自环境变量(AD_* 四件套未注入时 MCP 整体关闭,静默降级兜底链):
  AD_USERNAME / AD_PASSWORD / AD_HOST / AD_PORT
  AD_MCP_SERVER   server.py 路径(默认本项目内 providers/china/ad_mcp/server.py,随仓库走)
  AD_MCP_PYTHON   跑 server 的 python(默认 sys.executable;该解释器需装 AmazingData + fastmcp)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# AmazingData 单账号仅单连接:全局串行化所有同步调用,防止并行分析师各自起子进程登录互踢。
_MCP_LOCK = threading.Lock()

AD_USERNAME = os.environ.get("AD_USERNAME")
AD_PASSWORD = os.environ.get("AD_PASSWORD")
AD_HOST = os.environ.get("AD_HOST")
AD_PORT = os.environ.get("AD_PORT")
_DEFAULT_MCP_SERVER = Path(__file__).resolve().parent / "ad_mcp" / "server.py"
AD_MCP_SERVER = os.environ.get("AD_MCP_SERVER", str(_DEFAULT_MCP_SERVER))
AD_MCP_PYTHON = os.environ.get("AD_MCP_PYTHON", sys.executable)


def to_int_date(value: Any) -> Optional[int]:
    """schema 改写:工具入参是 8 位 int 日期,对外接受 yyyy-mm-dd 字符串。"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return None
    return int(s.replace("-", "").replace("/", ""))


def to_code_list(codes: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(codes, str):
        return [c.strip() for c in codes.split(",") if c.strip()]
    return list(codes)


def _server_env() -> Dict[str, str]:
    """子进程注入的 env:照抄父进程 + 保证 AD_* 必有键(调用时读,避免 import 期冻结)。"""
    env = dict(os.environ)
    for key in ("AD_USERNAME", "AD_PASSWORD", "AD_HOST", "AD_PORT"):
        env.setdefault(key, "")
    return env


class MCPSession:
    """包装一个已建立的 ClientSession,提供工具调用与 JSON 解析。"""

    def __init__(self, session: ClientSession):
        self._session = session

    async def call(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        """调用 MCP 工具,返回解析后的 dict;非 JSON 原始文本包进 {"raw": ...}。"""
        result = await self._session.call_tool(name, arguments=kwargs or None)
        text = _extract_text(result)
        if text is None or not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}


def _extract_text(result: Any) -> Optional[str]:
    content = getattr(result, "content", None)
    if not content:
        return None
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict):
            parts.append(str(block.get("text", block)))
        else:
            parts.append(getattr(block, "text", str(block)))
    return "\n".join(parts)


@asynccontextmanager
async def open_session():
    """起一个 server 子进程并建立 ClientSession;退出时关闭并回收子进程。"""
    params = StdioServerParameters(
        command=AD_MCP_PYTHON,
        args=[AD_MCP_SERVER],
        env=_server_env(),
        cwd=os.path.dirname(AD_MCP_SERVER) or None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPSession(session)


def call_sync(name: str, **kwargs: Any) -> Dict[str, Any]:
    """同步便捷入口:起一个独立子进程,单次调用后即退出(无持久会话)。"""
    async def _run() -> Dict[str, Any]:
        async with open_session() as mcp:
            return await mcp.call(name, **kwargs)

    with _MCP_LOCK:
        return asyncio.run(_run())


def call_tool(
    name: str,
    *,
    codes: Optional[Union[str, Iterable[str]]] = None,
    begin_date: Any = None,
    end_date: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """同步桥(MCP 工具 → Toolkit @tool 用):schema 改写 + 全局锁串行。

    与 call_sync 的区别是把项目惯例参数(code 逗号串/序列、yyyy-mm-dd 日期)
    改写为 MCP 工具要求的 8 位 int 日期 + list 代码后转发给 call_sync。
    """
    args = dict(kwargs)
    if codes is not None:
        args["code_list"] = to_code_list(codes)
    if begin_date is not None:
        args["begin_date"] = to_int_date(begin_date)
    if end_date is not None:
        args["end_date"] = to_int_date(end_date)
    return call_sync(name, **args)


def format_result(result: Any, max_rows: int = 80, max_cell: int = 120) -> str:
    """把 MCP 返回的解析结果(dict)格式化成 LLM 可读的多行文本。

    MCP 工具一般返回 {code: [行...]} 或 {code: {...}};非 JSON 时 {"raw": text}。
    """
    if not result:
        return "(空)"
    if isinstance(result, dict) and set(result) == {"raw"}:
        return str(result["raw"])[:8000]

    lines: List[str] = []
    items = result.items() if isinstance(result, dict) else [("", result)]
    for code, payload in items:
        if isinstance(payload, list):
            if not payload:
                lines.append(f"[{code}] 无数据")
                continue
            header = list(payload[0].keys()) if isinstance(payload[0], dict) else None
            lines.append(f"[{code}] 共 {len(payload)} 行")
            if header:
                lines.append("  " + " | ".join(header))
                for row in payload[:max_rows]:
                    lines.append(
                        "  " + " | ".join(str(row.get(k, ""))[:max_cell] for k in header)
                    )
                if len(payload) > max_rows:
                    lines.append(f"  ... 其余 {len(payload) - max_rows} 行省略")
            else:
                for row in payload[:max_rows]:
                    lines.append("  " + str(row)[:max_cell])
        elif isinstance(payload, dict):
            lines.append(f"[{code}]")
            for k, v in payload.items():
                lines.append(f"  {k}: {str(v)[:max_cell]}")
        else:
            lines.append(f"[{code}] {payload}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 阶段 1 验收目标:行情 / 三表 / 快照 三类薄封装(schema 改写在这里落地)
# ---------------------------------------------------------------------------

async def get_kline(
    mcp: MCPSession,
    codes: Union[str, Iterable[str]],
    begin_date: Any,
    end_date: Any,
    period: str = "day",
    **kwargs: Any,
) -> Dict[str, Any]:
    return await mcp.call(
        "mcp_kline",
        code_list=to_code_list(codes),
        begin_date=to_int_date(begin_date),
        end_date=to_int_date(end_date),
        period=period,
        **kwargs,
    )


async def get_snapshot(
    mcp: MCPSession, codes: Union[str, Iterable[str]], **kwargs: Any
) -> Dict[str, Any]:
    return await mcp.call("mcp_snapshot", code_list=to_code_list(codes), **kwargs)


async def get_income(
    mcp: MCPSession,
    codes: Union[str, Iterable[str]],
    begin_date: Any = None,
    end_date: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return await mcp.call(
        "mcp_income",
        code_list=to_code_list(codes),
        begin_date=to_int_date(begin_date),
        end_date=to_int_date(end_date),
        **kwargs,
    )


async def get_balance_sheet(
    mcp: MCPSession,
    codes: Union[str, Iterable[str]],
    begin_date: Any = None,
    end_date: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return await mcp.call(
        "mcp_balance_sheet",
        code_list=to_code_list(codes),
        begin_date=to_int_date(begin_date),
        end_date=to_int_date(end_date),
        **kwargs,
    )


async def get_cash_flow(
    mcp: MCPSession,
    codes: Union[str, Iterable[str]],
    begin_date: Any = None,
    end_date: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return await mcp.call(
        "mcp_cash_flow",
        code_list=to_code_list(codes),
        begin_date=to_int_date(begin_date),
        end_date=to_int_date(end_date),
        **kwargs,
    )


async def get_equity_structure(
    mcp: MCPSession,
    codes: Union[str, Iterable[str]],
    begin_date: Any = None,
    end_date: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """股本结构(总股本等),返回 {code: [行]} 或 {"success":..,"data":{code:[行]}}。"""
    return await mcp.call(
        "mcp_equity_structure",
        code_list=to_code_list(codes),
        begin_date=to_int_date(begin_date),
        end_date=to_int_date(end_date),
        **kwargs,
    )


async def get_dividend(
    mcp: MCPSession,
    codes: Union[str, Iterable[str]],
    begin_date: Any = None,
    end_date: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """分红数据(每股派息/基准股本/派息日),返回 {code: [行]}。"""
    return await mcp.call(
        "mcp_dividend",
        code_list=to_code_list(codes),
        begin_date=to_int_date(begin_date),
        end_date=to_int_date(end_date),
        **kwargs,
    )


async def get_stock_basic(
    mcp: MCPSession,
    codes: Union[str, Iterable[str]],
    summary_only: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """证券基础数据(沪深北全股票含已退市):证券简称/中文名/上市日期/板块。返回 {success,count,data:[行]}。"""
    return await mcp.call(
        "mcp_stock_basic",
        code_list=to_code_list(codes),
        summary_only=summary_only,
        **kwargs,
    )