本目录集中存放各 Agent 的提示词模板，由 `tradingagents.config.prompt_manager` 加载。

路径：`tradingagents/prompts/`（随包分发）。

## 目录结构

```
tradingagents/prompts/
├── analysts/          # 分析师（基本面/市场/新闻/社媒/中国市场）
├── researchers/       # 看涨/看跌研究员
├── managers/          # 研究经理、风控经理
├── risk_mgmt/         # 激进/保守/中性风险辩论
├── trader/            # 交易员
├── graph/             # 反思模块
└── shared/            # 跨 Agent 共享模板
```

## 用法

```python
from tradingagents.config.prompt_manager import get_prompt

# 仅取模板（保留 {tool_names} 等占位符）
tpl = get_prompt("analysts/fundamentals", "system_prompt")

# 渲染变量（未传入的占位符会原样保留）
text = get_prompt(
    "researchers/bull",
    "prompt",
    company_name="贵州茅台",
    ticker="600519",
)
```

## 约定

- 静态角色与指令写在 YAML；`ticker`、报告内容等运行时变量由代码注入。
- 与 `ChatPromptTemplate.partial` 混用时，可先 `render` 部分变量，剩余 `{var}` 留给 LangChain。
- 修改提示词后无需改业务逻辑；开发时可调用 `prompt_manager.clear_cache()` 热加载。
