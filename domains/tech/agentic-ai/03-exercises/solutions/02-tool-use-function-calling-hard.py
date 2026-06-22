"""
Solutions -- Day 2 (HARD): Tool Use & Function Calling

Contains solutions for:
  - Hard Ex 1: Dynamic tool discovery -- an agent that learns its tools at
               runtime (discover -> learn -> execute), like MCP servers do
  - Hard Ex 2: Tool composition engine -- declarative "recipes" that chain
               atomic tools and appear as one normal tool in the registry

All LLM decisions are mocked (scripted, deterministic) so the file runs
offline with no API key -- same convention as the other solution files.
The tools are real: the SQLite database and the calculator actually execute.

Run:  python 03-exercises/solutions/02-tool-use-function-calling-hard.py
Each solution is self-contained.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

# ==========================================================================
# SHARED -- Atomic tools backed by real implementations
# ==========================================================================

def _make_db() -> sqlite3.Cursor:
    """In-memory products DB -- same dataset as the Day 2 medium solutions."""
    db = sqlite3.connect(":memory:")
    cur = db.cursor()
    cur.executescript("""
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER);
        INSERT INTO products VALUES (1, 'Laptop Pro 16', 'electronics', 1299.99, 45);
        INSERT INTO products VALUES (2, 'Wireless Mouse', 'electronics', 29.99, 200);
        INSERT INTO products VALUES (3, 'Python Cookbook', 'books', 49.99, 120);
        INSERT INTO products VALUES (4, 'Standing Desk', 'furniture', 599.00, 30);
        INSERT INTO products VALUES (5, 'Mechanical Keyboard', 'electronics', 149.99, 85);
    """)
    db.commit()
    return cur


_CUR = _make_db()


def impl_calculator(expression: str) -> str:
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%]+$', expression):
        raise ValueError(f"Unsafe expression: {expression!r}")
    return str(eval(expression))


def impl_database_query(query: str) -> str:
    q = query.strip().upper()
    if not q.startswith("SELECT"):
        raise PermissionError("Only SELECT queries allowed")
    _CUR.execute(query.strip())
    rows = _CUR.fetchall()
    cols = [d[0] for d in _CUR.description] if _CUR.description else []
    return json.dumps([dict(zip(cols, r)) for r in rows], default=str)


def impl_search_web(query: str) -> str:
    return json.dumps([{"title": f"Result for: {query}",
                        "snippet": f"Mock snippet about '{query}'."}])


def impl_analyze_text(text: str, analysis_type: str = "stats") -> str:
    """Tiny text analyzer -- gives the composition engine a 3rd step to chain."""
    if analysis_type == "stats":
        return json.dumps({"chars": len(text), "words": len(text.split()),
                           "preview": text[:60]})
    raise ValueError(f"Unknown analysis_type: {analysis_type!r}")


# ==========================================================================
# HARD EXERCISE 1 -- Dynamic tool discovery
# ==========================================================================
#
# The agent starts BLIND: its system prompt names zero domain tools. It only
# has 3 meta-tools (discover_tools, get_tool_schema, call_tool) that hit a
# simulated Tool Server API. This is exactly how MCP works: clients discover
# tools at runtime instead of hardcoding them.

class ToolServer:
    """
    Simulated remote tool API with 3 endpoints:
      GET  /tools                -> list_tools()
      GET  /tools/{name}/schema  -> get_schema(name)
      POST /tools/{name}/execute -> execute(name, params)
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}

    def register(self, name: str, short_description: str, schema: dict,
                 fn: Callable[..., str]) -> None:
        self._tools[name] = {"short": short_description, "schema": schema, "fn": fn}

    # GET /tools -- names + short descriptions only (schemas cost tokens,
    # so a real server returns them lazily, one tool at a time)
    def list_tools(self) -> list[dict]:
        return [{"name": n, "description": t["short"]} for n, t in self._tools.items()]

    # GET /tools/{name}/schema
    def get_schema(self, name: str) -> dict:
        if name not in self._tools:
            return {"error": f"Unknown tool '{name}'. Available: {list(self._tools)}"}
        return {"name": name, "description": self._tools[name]["short"],
                "parameters": self._tools[name]["schema"]}

    # POST /tools/{name}/execute
    def execute(self, name: str, params: dict) -> dict:
        if name not in self._tools:
            return {"status": "error", "error": f"Unknown tool '{name}'"}
        try:
            return {"status": "success", "result": self._tools[name]["fn"](**params)}
        except Exception as e:
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def make_tool_server() -> ToolServer:
    server = ToolServer()
    server.register(
        "calculator", "Evaluate a math expression",
        {"type": "object",
         "properties": {"expression": {"type": "string", "description": "Math expression"}},
         "required": ["expression"]},
        impl_calculator)
    server.register(
        "database_query", "Run SQL SELECT on the products table",
        {"type": "object",
         "properties": {"query": {"type": "string",
                                  "description": "SQL SELECT on products(id, name, category, price, stock)"}},
         "required": ["query"]},
        impl_database_query)
    server.register(
        "search_web", "Search the web for current information",
        {"type": "object",
         "properties": {"query": {"type": "string", "description": "Search query"}},
         "required": ["query"]},
        impl_search_web)
    return server


# System prompt for the discovery agent: NO domain tool is named here.
DISCOVERY_SYSTEM_PROMPT = (
    "You are an agent connected to a remote tool server. You do NOT know what "
    "tools are available. Start by calling discover_tools, then call "
    "get_tool_schema for any tool you intend to use, then call_tool to execute it."
)

# Scripted agent decisions -- in production each entry is one LLM call that
# sees the system prompt + the accumulated observations.
DISCOVERY_SCRIPT: list[dict] = [
    {"thought": "I know nothing about my tools. Discover them first.",
     "meta": "discover_tools", "args": {}},
    {"thought": "database_query sounds right to find the most expensive product. "
                "Learn its schema before calling it.",
     "meta": "get_tool_schema", "args": {"name": "database_query"}},
    {"thought": "The schema wants a SQL SELECT. Query the priciest product.",
     "meta": "call_tool",
     "args": {"name": "database_query",
              "params": {"query": "SELECT name, price FROM products ORDER BY price DESC LIMIT 1"}}},
    {"thought": "Now I need 15% tax. Learn the calculator schema first.",
     "meta": "get_tool_schema", "args": {"name": "calculator"}},
    {"thought": "The calculator takes an expression. Compute the tax.",
     "meta": "call_tool",
     "args": {"name": "calculator", "params": {"expression": "1299.99 * 0.15"}}},
    {"thought": "I have both numbers. Answer.",
     "meta": "finish",
     "args": {"answer": "The most expensive product is Laptop Pro 16 at $1,299.99. "
                        "15% tax = $195.00 (1299.99 * 0.15 = 194.9985, rounded)."}},
]

# Phase label per meta-tool: makes the discover -> learn -> execute
# trajectory visible in the trace.
PHASE_OF = {"discover_tools": "DISCOVER", "get_tool_schema": "LEARN",
            "call_tool": "EXECUTE", "finish": "FINISH"}


class DiscoveryAgent:
    """Agent whose ONLY hardcoded tools are the 3 meta-tools."""

    def __init__(self, server: ToolServer):
        self.server = server
        self.trace: list[dict] = []
        # Meta-tool dispatch -- the agent's whole world. Adding a new domain
        # tool to the server requires ZERO change here (criterion 7).
        self._meta: dict[str, Callable[..., Any]] = {
            "discover_tools": lambda: self.server.list_tools(),
            "get_tool_schema": lambda name: self.server.get_schema(name),
            "call_tool": lambda name, params: self.server.execute(name, params),
        }

    def run(self, task: str, script: list[dict], verbose: bool = True) -> str:
        if verbose:
            print(f"\n  System prompt: {DISCOVERY_SYSTEM_PROMPT[:80]}...")
            print(f"  Task: {task}\n")
        answer = ""
        for i, step in enumerate(script):
            meta, args = step["meta"], step["args"]
            phase = PHASE_OF[meta]
            if meta == "finish":
                answer = args["answer"]
                self.trace.append({"step": i + 1, "phase": phase, "meta": meta})
                if verbose:
                    print(f"  [{phase:8s}] Step {i + 1}: {answer}")
                break
            observation = self._meta[meta](**args)
            self.trace.append({"step": i + 1, "phase": phase, "meta": meta,
                               "args": args, "observation": observation})
            if verbose:
                print(f"  [{phase:8s}] Step {i + 1}: {step['thought']}")
                print(f"             {meta}({json.dumps(args)[:70]})")
                print(f"             -> {json.dumps(observation)[:90]}")
        return answer


def hard_ex1_dynamic_discovery():
    """
    Solution: Tool Server (3 simulated endpoints) + Discovery Agent that
    resolves a task in 3 phases: discover -> learn -> execute.
    """
    print("\n" + "=" * 60)
    print("  Hard Ex 1 -- Dynamic Tool Discovery")
    print("=" * 60)

    server = make_tool_server()
    agent = DiscoveryAgent(server)
    answer = agent.run("Find the most expensive product and calculate 15% tax on it.",
                       DISCOVERY_SCRIPT)

    # --- Verify the success criteria -------------------------------------
    # 1. The system prompt names NO server tool (the agent starts blind)
    for tool in ("calculator", "database_query", "search_web"):
        assert tool not in DISCOVERY_SYSTEM_PROMPT, f"Prompt leaks tool '{tool}'"
    # 2. The trajectory shows the 3 phases, in order
    phases = [t["phase"] for t in agent.trace]
    assert phases == ["DISCOVER", "LEARN", "EXECUTE", "LEARN", "EXECUTE", "FINISH"], phases
    # 3. discover_tools really listed the server's tools
    discovered = [t["name"] for t in agent.trace[0]["observation"]]
    assert discovered == ["calculator", "database_query", "search_web"]
    # 4. Schemas were fetched BEFORE the corresponding call_tool
    assert agent.trace[1]["args"]["name"] == agent.trace[2]["args"]["name"]
    assert agent.trace[3]["args"]["name"] == agent.trace[4]["args"]["name"]
    # 5. The real tools produced the real numbers
    assert "Laptop Pro 16" in json.dumps(agent.trace[2]["observation"])
    assert "194.9985" in json.dumps(agent.trace[4]["observation"])
    assert "Laptop Pro 16" in answer and "195.00" in answer

    # --- Generalizability: add a tool server-side, agent code unchanged ---
    print("\n  --- Generalizability test: new server tool, same agent ---")
    server.register(
        "send_email", "Send an email to a recipient",
        {"type": "object", "properties": {"to": {"type": "string"},
                                          "body": {"type": "string"}},
         "required": ["to", "body"]},
        lambda to, body: f"Email sent to {to} ({len(body)} chars)")
    rediscovered = [t["name"] for t in server.list_tools()]
    print(f"  discover_tools now returns: {rediscovered}")
    assert "send_email" in rediscovered, "New tool must be discoverable"
    schema = server.get_schema("send_email")
    assert schema["parameters"]["required"] == ["to", "body"]
    print("  Agent learned the new tool with the SAME 3 meta-tools -- zero code change.")

    print("\n  PASS -- discover -> learn -> execute, blind prompt, extensible server.\n")


# ==========================================================================
# HARD EXERCISE 2 -- Tool composition engine
# ==========================================================================
#
# Recipes are DECLARATIVE: a list of step dicts (tool + params templates +
# what to extract from each result). The engine resolves {placeholders},
# threads data between steps, handles per-step errors, and registers the
# whole recipe as ONE normal tool -- callers can't tell it's composite.
# The engine lives between the sentinels and is asserted < 200 lines.

# --- ENGINE START ----------------------------------------------------------

@dataclass
class RecipeStep:
    """One declarative step of a recipe.

    params_template: tool params with {placeholders} resolved from context.
    extract: {context_var: path} -- pulls values out of the step result.
             Path syntax: "" (whole result), "key", "[0].key", "key[2]".
    continue_on_error: if True, a failure here does not abort the recipe.
    """
    tool: str
    params_template: dict[str, str]
    extract: dict[str, str] = field(default_factory=dict)
    continue_on_error: bool = False


class CompositionEngine:
    """Composes atomic tools into recipes exposed as normal tools."""

    def __init__(self, tools: dict[str, Callable[..., str]]):
        self._tools = tools            # Atomic implementations
        self._registry: dict[str, dict] = {}  # name -> {description, schema, fn}
        # Atomic tools are also in the registry, so recipes and atoms are
        # indistinguishable to a caller (criterion 3).
        for name, fn in tools.items():
            self._registry[name] = {"description": f"(atomic) {name}",
                                    "schema": {}, "fn": fn, "composite": False}

    # --- Public registry API (what an agent would see) --------------------

    def list_tools(self) -> list[str]:
        return list(self._registry)

    def describe(self, name: str) -> str:
        return self._registry[name]["description"]

    def call(self, name: str, **params: Any) -> str:
        return self._registry[name]["fn"](**params)

    # --- Recipe registration ----------------------------------------------

    def register_recipe(self, name: str, description: str,
                        steps: list[RecipeStep], input_schema: dict) -> None:
        def recipe_fn(**inputs: Any) -> str:
            return self._run_recipe(name, steps, inputs)
        self._registry[name] = {"description": description, "schema": input_schema,
                                "fn": recipe_fn, "composite": True}

    # --- Execution machinery ------------------------------------------------

    @staticmethod
    def _resolve(template: str, context: dict[str, Any]) -> str:
        """Replace {var} placeholders. Unknown vars fail loudly (better than
        silently passing '{var}' to a tool)."""
        def sub(match: re.Match) -> str:
            key = match.group(1)
            if key not in context:
                raise KeyError(f"Template variable '{{{key}}}' not in context "
                               f"(have: {sorted(context)})")
            return str(context[key])
        return re.sub(r"\{(\w+)\}", sub, template)

    @staticmethod
    def _extract_path(result: str, path: str) -> Any:
        """Extract a value from a (possibly JSON) result using a mini-path:
        '' -> whole result, 'key' -> dict key, '[i]' -> list index."""
        if path == "":
            return result
        try:
            value: Any = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            value = result
        for token in re.findall(r"\[(\d+)\]|(\w+)", path):
            idx, key = token
            value = value[int(idx)] if idx else value[key]
        return value

    def _run_recipe(self, name: str, steps: list[RecipeStep],
                    inputs: dict[str, Any]) -> str:
        context: dict[str, Any] = dict(inputs)  # Inputs seed the context
        last_result = ""
        for n, step in enumerate(steps, start=1):
            try:
                params = {k: self._resolve(v, context)
                          for k, v in step.params_template.items()}
                last_result = self._tools[step.tool](**params)
                # Thread data to later steps via extraction into the context
                for var, path in step.extract.items():
                    context[var] = self._extract_path(last_result, path)
                context[f"step{n}_result"] = last_result
            except Exception as e:
                msg = f"Step {n} ({step.tool}) failed: {type(e).__name__}: {e}"
                if step.continue_on_error:
                    context[f"step{n}_result"] = f"(skipped: {msg})"
                    print(f"      [engine] {msg} -- continue_on_error=True, skipping")
                    continue
                # Fail the WHOLE recipe with a message naming the bad step
                raise RuntimeError(f"Recipe '{name}' aborted. {msg}") from e
        return last_result

# --- ENGINE END ------------------------------------------------------------


def hard_ex2_composition_engine():
    """
    Solution: declarative recipes over atomic tools, with template-based
    data passing, per-step error reporting, and continue_on_error.
    """
    print("\n" + "=" * 60)
    print("  Hard Ex 2 -- Tool Composition Engine")
    print("=" * 60)

    engine = CompositionEngine({
        "calculator": impl_calculator,
        "database_query": impl_database_query,
        "search_web": impl_search_web,
        "analyze_text": impl_analyze_text,
    })

    # --- Recipe 1: 3 steps, data threaded between them ---------------------
    engine.register_recipe(
        name="get_product_report",
        description="Generate a complete report for a product category",
        steps=[
            RecipeStep(
                tool="database_query",
                params_template={"query": "SELECT COUNT(*) AS count, SUM(price) AS total_price "
                                          "FROM products WHERE category = '{category}'"},
                # Pull aggregates out of the JSON result into the context
                extract={"count": "[0].count", "total_price": "[0].total_price"}),
            RecipeStep(
                tool="calculator",
                params_template={"expression": "{total_price} / {count}"},
                extract={"avg_price": ""}),  # "" = whole result
            RecipeStep(
                tool="analyze_text",
                params_template={"text": "Category {category}: {count} products, "
                                         "avg price {avg_price} EUR",
                                 "analysis_type": "stats"}),
        ],
        input_schema={"type": "object",
                      "properties": {"category": {"type": "string"}},
                      "required": ["category"]})

    # --- Recipe 2: step 2 fails (division by zero -> clear error) ----------
    engine.register_recipe(
        name="audit_pricing",
        description="Audit pricing consistency (intentionally broken at step 2)",
        steps=[
            RecipeStep(tool="database_query",
                       params_template={"query": "SELECT COUNT(*) AS count FROM products "
                                                 "WHERE category = 'toys'"},  # 0 rows match
                       extract={"count": "[0].count"}),
            RecipeStep(tool="calculator",
                       params_template={"expression": "100 / {count}"}),  # 100 / 0 -> boom
            RecipeStep(tool="analyze_text",
                       params_template={"text": "never reached", "analysis_type": "stats"}),
        ],
        input_schema={"type": "object", "properties": {}})

    # --- Recipe 3: non-critical step fails, continue_on_error skips it -----
    engine.register_recipe(
        name="resilient_report",
        description="Report that tolerates a failing enrichment step",
        steps=[
            RecipeStep(tool="database_query",
                       params_template={"query": "SELECT name, price FROM products "
                                                 "ORDER BY price DESC LIMIT 1"},
                       extract={"top_name": "[0].name", "top_price": "[0].price"}),
            RecipeStep(tool="database_query",
                       params_template={"query": "SELECT margin FROM products LIMIT 1"},  # No such column
                       continue_on_error=True),  # Enrichment only -- not critical
            RecipeStep(tool="analyze_text",
                       params_template={"text": "Top product: {top_name} at {top_price} EUR",
                                        "analysis_type": "stats"}),
        ],
        input_schema={"type": "object", "properties": {}})

    # --- Criterion: recipes look like normal tools in the registry ---------
    print(f"\n  Registry contents: {engine.list_tools()}")
    assert "get_product_report" in engine.list_tools()
    assert "calculator" in engine.list_tools()  # Atoms and recipes side by side
    print(f"  describe('get_product_report'): {engine.describe('get_product_report')}")

    # --- Test 1: happy path -------------------------------------------------
    print("\n  --- Test 1: get_product_report(category='electronics') ---")
    result = engine.call("get_product_report", category="electronics")
    parsed = json.loads(result)
    print(f"    Result: {result}")
    # 3 electronics: 1299.99 + 29.99 + 149.99 = 1479.97 -> avg 493.32...
    assert "3 products" in parsed["preview"], parsed
    assert "493.32" in parsed["preview"], parsed

    # --- Test 2: failing step reports WHICH step broke ---------------------
    print("\n  --- Test 2: audit_pricing (step 2 divides by zero) ---")
    try:
        engine.call("audit_pricing")
        raise AssertionError("audit_pricing should have failed")
    except RuntimeError as e:
        print(f"    Caught: {e}")
        assert "Step 2 (calculator) failed" in str(e), str(e)

    # --- Test 3: continue_on_error skips the broken step --------------------
    print("\n  --- Test 3: resilient_report (step 2 fails, non-critical) ---")
    result3 = engine.call("resilient_report")
    parsed3 = json.loads(result3)
    print(f"    Result: {result3}")
    assert "Laptop Pro 16" in parsed3["preview"], parsed3

    # --- Engine size: < 200 lines (complexity budget) -----------------------
    from pathlib import Path
    source = Path(__file__).read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(source) if "ENGINE START" in l)
    end = next(i for i, l in enumerate(source) if "ENGINE END" in l)
    engine_lines = end - start - 1
    print(f"\n  Engine size: {engine_lines} lines (< 200 required)")
    assert engine_lines < 200, f"Engine too big: {engine_lines} lines"

    print("\n  PASS -- 3 declarative recipes: success, step-2 failure, resilient skip.\n")


# ==========================================================================
# MAIN -- Run both hard solutions
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 2 HARD Solutions -- Tool Use & Function Calling")
    print("#" * 60)

    hard_ex1_dynamic_discovery()
    hard_ex2_composition_engine()

    print("\n" + "#" * 60)
    print("  All hard solutions executed successfully.")
    print("#" * 60 + "\n")
