"""
Solutions -- Day 2: Tool Use & Function Calling

Contains solutions for:
  - Easy Ex 1: Translate tool with enum params
  - Easy Ex 2: Convert tools between OpenAI and Anthropic formats
  - Easy Ex 3: Actionable error messages
  - Medium Ex 1: Function calling agent with self-correction
  - Medium Ex 2: Structured output -- invoice extraction
  - Medium Ex 3: Tool middleware (logging, timing, rate limiting, retry)

Run:  python 03-exercises/solutions/02-tool-use-function-calling.py
Each solution is self-contained.
"""

import json
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

# ==========================================================================
# SHARED -- Minimal tool registry (same as 02-code but simplified for exercises)
# ==========================================================================

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_anthropic_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}'. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, params: dict) -> dict:
        try:
            tool = self.get(name)
        except KeyError as e:
            return {"status": "error", "error": str(e)}
        required = tool.parameters.get("required", [])
        missing = [p for p in required if p not in params]
        if missing:
            return {"status": "error", "error": f"Missing required parameters: {missing}"}
        try:
            result = tool.fn(**params)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": f"{tool.name} failed: {type(e).__name__}: {e}"}


def create_base_registry() -> ToolRegistry:
    """Create a registry with the basic tools for testing."""
    registry = ToolRegistry()

    registry.register(Tool(
        name="calculator",
        description="Evaluate a math expression. Returns the numeric result.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        },
        fn=lambda expression: str(eval(expression)) if re.match(r'^[\d\s\+\-\*\/\.\(\)\%\^]+$', expression) else (_ for _ in ()).throw(ValueError(f"Unsafe expression: {expression}")),
    ))

    registry.register(Tool(
        name="search_web",
        description="Search the web for current information. Returns {title, url, snippet} results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 3}
            },
            "required": ["query"]
        },
        fn=lambda query, max_results=3: json.dumps([{"title": f"Result for: {query}", "url": "https://example.com", "snippet": f"Mock result for '{query}'."}]),
    ))

    # Setup in-memory database for database_query tool
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

    def _db_query(query: str) -> str:
        q = query.strip().upper()
        if not q.startswith("SELECT"):
            raise PermissionError("Only SELECT queries allowed")
        for kw in ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER"]:
            if re.search(rf'\b{kw}\b', q):
                raise PermissionError(f"Forbidden keyword: {kw}")
        cur.execute(query.strip())
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return json.dumps([dict(zip(cols, r)) for r in rows], default=str)

    registry.register(Tool(
        name="database_query",
        description="Execute SQL SELECT on products table (id, name, category, price, stock). Only SELECT allowed.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query"}
            },
            "required": ["query"]
        },
        fn=_db_query,
    ))

    return registry


# ==========================================================================
# EASY EXERCISE 1 -- Translate tool with enum parameters
# ==========================================================================

def easy_ex1_translate_tool():
    """
    Solution: define a tool with enum params and a default value.
    Key insight: enums constrain the LLM's output space -- fewer errors.
    """
    print("\n" + "=" * 60)
    print("  Easy Ex 1 -- Translate Text Tool")
    print("=" * 60)

    registry = create_base_registry()

    # Mock translations -- just prefix with language tag for the demo
    mock_translations = {
        "en": lambda text, formality: f"[EN-{formality}] {text}",
        "fr": lambda text, formality: f"[FR-{formality}] {text}" if formality == "formal" else f"[FR-informal] {text.lower()}",
        "es": lambda text, formality: f"[ES-{formality}] {text}",
        "de": lambda text, formality: f"[DE-{formality}] {text}",
        "pt": lambda text, formality: f"[PT-{formality}] {text}",
    }

    def translate_text(text: str, target_language: str, formality: str = "formal") -> str:
        """Mock translation tool."""
        if target_language not in mock_translations:
            raise ValueError(f"Unsupported language: '{target_language}'. Supported: {list(mock_translations.keys())}")
        if formality not in ("formal", "informal"):
            raise ValueError(f"Invalid formality: '{formality}'. Must be 'formal' or 'informal'")
        return mock_translations[target_language](text, formality)

    # Register with proper schema
    registry.register(Tool(
        name="translate_text",
        description=(
            "Translate text to a target language. Returns the translated text. "
            "Use when the user asks to translate content or needs text in another language. "
            "Do NOT use for language detection or grammar correction."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to translate"
                },
                "target_language": {
                    "type": "string",
                    "description": "Target language code",
                    "enum": ["en", "fr", "es", "de", "pt"]
                },
                "formality": {
                    "type": "string",
                    "description": "Formality level of the translation",
                    "enum": ["formal", "informal"],
                    "default": "formal"
                }
            },
            "required": ["text", "target_language"]
        },
        fn=translate_text,
    ))

    # Test cases
    print("\n  Test 1 -- Success (formal FR):")
    r = registry.execute("translate_text", {"text": "Hello, how are you?", "target_language": "fr"})
    print(f"    {r}")

    print("\n  Test 2 -- Success (informal FR):")
    r = registry.execute("translate_text", {"text": "Hello, how are you?", "target_language": "fr", "formality": "informal"})
    print(f"    {r}")

    print("\n  Test 3 -- Missing required param:")
    r = registry.execute("translate_text", {"target_language": "es"})
    print(f"    {r}")

    print("\n  Test 4 -- Invalid language:")
    r = registry.execute("translate_text", {"text": "Hello", "target_language": "zz"})
    print(f"    {r}")

    print(f"\n  PASS -- Tool with enums, defaults, and validation.\n")


# ==========================================================================
# EASY EXERCISE 2 -- Convert between OpenAI and Anthropic formats
# ==========================================================================

def easy_ex2_format_conversion():
    """
    Solution: convert tools between OpenAI and Anthropic formats.
    Key insight: the two formats carry the same info, just structured differently.
    """
    print("\n" + "=" * 60)
    print("  Easy Ex 2 -- OpenAI <-> Anthropic Format Conversion")
    print("=" * 60)

    def openai_to_anthropic(tool: dict) -> dict:
        """
        Convert OpenAI function calling format to Anthropic tool use format.

        OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Anthropic: {"name": ..., "description": ..., "input_schema": ...}
        """
        fn = tool["function"]
        return {
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        }

    def anthropic_to_openai(tool: dict) -> dict:
        """
        Convert Anthropic tool use format to OpenAI function calling format.

        Anthropic: {"name": ..., "description": ..., "input_schema": ...}
        OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
        }

    registry = create_base_registry()

    # Convert 3 tools and verify round-trip
    test_tools = ["calculator", "search_web", "database_query"]

    for name in test_tools:
        tool = registry.get(name)
        openai_fmt = tool.to_openai_format()
        anthropic_fmt = openai_to_anthropic(openai_fmt)
        round_trip = anthropic_to_openai(anthropic_fmt)

        print(f"\n  Tool: {name}")
        print(f"    OpenAI format keys:   {list(openai_fmt.keys())} + function.{list(openai_fmt['function'].keys())}")
        print(f"    Anthropic format keys: {list(anthropic_fmt.keys())}")
        print(f"    Round-trip matches:    {openai_fmt == round_trip}")

        # Verify specific fields
        assert anthropic_fmt["name"] == name, f"Name mismatch for {name}"
        assert "input_schema" in anthropic_fmt, f"Missing input_schema for {name}"
        assert "type" not in anthropic_fmt, f"Anthropic format should not have 'type' at root"
        assert round_trip == openai_fmt, f"Round-trip failed for {name}"

    print(f"\n  PASS -- All 3 tools converted losslessly between formats.\n")


# ==========================================================================
# EASY EXERCISE 3 -- Actionable error messages
# ==========================================================================

def easy_ex3_actionable_errors():
    """
    Solution: wrap tool execution with structured, actionable error messages.
    Key insight: the LLM needs CONTEXT to self-correct. "Error" is useless.
    "Column 'cost' not found. Available: id, name, price" is actionable.
    """
    print("\n" + "=" * 60)
    print("  Easy Ex 3 -- Actionable Error Messages")
    print("=" * 60)

    registry = create_base_registry()

    def execute_with_feedback(registry: ToolRegistry, tool_name: str, params: dict) -> str:
        """
        Execute a tool and return either the result or a structured error message
        that the LLM can use to self-correct.
        """
        result = registry.execute(tool_name, params)

        if result["status"] == "success":
            return result["result"]

        # Determine suggestion based on error type
        error_msg = result["error"]
        suggestion = "Retry with different parameters or try a different approach."

        if "Unknown tool" in error_msg:
            suggestion = "Check the available tools listed below and pick the correct one."
        elif "PermissionError" in error_msg or "not allowed" in error_msg.lower():
            suggestion = "This operation is not allowed. Try a read-only alternative (SELECT only for SQL)."
        elif "Missing required" in error_msg:
            suggestion = "Check the tool schema and provide all required parameters."
        elif "ValueError" in error_msg:
            suggestion = "Check your parameter values against the tool schema (types, ranges, formats)."
        elif "FileNotFoundError" in error_msg:
            suggestion = "Verify the file path exists. Use list_directory to check available files."
        elif "no such column" in error_msg.lower() or "no column" in error_msg.lower():
            suggestion = "Check column names. Use 'PRAGMA table_info(table_name)' or see the tool description for available columns."

        return (
            f"TOOL_ERROR: {tool_name} failed.\n"
            f"REASON: {error_msg}\n"
            f"SUGGESTION: {suggestion}\n"
            f"AVAILABLE_TOOLS: {registry.list_tools()}"
        )

    # Test 5 error scenarios
    test_cases = [
        ("unknown_tool", {"x": 1}, "Unknown tool"),
        ("database_query", {"query": "DELETE FROM products"}, "Permission denied"),
        ("calculator", {}, "Missing required param"),
        ("calculator", {"expression": "import os"}, "Unsafe input"),
        ("database_query", {"query": "SELECT cost FROM products"}, "Wrong column name"),
    ]

    for tool_name, params, description in test_cases:
        print(f"\n  Scenario: {description}")
        print(f"  Call: {tool_name}({json.dumps(params)})")
        feedback = execute_with_feedback(registry, tool_name, params)
        # Show first 3 lines of the feedback
        lines = feedback.split("\n")
        for line in lines[:4]:
            print(f"    {line}")

    # Test success case
    print(f"\n  Scenario: Success case")
    feedback = execute_with_feedback(registry, "calculator", {"expression": "2 + 2"})
    print(f"  Call: calculator({{'expression': '2 + 2'}})")
    print(f"    Result: {feedback}")

    print(f"\n  PASS -- 5 error types with actionable suggestions + 1 success case.\n")


# ==========================================================================
# MEDIUM EXERCISE 1 -- Function calling agent with self-correction
# ==========================================================================

def medium_ex1_function_calling_agent():
    """
    Solution: agent using function calling format (not ReAct text parsing)
    with self-correction when a tool fails.
    Key insight: errors are just another message in the conversation.
    """
    print("\n" + "=" * 60)
    print("  Medium Ex 1 -- Function Calling Agent with Self-Correction")
    print("=" * 60)

    registry = create_base_registry()

    # Build the full conversation as the API would see it
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful data analyst. Use tools to answer questions."},
        {"role": "user", "content": "How many products are in each category, and what's the average price?"},
    ]

    # --- Step 1: LLM generates a tool call with a wrong column name ---
    print("\n  Step 1: LLM calls database_query (with error)")

    step1_response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_s1",
            "type": "function",
            "function": {
                "name": "database_query",
                "arguments": json.dumps({"query": "SELECT category, COUNT(*), AVG(cost) FROM products GROUP BY category"})
            }
        }]
    }
    messages.append(step1_response)

    # Execute the tool
    tc = step1_response["tool_calls"][0]
    fn_args = json.loads(tc["function"]["arguments"])
    result = registry.execute(tc["function"]["name"], fn_args)
    print(f"    Tool call: database_query({fn_args})")
    print(f"    Result: {result['status']} -- {result.get('error', result.get('result', ''))[:80]}")

    # Feed error back to LLM
    tool_content = result.get("result", "") if result["status"] == "success" else f"Error: {result['error']}"
    messages.append({
        "role": "tool",
        "tool_call_id": tc["id"],
        "content": tool_content,
    })

    # --- Step 2: LLM self-corrects with the right column name ---
    print("\n  Step 2: LLM retries with corrected query")

    step2_response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_s2",
            "type": "function",
            "function": {
                "name": "database_query",
                "arguments": json.dumps({"query": "SELECT category, COUNT(*) as count, ROUND(AVG(price), 2) as avg_price FROM products GROUP BY category"})
            }
        }]
    }
    messages.append(step2_response)

    tc2 = step2_response["tool_calls"][0]
    fn_args2 = json.loads(tc2["function"]["arguments"])
    result2 = registry.execute(tc2["function"]["name"], fn_args2)
    print(f"    Tool call: database_query({fn_args2})")
    print(f"    Result: {result2['status']}")
    if result2["status"] == "success":
        for row in json.loads(result2["result"]):
            print(f"      {row}")

    messages.append({
        "role": "tool",
        "tool_call_id": tc2["id"],
        "content": result2.get("result", f"Error: {result2.get('error', '')}"),
    })

    # --- Step 3: LLM generates final answer (no tool call) ---
    print("\n  Step 3: LLM generates final answer")

    final_response = {
        "role": "assistant",
        "content": (
            "Here are the products by category:\n"
            "- Books: 1 product, avg price $49.99\n"
            "- Electronics: 3 products, avg price $493.32\n"
            "- Furniture: 1 product, avg price $599.00"
        ),
    }
    messages.append(final_response)
    print(f"    {final_response['content']}")

    # --- Show full message history ---
    print(f"\n --- Full message history ({len(messages)} messages) ---\n")
    for i, msg in enumerate(messages):
        role = msg["role"]
        if msg.get("tool_calls"):
            names = [tc["function"]["name"] for tc in msg["tool_calls"]]
            print(f"    [{i}] {role}: tool_calls -> {names}")
        elif role == "tool":
            content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
            print(f"    [{i}] {role} (id={msg['tool_call_id']}): {content}")
        else:
            content = msg.get("content", "(null)")
            if content and len(content) > 60:
                content = content[:60] + "..."
            print(f"    [{i}] {role}: {content}")

    assert len(messages) == 7, f"Expected 7 messages, got {len(messages)}"
    print(f"\n  Message count: {len(messages)} (system + user + assistant + tool + assistant + tool + assistant)")
    print(f"\n  PASS -- Self-correction via error feedback in function calling format.\n")


# ==========================================================================
# MEDIUM EXERCISE 2 -- Structured output: invoice extraction
# ==========================================================================

def medium_ex2_structured_output():
    """
    Solution: force structured output using tool_choice + fake tool.
    Key insight: tool schemas are the most reliable way to get structured data.
    """
    print("\n" + "=" * 60)
    print("  Medium Ex 2 -- Structured Output: Invoice Extraction")
    print("=" * 60)

    # Define the extraction schema
    invoice_schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string", "description": "Invoice number (e.g. INV-2026-0042)"},
            "date": {"type": "string", "description": "Invoice date in YYYY-MM-DD format"},
            "vendor": {"type": "string", "description": "Vendor name"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "unit_price": {"type": "number"}
                    },
                    "required": ["description", "quantity", "unit_price"]
                },
                "description": "Line items on the invoice"
            },
            "total": {"type": "number", "description": "Total amount"},
            "currency": {
                "type": "string",
                "enum": ["USD", "EUR", "GBP"],
                "description": "Currency code"
            }
        },
        "required": ["invoice_number", "date", "vendor", "items", "total", "currency"]
    }

    # Input text
    invoice_text = """Invoice #INV-2026-0042
Date: April 10, 2026
Vendor: TechCorp Solutions

Items:
- 3x Laptop Pro 16 @ $1,299.99 each
- 10x Wireless Mouse @ $29.99 each
- 1x Standing Desk @ $599.00

Total: $4,798.87 USD"""

    print(f"\n  Input text:\n  {invoice_text}\n")

    # Simulated extraction (what the LLM would return)
    extracted = {
        "invoice_number": "INV-2026-0042",
        "date": "2026-04-10",
        "vendor": "TechCorp Solutions",
        "items": [
            {"description": "Laptop Pro 16", "quantity": 3, "unit_price": 1299.99},
            {"description": "Wireless Mouse", "quantity": 10, "unit_price": 29.99},
            {"description": "Standing Desk", "quantity": 1, "unit_price": 599.00},
        ],
        "total": 4798.87,
        "currency": "USD"
    }

    print(f"  Extracted data:")
    print(f"  {json.dumps(extracted, indent=4)}")

    # --- Validation function ---
    def validate_extraction(data: dict, schema: dict) -> list[str]:
        """
        Validate extracted data against a JSON Schema.
        Returns a list of error messages (empty = valid).
        """
        errors = []
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field_name in required:
            if field_name not in data:
                errors.append(f"Missing required field: '{field_name}'")

        # Check types and enums
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        for field_name, field_schema in props.items():
            if field_name not in data:
                continue  # Already caught by required check

            value = data[field_name]
            expected_type = field_schema.get("type")

            # Type check
            if expected_type and expected_type in type_map:
                py_type = type_map[expected_type]
                if not isinstance(value, py_type):
                    errors.append(
                        f"Wrong type for '{field_name}': expected {expected_type}, "
                        f"got {type(value).__name__}"
                    )

            # Enum check
            if "enum" in field_schema and value not in field_schema["enum"]:
                errors.append(
                    f"Invalid enum value for '{field_name}': '{value}'. "
                    f"Must be one of {field_schema['enum']}"
                )

            # Nested array validation
            if expected_type == "array" and isinstance(value, list) and "items" in field_schema:
                item_schema = field_schema["items"]
                for i, item in enumerate(value):
                    item_errors = validate_extraction(item, item_schema)
                    for err in item_errors:
                        errors.append(f"items[{i}].{err}")

        return errors

    # Test 1: Valid extraction
    print(f"\n --- Validation: valid data ---")
    errors = validate_extraction(extracted, invoice_schema)
    print(f"  Errors: {errors if errors else '(none -- valid!)'}")

    # Test 2: Extraction with errors
    print(f"\n --- Validation: data with errors ---")
    bad_extraction = {
        "invoice_number": "INV-2026-0042",
        "date": "2026-04-10",
        # "vendor" is missing (required)
        "items": [
            {"description": "Laptop", "quantity": "three", "unit_price": 1299.99},  # quantity should be int
        ],
        "total": "4798.87",  # should be number, not string
        "currency": "JPY",   # not in enum
    }

    errors = validate_extraction(bad_extraction, invoice_schema)
    print(f"  Errors found ({len(errors)}):")
    for err in errors:
        print(f"    - {err}")

    # Show how tool_choice would be used
    print(f"\n --- API call (how tool_choice forces extraction) ---")
    print(f'  tool_choice: {{"type": "function", "function": {{"name": "extract_invoice_data"}}}}')
    print(f"  (Forces the LLM to output ONLY the structured data, no free text)")

    print(f"\n  PASS -- Schema validation catches 4 types of errors.\n")


# ==========================================================================
# MEDIUM EXERCISE 3 -- Tool middleware: logging, timing, rate limiting, retry
# ==========================================================================

def medium_ex3_middleware():
    """
    Solution: middleware chain for tool execution.
    Key insight: middlewares separate concerns (logging, auth, rate limiting)
    from tool logic. Same pattern as Express.js middleware or Django middleware.
    """
    print("\n" + "=" * 60)
    print("  Medium Ex 3 -- Tool Middleware (Log, Time, Rate Limit, Retry)")
    print("=" * 60)

    # --- Middleware base class ---
    class ToolMiddleware:
        """Base class. Subclasses override __call__ and invoke next_fn to continue the chain."""
        def __call__(self, tool_name: str, params: dict, next_fn: Callable) -> dict:
            return next_fn(tool_name, params)

    # --- Logging middleware ---
    class LogMiddleware(ToolMiddleware):
        """Log every tool call with timestamp, params, result, and duration."""
        def __init__(self):
            self.logs: list[dict] = []

        def __call__(self, tool_name: str, params: dict, next_fn: Callable) -> dict:
            start = time.time()
            timestamp = datetime.now().isoformat()

            result = next_fn(tool_name, params)

            duration = time.time() - start
            log_entry = {
                "timestamp": timestamp,
                "tool": tool_name,
                "params": params,
                "status": result["status"],
                "duration_ms": round(duration * 1000, 1),
                "result_preview": str(result.get("result", result.get("error", "")))[:80],
            }
            self.logs.append(log_entry)
            return result

    # --- Timing middleware ---
    class TimingMiddleware(ToolMiddleware):
        """Add execution time to the result dict."""
        def __call__(self, tool_name: str, params: dict, next_fn: Callable) -> dict:
            start = time.time()
            result = next_fn(tool_name, params)
            result["duration_ms"] = round((time.time() - start) * 1000, 1)
            return result

    # --- Rate limiting middleware ---
    class RateLimitMiddleware(ToolMiddleware):
        """Limit calls per tool to max_calls per window_seconds."""
        def __init__(self, max_calls: int = 3, window_seconds: float = 60.0):
            self.max_calls = max_calls
            self.window = window_seconds
            self.call_times: dict[str, list[float]] = defaultdict(list)

        def __call__(self, tool_name: str, params: dict, next_fn: Callable) -> dict:
            now = time.time()
            # Remove calls outside the window
            self.call_times[tool_name] = [
                t for t in self.call_times[tool_name] if now - t < self.window
            ]
            # Check limit
            if len(self.call_times[tool_name]) >= self.max_calls:
                return {
                    "status": "error",
                    "error": f"Rate limit exceeded for {tool_name}: {self.max_calls} calls per {self.window}s. Try again later."
                }
            self.call_times[tool_name].append(now)
            return next_fn(tool_name, params)

    # --- Retry middleware ---
    class RetryMiddleware(ToolMiddleware):
        """Retry failed tool calls with exponential backoff."""
        def __init__(self, max_retries: int = 3, base_delay: float = 0.1):
            self.max_retries = max_retries
            self.base_delay = base_delay

        def __call__(self, tool_name: str, params: dict, next_fn: Callable) -> dict:
            for attempt in range(self.max_retries + 1):
                result = next_fn(tool_name, params)
                if result["status"] == "success":
                    if attempt > 0:
                        result["retries"] = attempt
                    return result
                # Don't retry on certain errors (permission, validation)
                error = result.get("error", "")
                if "PermissionError" in error or "Missing required" in error:
                    return result  # Not retryable
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    time.sleep(delay)
            result["retries_exhausted"] = True
            return result

    # --- Middleware-aware registry ---
    class MiddlewareRegistry(ToolRegistry):
        """ToolRegistry that runs calls through a middleware chain."""
        def __init__(self):
            super().__init__()
            self._middlewares: list[ToolMiddleware] = []

        def add_middleware(self, middleware: ToolMiddleware) -> None:
            self._middlewares.append(middleware)
            print(f"  [Middleware] Added: {type(middleware).__name__}")

        def execute(self, name: str, params: dict) -> dict:
            """Execute through the middleware chain."""
            # Build the chain: middleware_0 → middleware_1 → ... → actual execution
            def actual_execute(n: str, p: dict) -> dict:
                return super(MiddlewareRegistry, self).execute(n, p)

            # Wrap from inside out so the first middleware is the outermost
            chain = actual_execute
            for mw in reversed(self._middlewares):
                # Capture the current middleware and chain in the closure
                def make_wrapped(middleware, next_chain):
                    def wrapped(n: str, p: dict) -> dict:
                        return middleware(n, p, next_chain)
                    return wrapped
                chain = make_wrapped(mw, chain)

            return chain(name, params)

    # --- Build the registry with all middlewares ---
    registry = MiddlewareRegistry()

    # Register a simple calculator tool
    registry.register(Tool(
        name="calculator",
        description="Evaluate math expression",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        fn=lambda expression: str(eval(expression)) if re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression) else (_ for _ in ()).throw(ValueError("Unsafe")),
    ))

    # Register a flaky tool that fails sometimes (for retry testing)
    _flaky_call_count = {"n": 0}

    def flaky_tool(query: str) -> str:
        _flaky_call_count["n"] += 1
        if _flaky_call_count["n"] <= 2:
            raise ConnectionError(f"Simulated network error (attempt {_flaky_call_count['n']})")
        return f"Success on attempt {_flaky_call_count['n']}: results for '{query}'"

    registry.register(Tool(
        name="flaky_search",
        description="Search that fails sometimes (for testing retry)",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        fn=flaky_tool,
    ))

    # Add middlewares (order matters: log wraps everything)
    logger = LogMiddleware()
    rate_limiter = RateLimitMiddleware(max_calls=3, window_seconds=60)
    retry = RetryMiddleware(max_retries=3, base_delay=0.05)
    timing = TimingMiddleware()

    print()
    registry.add_middleware(logger)       # Outermost: logs everything
    registry.add_middleware(rate_limiter)  # Then rate limit check
    registry.add_middleware(retry)         # Then retry on failure
    registry.add_middleware(timing)        # Innermost: measures actual execution time

    # --- Test 1: Normal execution ---
    print(f"\n --- Test 1: Normal execution ---")
    r = registry.execute("calculator", {"expression": "42 * 10"})
    print(f"    Result: {r}")

    # --- Test 2: Retry on flaky tool ---
    print(f"\n --- Test 2: Retry on flaky tool (fails 2x, succeeds on 3rd) ---")
    _flaky_call_count["n"] = 0
    start = time.time()
    r = registry.execute("flaky_search", {"query": "test"})
    elapsed = time.time() - start
    print(f"    Result: {r}")
    print(f"    Total time (includes retry delays): {elapsed:.3f}s")

    # --- Test 3: Rate limiting ---
    print(f"\n --- Test 3: Rate limiting (max 3 calls) ---")
    for i in range(4):
        r = registry.execute("calculator", {"expression": f"{i} + 1"})
        status = r["status"]
        detail = r.get("result", r.get("error", ""))[:50]
        print(f"    Call {i + 1}: {status} -- {detail}")

    # --- Show logs ---
    print(f"\n --- Middleware logs ({len(logger.logs)} entries) ---")
    for log in logger.logs:
        print(f"    [{log['timestamp'][:19]}] {log['tool']}({log['params']}) "
              f"-> {log['status']} ({log['duration_ms']}ms)")

    print(f"\n  PASS -- 4 middlewares working in chain.\n")


# ==========================================================================
# MAIN -- Run all solutions
# ==========================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  Day 2 Solutions -- Tool Use & Function Calling")
    print("#" * 60)

    # Easy exercises
    easy_ex1_translate_tool()
    easy_ex2_format_conversion()
    easy_ex3_actionable_errors()

    # Medium exercises
    medium_ex1_function_calling_agent()
    medium_ex2_structured_output()
    medium_ex3_middleware()

    # Hard exercises are not included here because they are substantial
    # projects (150-300 lines each). Key hints:
    #
    # Hard Ex 1 (Dynamic Tool Discovery):
    #   - The "Tool Server" is just a dict with 3 endpoints simulated as functions
    #   - The agent starts with 3 meta-tools: discover_tools, get_tool_schema, call_tool
    #   - The system prompt says: "You don't know what tools are available. Use
    #     discover_tools first, then get_tool_schema to learn how to use them."
    #   - The agent loop is standard ReAct, but the tools are meta-tools
    #   - This pattern is how MCP servers work -- tools are discovered at runtime
    #
    # Hard Ex 2 (Tool Composition Engine):
    #   - Steps are dicts with "tool", "params_template", "output_key"
    #   - Template resolution: replace {var} in params_template with values from context
    #   - Context is a dict that accumulates outputs: {"step1_result": ..., "step2_result": ...}
    #   - Error handling: wrap each step in try/except, check continue_on_error flag
    #   - The composed tool's fn() runs all steps sequentially and returns the final result
    #   - Register the composed tool normally in the registry -- the agent can't tell the difference

    print("\n" + "#" * 60)
    print("  All solutions executed successfully.")
    print("#" * 60 + "\n")
