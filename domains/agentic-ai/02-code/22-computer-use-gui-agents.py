"""
Day 22 -- Computer use & GUI agents: virtual screen, set-of-marks, grounding.

Demonstrates:
  1. VirtualScreen    -- ASCII grid representing a GUI page with interactive elements
  2. screenshot()     -- renders the screen as a text representation (simulates vision)
  3. set_of_marks()   -- annotates each interactive element with [N] labels
  4. GUIAction        -- typed action primitives (click, type, scroll, key)
  5. ActionExecutor   -- resolves mark ids to coordinates, executes actions on VirtualScreen
  6. GUIAgent         -- scripted policy: perceive->mark->act loop to fill a login form
  7. Grounding error  -- demonstration of clicking the wrong element (mark confusion)

No real GUI, no API key, no third-party library required.
stdlib only.

Run:
    python domains/agentic-ai/02-code/22-computer-use-gui-agents.py
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any


# ===========================================================================
# 1. SCREEN ELEMENTS — what a GUI is made of
# ===========================================================================

@dataclass
class BoundingBox:
    """Top-left corner + dimensions of a UI element (in character cells)."""
    x: int       # column
    y: int       # row
    width: int
    height: int

    def contains(self, px: int, py: int) -> bool:
        """Return True if pixel (px, py) is inside this bounding box."""
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def center(self) -> tuple[int, int]:
        """Return the center (column, row) of this box."""
        return self.x + self.width // 2, self.y + self.height // 2


@dataclass
class UIElement:
    """A single interactive element on the virtual screen."""
    kind: str         # "button" | "input" | "link" | "label"
    label: str        # visible text
    bbox: BoundingBox
    value: str = ""   # current value (for input fields)
    focusable: bool = True   # inputs and buttons are focusable; static labels are not


# ===========================================================================
# 2. VIRTUAL SCREEN — the simulated GUI environment
# ===========================================================================

class VirtualScreen:
    """
    A simple ASCII-art GUI environment.

    The 'screen' is a 2D grid of character cells. UIElements occupy regions
    defined by their BoundingBox. Only focusable elements accept actions.
    """

    WIDTH = 50    # columns
    HEIGHT = 14   # rows

    def __init__(self, elements: list[UIElement]) -> None:
        self.elements = elements
        self.focused: UIElement | None = None  # currently focused element

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _build_grid(self, marks: dict[int, UIElement] | None = None) -> list[list[str]]:
        """
        Build the raw character grid.

        If `marks` is provided, each element also shows its mark id as [N].
        """
        # Start with a blank grid filled with spaces
        grid: list[list[str]] = [[" "] * self.WIDTH for _ in range(self.HEIGHT)]

        # Draw a simple border
        for col in range(self.WIDTH):
            grid[0][col] = "─"
            grid[self.HEIGHT - 1][col] = "─"
        for row in range(self.HEIGHT):
            grid[row][0] = "│"
            grid[row][self.WIDTH - 1] = "│"
        grid[0][0] = "┌"
        grid[0][self.WIDTH - 1] = "┐"
        grid[self.HEIGHT - 1][0] = "└"
        grid[self.HEIGHT - 1][self.WIDTH - 1] = "┘"

        # Draw each element
        for elem in self.elements:
            b = elem.bbox
            text = self._render_element_text(elem)

            # Draw top/bottom border of the element box
            for col in range(b.x, b.x + b.width):
                if 0 <= b.y < self.HEIGHT:
                    grid[b.y][col] = "─"
                if 0 <= b.y + b.height - 1 < self.HEIGHT:
                    grid[b.y + b.height - 1][col] = "─"

            # Draw side borders
            for row in range(b.y, b.y + b.height):
                if 0 <= row < self.HEIGHT:
                    if 0 <= b.x < self.WIDTH:
                        grid[row][b.x] = "│"
                    if 0 <= b.x + b.width - 1 < self.WIDTH:
                        grid[row][b.x + b.width - 1] = "│"

            # Draw corners
            corners = [
                (b.y, b.x, "┌"), (b.y, b.x + b.width - 1, "┐"),
                (b.y + b.height - 1, b.x, "└"), (b.y + b.height - 1, b.x + b.width - 1, "┘"),
            ]
            for row, col, ch in corners:
                if 0 <= row < self.HEIGHT and 0 <= col < self.WIDTH:
                    grid[row][col] = ch

            # Place the text content in the center row of the element
            center_row = b.y + b.height // 2
            inner_width = b.width - 2  # subtract borders
            padded = text[:inner_width].ljust(inner_width)

            # Append mark label if marks is provided
            if marks:
                for mark_id, marked_elem in marks.items():
                    if marked_elem is elem:
                        mark_str = f"[{mark_id}]"
                        # Place mark at the right of the inner text
                        padded = (text[:max(0, inner_width - len(mark_str))] + mark_str)[:inner_width]

            if 0 <= center_row < self.HEIGHT:
                for i, ch in enumerate(padded):
                    col = b.x + 1 + i
                    if 0 <= col < self.WIDTH - 1:
                        grid[center_row][col] = ch

        return grid

    def _render_element_text(self, elem: UIElement) -> str:
        """Format the visible text for an element."""
        if elem.kind == "input":
            # Show the current value or a placeholder
            display = elem.value if elem.value else f"({elem.label})"
            return display
        elif elem.kind == "button":
            return f"[ {elem.label} ]"
        elif elem.kind == "link":
            return f">> {elem.label}"
        else:
            return elem.label

    # ------------------------------------------------------------------
    # Public actions
    # ------------------------------------------------------------------

    def screenshot(self) -> str:
        """
        Capture the current state of the screen as an ASCII string.

        In a real system, this would return a base64-encoded PNG.
        Here we return a human-readable text grid (same information).
        """
        grid = self._build_grid()
        lines = ["".join(row) for row in grid]
        return "\n".join(lines)

    def set_of_marks(self) -> tuple[str, dict[int, UIElement]]:
        """
        Annotate the screenshot with numeric marks [1], [2], ... for each
        focusable element. Returns:
            - annotated screenshot (str)
            - marks dict: {mark_id: UIElement}
        """
        # Build the marks mapping (only focusable elements get a mark)
        marks: dict[int, UIElement] = {}
        for i, elem in enumerate(self.elements, start=1):
            if elem.focusable:
                marks[i] = elem

        grid = self._build_grid(marks=marks)
        lines = ["".join(row) for row in grid]
        annotated = "\n".join(lines)

        return annotated, marks

    def click(self, x: int, y: int) -> str:
        """
        Simulate a left-click at coordinates (x, y).

        Returns a description of what happened (success or miss).
        """
        for elem in self.elements:
            if elem.focusable and elem.bbox.contains(x, y):
                self.focused = elem
                if elem.kind == "button":
                    return f"[CLICK] Button '{elem.label}' activated"
                elif elem.kind == "link":
                    return f"[CLICK] Link '{elem.label}' followed"
                elif elem.kind == "input":
                    return f"[CLICK] Input '{elem.label}' focused"
        return f"[CLICK] Miss — no focusable element at ({x}, {y})"

    def type_text(self, text: str) -> str:
        """
        Type text into the currently focused input field.

        Returns an error if no input is focused.
        """
        if self.focused is None:
            return "[TYPE] Error — no element focused. Click an input first."
        if self.focused.kind != "input":
            return f"[TYPE] Error — focused element '{self.focused.label}' is not an input."
        self.focused.value += text
        return f"[TYPE] Typed '{text}' into '{self.focused.label}' → value='{self.focused.value}'"

    def scroll(self, direction: str, amount: int = 3) -> str:
        """Simulate a scroll action (no-op in our static screen, shown for API parity)."""
        return f"[SCROLL] {direction} by {amount} units (no-op on static screen)"


# ===========================================================================
# 3. ACTION EXECUTOR — bridges mark ids to screen actions
# ===========================================================================

@dataclass
class GUIAction:
    """A typed action primitive, analogous to Claude computer use tool input."""
    action: str                    # "click_mark" | "click_coord" | "type" | "scroll" | "screenshot"
    mark_id: int | None = None     # for click_mark
    x: int | None = None           # for click_coord
    y: int | None = None
    text: str | None = None        # for type
    direction: str | None = None   # for scroll


class ActionExecutor:
    """
    Resolves GUIActions against a VirtualScreen.

    In production, this would call pyautogui / playwright / xdotool.
    Here it calls VirtualScreen methods directly.
    """

    def __init__(self, screen: VirtualScreen) -> None:
        self.screen = screen
        self._marks: dict[int, UIElement] = {}  # populated after set_of_marks

    def perceive(self) -> str:
        """Take a screenshot and return its text representation."""
        return self.screen.screenshot()

    def perceive_with_marks(self) -> tuple[str, dict[int, UIElement]]:
        """Take an annotated screenshot, storing mark mapping for later use."""
        annotated, marks = self.screen.set_of_marks()
        self._marks = marks
        return annotated, marks

    def execute(self, action: GUIAction) -> str:
        """Execute a GUIAction and return the result string."""
        if action.action == "screenshot":
            return self.screen.screenshot()

        elif action.action == "click_mark":
            if action.mark_id is None:
                return "[EXECUTOR] Error: click_mark requires mark_id"
            if action.mark_id not in self._marks:
                return f"[EXECUTOR] Error: mark [{action.mark_id}] not found — did you call perceive_with_marks first?"
            elem = self._marks[action.mark_id]
            cx, cy = elem.bbox.center()
            return self.screen.click(cx, cy)

        elif action.action == "click_coord":
            if action.x is None or action.y is None:
                return "[EXECUTOR] Error: click_coord requires x and y"
            return self.screen.click(action.x, action.y)

        elif action.action == "type":
            if action.text is None:
                return "[EXECUTOR] Error: type requires text"
            return self.screen.type_text(action.text)

        elif action.action == "scroll":
            return self.screen.scroll(action.direction or "down", 3)

        else:
            return f"[EXECUTOR] Unknown action: {action.action}"


# ===========================================================================
# 4. GUI AGENT — scripted policy (no LLM needed for the demo)
# ===========================================================================

class GUIAgent:
    """
    A scripted GUI agent that fills a virtual login form.

    In production, the 'decide_next_action' method would call a vision LLM
    (Claude computer use, GPT-4o, etc.) and parse its output.
    Here we use a deterministic policy to illustrate the perceive→mark→act loop.
    """

    def __init__(self, executor: ActionExecutor, max_steps: int = 20) -> None:
        self.executor = executor
        self.max_steps = max_steps
        self.step = 0
        self.done = False
        self.log: list[str] = []

    def _log(self, msg: str) -> None:
        self.log.append(msg)
        print(msg)

    def run(self, task: str, policy: list[GUIAction]) -> None:
        """
        Execute a fixed policy (list of actions) against the virtual screen.

        In a real agent, 'policy' would be generated step-by-step by a LLM
        based on the annotated screenshot and the task description.
        """
        self._log(f"\n{'='*60}")
        self._log(f"TASK: {task}")
        self._log(f"{'='*60}")

        for action in policy:
            if self.step >= self.max_steps:
                self._log(f"[AGENT] Max steps ({self.max_steps}) reached — stopping.")
                break

            self.step += 1
            self._log(f"\n--- Step {self.step}: action={action.action} ---")

            if action.action in ("screenshot", "click_mark", "click_coord", "type", "scroll"):
                # Before each action, show the current (annotated) screen
                if action.action != "screenshot":
                    annotated, marks = self.executor.perceive_with_marks()
                    self._log("\n[SCREEN — set-of-marks view]")
                    self._log(annotated)
                    marks_desc = {k: f"{v.kind}:'{v.label}'" for k, v in marks.items()}
                    self._log(f"[MARKS] {marks_desc}")

                # Execute the action
                result = self.executor.execute(action)
                self._log(f"[RESULT] {result}")

        # Final screenshot
        self._log("\n[SCREEN — final state]")
        self._log(self.executor.perceive())
        self._log(f"\n[AGENT] Completed {self.step} steps.")


# ===========================================================================
# 5. DEMO SETUP — virtual login page
# ===========================================================================

def make_login_screen() -> VirtualScreen:
    """Build a simple virtual login form with username, password, submit, cancel."""
    elements = [
        UIElement(
            kind="label",
            label="  Login to FleetSim",
            bbox=BoundingBox(x=1, y=1, width=48, height=2),
            focusable=False,
        ),
        UIElement(
            kind="input",
            label="Username",
            bbox=BoundingBox(x=5, y=4, width=38, height=2),
        ),
        UIElement(
            kind="input",
            label="Password",
            bbox=BoundingBox(x=5, y=7, width=38, height=2),
        ),
        UIElement(
            kind="button",
            label="Submit",
            bbox=BoundingBox(x=8, y=10, width=14, height=2),
        ),
        UIElement(
            kind="button",
            label="Cancel",
            bbox=BoundingBox(x=26, y=10, width=14, height=2),
        ),
    ]
    return VirtualScreen(elements)


# ===========================================================================
# 6. GROUNDING ERROR DEMO — clicking the wrong mark
# ===========================================================================

def demo_grounding_error(screen: VirtualScreen) -> None:
    """
    Illustrate a grounding error: the agent intends to click Submit [4]
    but uses the wrong mark id [5] = Cancel.

    Marks in our login screen: [2]=Username, [3]=Password, [4]=Submit, [5]=Cancel.

    This simulates what happens when:
      - The LLM hallucinates a mark id (off-by-one)
      - The marks shift after a DOM update (a new element was inserted)
      - The agent reuses a stale marks snapshot from a previous perceive step
    """
    print("\n" + "="*60)
    print("DEMO: Grounding Error — clicking the wrong button")
    print("="*60)

    executor = ActionExecutor(screen)

    # Obtain the correct mark mapping
    annotated, marks = executor.perceive_with_marks()
    print("\n[SCREEN with marks]")
    print(annotated)
    print(f"[MARKS] correct mapping: {{{', '.join(f'{k}: {v.label}' for k, v in marks.items())}}}")

    # Agent INTENDS to click Submit ([4]) but accidentally uses mark [5] = Cancel
    # This is a classic off-by-one grounding error (LLM hallucinated the wrong id).
    wrong_action = GUIAction(action="click_mark", mark_id=5)
    print("\n[AGENT INTENT] Click 'Submit' [4] — but uses wrong mark id [5] = Cancel")
    result = executor.execute(wrong_action)
    print(f"[RESULT] {result}")
    print("\n[DIAGNOSIS] The agent clicked 'Cancel' instead of 'Submit'.")
    print("  Root cause: LLM produced mark id [5] (off-by-one hallucination).")
    print("  In production this would silently cancel the form without any error.")
    print("  Fix: always call perceive_with_marks() immediately before each click,")
    print("  and validate that the resolved element label matches the intent.")


# ===========================================================================
# 7. MAIN DEMO
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("Day 22 — Computer use & GUI agents (virtual demo)")
    print("=" * 60)

    # ----- PART 1: plain screenshot -----
    print("\n[1] Plain screenshot (what a raw vision LLM would see)")
    screen = make_login_screen()
    print(screen.screenshot())

    # ----- PART 2: set-of-marks screenshot -----
    print("\n[2] Set-of-Marks annotated screenshot")
    executor = ActionExecutor(screen)
    annotated, marks = executor.perceive_with_marks()
    print(annotated)
    print("\n[MARKS TABLE]")
    for mid, elem in marks.items():
        cx, cy = elem.bbox.center()
        print(f"  [{mid}] kind={elem.kind:6s}  label='{elem.label}'  center=({cx},{cy})")

    # ----- PART 3: GUIAgent fills the login form -----
    # The 'policy' below simulates what a vision LLM would produce step by step
    # after seeing the annotated screenshot and the task description.
    # Marks: [2]=Username input, [3]=Password input, [4]=Submit button, [5]=Cancel button
    # (The non-focusable title label occupies no mark slot; numbering starts at 2.)
    policy: list[GUIAction] = [
        # Click the Username input (mark [2])
        GUIAction(action="click_mark", mark_id=2),
        # Type the username
        GUIAction(action="type", text="alice"),
        # Click the Password input (mark [3])
        GUIAction(action="click_mark", mark_id=3),
        # Type the password
        GUIAction(action="type", text="s3cr3t!"),
        # Click the Submit button (mark [4])
        GUIAction(action="click_mark", mark_id=4),
    ]

    # Fresh screen for the agent demo
    fresh_screen = make_login_screen()
    agent_executor = ActionExecutor(fresh_screen)
    agent = GUIAgent(agent_executor, max_steps=20)
    agent.run(
        task="Log in to FleetSim with username 'alice' and password 's3cr3t!'",
        policy=policy,
    )

    # ----- PART 4: grounding error demo -----
    # Use yet another fresh screen so the values are empty
    error_screen = make_login_screen()
    demo_grounding_error(error_screen)

    print("\n" + "="*60)
    print("End of Day 22 demo.")
    print("Key takeaways:")
    print("  - perceive→mark→act is the core GUI agent loop")
    print("  - set-of-marks reduces grounding errors (pixel → id)")
    print("  - always refresh marks before each click to avoid stale ids")
    print("  - real agents replace the scripted policy with a vision LLM call")
    print("="*60)


if __name__ == "__main__":
    main()
