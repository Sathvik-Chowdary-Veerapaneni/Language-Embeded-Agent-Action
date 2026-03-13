"""
LEAA Language Layer — Command Parser

Parses natural-language archery commands into structured ParsedCommand objects.

Strategy:
  1. Rule-based extraction (fast, free, no API calls).
  2. LLM fallback via Anthropic API only when rules produce low confidence
     or fail to identify a target.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ParsedCommand dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedCommand:
    """Structured representation of a parsed archery command."""

    target_color: Optional[str] = None
    target_id: Optional[str] = None
    spatial_ref: Optional[str] = None
    speed_ref: Optional[str] = None
    ordinal: Optional[int] = None
    arrow_type: Optional[str] = None
    confidence: float = 0.0

    @property
    def has_target(self) -> bool:
        """True when at least one target-identifying field is set."""
        return any([
            self.target_color,
            self.target_id,
            self.spatial_ref,
            self.speed_ref,
            self.ordinal is not None,
        ])


# ---------------------------------------------------------------------------
# Lookup tables for rule-based extraction
# ---------------------------------------------------------------------------

_COLORS = {"red", "blue", "yellow", "green", "white"}

_SPATIAL_MAP = {
    "closest": "closest",
    "nearest": "closest",
    "farthest": "farthest",
    "furthest": "farthest",
    "leftmost": "leftmost",
    "rightmost": "rightmost",
}

_SPEED_MAP = {
    "fastest": "fastest",
    "slowest": "slowest",
    "stationary": "stationary",
    "still": "stationary",
    "stopped": "stationary",
}

_ARROW_MAP = {
    "fire": "fire",
    "ice": "ice",
    "heavy": "heavy",
    "normal": "normal",
    "standard": "normal",
    "light": "light",
}

_ORDINAL_MAP = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
    "5th": 5,
}

# Regex for explicit object IDs like obj_0, obj_12, target_3
_ID_PATTERN = re.compile(r"\b(obj_\d+|target_\d+)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# CommandParser
# ---------------------------------------------------------------------------

class CommandParser:
    """Parses NL archery commands via rules first, LLM fallback second."""

    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback

    # ---- public API -------------------------------------------------------

    def parse(self, command: str) -> ParsedCommand:
        """Parse a natural-language command string.

        Tries rule-based extraction first.  Falls back to LLM only when
        ``has_target`` is False or confidence < 0.7.
        """
        parsed = self._parse_rules(command)

        if parsed.has_target and parsed.confidence >= 0.7:
            return parsed

        # Attempt LLM fallback
        if self.use_llm_fallback:
            llm_parsed = self._parse_llm(command)
            if llm_parsed is not None and llm_parsed.has_target:
                # Merge: prefer LLM results but keep any rule hits not in LLM
                return self._merge(parsed, llm_parsed)

        return parsed

    # ---- rule-based parser ------------------------------------------------

    def _parse_rules(self, command: str) -> ParsedCommand:
        """Deterministic, zero-cost extraction from the raw command string."""
        text = command.lower().strip()
        result = ParsedCommand()

        # Explicit ID (e.g. obj_3)
        id_match = _ID_PATTERN.search(text)
        if id_match:
            result.target_id = id_match.group(1).lower()

        # Color
        for color in _COLORS:
            if re.search(rf"\b{color}\b", text):
                result.target_color = color
                break

        # Spatial reference
        for token, ref in _SPATIAL_MAP.items():
            if re.search(rf"\b{token}\b", text):
                result.spatial_ref = ref
                break

        # Speed reference
        for token, ref in _SPEED_MAP.items():
            if re.search(rf"\b{token}\b", text):
                result.speed_ref = ref
                break

        # Arrow type
        for token, atype in _ARROW_MAP.items():
            # Avoid matching "fire" as a verb — require "fire arrow" or
            # "use fire" context, but keep it simple: any mention counts.
            if re.search(rf"\b{token}\b", text):
                result.arrow_type = atype
                break

        # Ordinal
        for token, ordinal in _ORDINAL_MAP.items():
            if re.search(rf"\b{token}\b", text):
                result.ordinal = ordinal
                break

        # Confidence heuristic
        filled = sum([
            result.target_color is not None,
            result.target_id is not None,
            result.spatial_ref is not None,
            result.speed_ref is not None,
            result.ordinal is not None,
        ])
        if result.target_id:
            result.confidence = 0.95
        elif filled >= 2:
            result.confidence = 0.9
        elif filled == 1:
            result.confidence = 0.8
        else:
            result.confidence = 0.3

        return result

    # ---- LLM fallback parser ----------------------------------------------

    def _parse_llm(self, command: str) -> Optional[ParsedCommand]:
        """Call Anthropic API to parse a command the rules couldn't handle."""
        try:
            import anthropic  # type: ignore
        except ImportError:
            return None

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        # Load system prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
        if prompt_path.exists():
            system_prompt = prompt_path.read_text()
        else:
            system_prompt = (
                "You are a command parser for an archery game. "
                "Output valid JSON with keys: target_color, target_id, "
                "spatial_ref, speed_ref, ordinal, arrow_type, confidence."
            )

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": command}],
            )
            raw = response.content[0].text.strip()

            # Extract JSON from response (may be wrapped in markdown fences)
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())

            return ParsedCommand(
                target_color=data.get("target_color"),
                target_id=data.get("target_id"),
                spatial_ref=data.get("spatial_ref"),
                speed_ref=data.get("speed_ref"),
                ordinal=data.get("ordinal"),
                arrow_type=data.get("arrow_type"),
                confidence=float(data.get("confidence", 0.6)),
            )
        except Exception:
            return None

    # ---- merge helper -----------------------------------------------------

    @staticmethod
    def _merge(rules: ParsedCommand, llm: ParsedCommand) -> ParsedCommand:
        """Merge rule-based and LLM results, preferring LLM where populated."""
        return ParsedCommand(
            target_color=llm.target_color or rules.target_color,
            target_id=llm.target_id or rules.target_id,
            spatial_ref=llm.spatial_ref or rules.spatial_ref,
            speed_ref=llm.speed_ref or rules.speed_ref,
            ordinal=llm.ordinal if llm.ordinal is not None else rules.ordinal,
            arrow_type=llm.arrow_type or rules.arrow_type,
            confidence=max(llm.confidence, rules.confidence),
        )
