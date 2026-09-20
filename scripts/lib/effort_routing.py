"""Deterministic, conservative effort routing for DukeOTR runtime quality traces.

Routing decides which *existing* quality checks are relevant before a Builder response is
produced. It does not decide that an answer is correct, bypass evaluation isolation, or promote
anything into training data. Ambiguous and security-sensitive requests deliberately route up,
not down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_ALL_CHECKS = (
    "structure_and_safety",
    "requirements",
    "english_clarity",
    "luau_code",
    "roblox_api",
    "client_server_security",
    "persistence_economy",
    "lifecycle_performance",
    "testability",
)

_CODE_TASK_TYPES = {
    "bug_fix",
    "code_generation",
    "code_explanation",
    "code_review",
    "completion",
    "diagnosis_correction",
    "natural_language_to_luau",
    "optimization",
    "requirements_implementation",
}

# A transparent term scan is intentionally explainable and conservative. New high-risk classes
# should be added to configuration rather than relying on a classifier model to infer risk.
_DEFAULT_HIGH_RISK_TERMS: dict[str, tuple[str, ...]] = {
    "client_server_security": (
        "remoteevent",
        "remotefunction",
        "fireserver",
        "fireclient",
        "onserverevent",
        "onserverinvoke",
        "client/server",
        "client server",
        "networking",
        "network",
        "replication",
        "server authority",
        "exploit",
        "security",
        "authorization",
        "authenticate",
        "untrusted client",
    ),
    "persistence_economy": (
        "datastore",
        "data store",
        "currency",
        "purchase",
        "receipt",
        "marketplace",
        "inventory",
        "shop",
        "gamepass",
        "developer product",
    ),
    "spatial_combat": (
        "combat",
        "damage",
        "raycast",
        "raycasting",
        "physics",
        "network ownership",
        "hit detection",
    ),
    "architecture_lifecycle": (
        "architecture",
        "matchmaking",
        "complex system",
        "large system",
        "complex debugging",
        "deep debugging",
        "subtle bug",
        "multi-server",
        "streaming",
        "race condition",
        "memory leak",
        "lifecycle",
        "concurrency",
    ),
}

_DEFAULT_AMBIGUITY_TERMS = (
    "ambiguous",
    "unspecified",
    "not sure",
    "unclear",
    "it depends",
    "unknown requirement",
)

# Explicit identifiers catch factual API questions that do not literally say "Roblox API".
# This affects review scope only; it is not an assertion that an identifier is documented.
_ROBLOX_API_INDICATORS = (
    "getservice",
    "workspace",
    "raycast",
    "raycastparams",
    "textbutton",
    "userinputservice",
    "tweenservice",
    "runservice",
    "playeradded",
    "characteradded",
    "remoteevent",
    "remotefunction",
    "datastore",
    "replicatedstorage",
    "serverstorage",
    "waitforchild",
    "collectionservice",
    "humanoid",
)

_SIMPLE_PATTERNS = (
    re.compile(r"^\s*(?:hi|hello|hey|thanks|thank you)[!.?\s]*$", re.IGNORECASE),
    re.compile(r"\b(?:what is|define|meaning of|explain the word)\b", re.IGNORECASE),
    re.compile(r"\b(?:rewrite|rephrase|word this|fix this sentence)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class EffortRoute:
    """Serializable result of deterministic task routing."""

    level: str
    reasons: tuple[str, ...]
    risk_labels: tuple[str, ...]
    selected_checks: tuple[str, ...]
    skipped_checks: tuple[str, ...]
    reviewer_passes: int
    default_max_rounds: int
    maximum_rounds: int
    minimum_assistant_characters: int
    code_book_context_limit: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "reasons": list(self.reasons),
            "risk_labels": list(self.risk_labels),
            "selected_checks": list(self.selected_checks),
            "skipped_checks": list(self.skipped_checks),
            "reviewer_passes": self.reviewer_passes,
            "default_max_rounds": self.default_max_rounds,
            "maximum_rounds": self.maximum_rounds,
            "minimum_assistant_characters": self.minimum_assistant_characters,
            "code_book_context_limit": self.code_book_context_limit,
        }


def _lower_text(seed: dict[str, Any]) -> str:
    values: list[str] = [
        str(seed.get("title", "")),
        str(seed.get("user_request", "")),
        str(seed.get("task_type", "")),
        str(seed.get("difficulty", "")),
    ]
    for field in ("concepts", "tags", "requirements", "expected_evidence"):
        value = seed.get(field, [])
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    return "\n".join(values).casefold()


def _risk_text(seed: dict[str, Any]) -> str:
    """Use affirmative task identity, not an `avoid`/negative requirement, for escalation."""
    values = [str(seed.get("title", "")), str(seed.get("user_request", "")), str(seed.get("task_type", ""))]
    for field in ("concepts", "tags"):
        value = seed.get(field, [])
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    return "\n".join(values).casefold()


def _term_is_explicitly_negated(statement: str, term_start: int) -> bool:
    """Tell whether one term occurrence belongs to a local negative-scope clause.

    Requirements are otherwise useful positive risk signals when a short user request omits
    implementation details. Looking only since the nearest punctuation means a general warning
    such as "do not trust the client; validate RemoteEvent input" does not negate the later
    RemoteEvent requirement.
    """
    clause_start = max(statement.rfind(marker, 0, term_start) for marker in (".", ";", ":", "\n")) + 1
    prefix = statement[clause_start:term_start]
    return bool(re.search(r"\b(?:do not|don't|avoid|never|must not|without|not introduce)\b", prefix))


def _high_risk_term_matches(seed: dict[str, Any], risk_text: str, term: str) -> bool:
    if term in risk_text:
        return True
    requirements = seed.get("requirements", [])
    if not isinstance(requirements, list):
        return False
    for requirement in requirements:
        statement = str(requirement).casefold()
        for match in re.finditer(re.escape(term), statement):
            if not _term_is_explicitly_negated(statement, match.start()):
                return True
    return False


def _normalise_terms(raw: Any, fallback: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return fallback
    if not isinstance(raw, dict):
        raise ValueError("adaptive_routing.high_risk_terms must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for label, terms in raw.items():
        if not isinstance(label, str) or not label.strip() or not isinstance(terms, list) or not terms:
            raise ValueError("Each adaptive_routing.high_risk_terms entry needs a name and non-empty string list")
        if not all(isinstance(term, str) and term.strip() for term in terms):
            raise ValueError("adaptive_routing.high_risk_terms values must be non-empty strings")
        result[label] = tuple(term.casefold() for term in terms)
    return result


def _profile(config: dict[str, Any], level: str) -> dict[str, Any]:
    routing = config.get("adaptive_routing", {})
    if not isinstance(routing, dict):
        raise ValueError("adaptive_routing must be an object")
    profiles = routing.get("profiles", {})
    if not isinstance(profiles, dict) or not isinstance(profiles.get(level), dict):
        raise ValueError(f"adaptive_routing.profiles.{level} must be an object")
    profile = profiles[level]
    expected_positive = (
        "reviewer_passes",
        "default_max_rounds",
        "maximum_rounds",
        "minimum_assistant_characters",
        "code_book_context_limit",
    )
    for key in expected_positive:
        value = profile.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"adaptive_routing.profiles.{level}.{key} must be a non-negative integer")
    if profile["default_max_rounds"] < 1 or profile["maximum_rounds"] < profile["default_max_rounds"]:
        raise ValueError(f"adaptive_routing.profiles.{level} has invalid round limits")
    return profile


def _checks(seed: dict[str, Any], level: str, risks: set[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    task_type = str(seed.get("task_type", "")).casefold()
    selected: set[str] = {"structure_and_safety"}
    if level != "simple":
        selected.update({"requirements", "english_clarity"})
    text = _lower_text(seed)
    risk_text = _risk_text(seed)
    if task_type in _CODE_TASK_TYPES or "luau" in text or "script" in risk_text or "code" in risk_text:
        selected.add("luau_code")
    if (
        "api" in risk_text
        or "roblox" in risk_text
        or any(identifier in risk_text for identifier in _ROBLOX_API_INDICATORS)
        or "roblox_api" in risks
    ):
        selected.add("roblox_api")
    if risks.intersection({"client_server_security", "spatial_combat"}):
        selected.add("client_server_security")
    if "persistence_economy" in risks:
        selected.add("persistence_economy")
    if risks.intersection({"architecture_lifecycle", "spatial_combat"}) or "optimization" in text:
        selected.add("lifecycle_performance")
    if level == "complex":
        selected.add("testability")
    ordered = tuple(check for check in _ALL_CHECKS if check in selected)
    skipped = tuple(check for check in _ALL_CHECKS if check not in selected)
    return ordered, skipped


def classify_task(seed: dict[str, Any], config: dict[str, Any]) -> EffortRoute:
    """Classify a curated task deterministically with high-risk/ambiguity escalation.

    `simple` is intentionally narrow. Any unknown task form, explicit ambiguity, advanced
    difficulty, many requirements, or a configured risk phrase avoids the fast path.
    """
    routing = config.get("adaptive_routing", {})
    if not isinstance(routing, dict) or routing.get("enabled") is not True:
        raise ValueError("adaptive_routing.enabled must be true for adaptive routing")
    text = _lower_text(seed)
    risk_text = _risk_text(seed)
    high_risk_terms = _normalise_terms(routing.get("high_risk_terms"), _DEFAULT_HIGH_RISK_TERMS)
    risks = {
        label
        for label, terms in high_risk_terms.items()
        if any(_high_risk_term_matches(seed, risk_text, term) for term in terms)
    }
    ambiguity_terms = routing.get("ambiguity_terms", list(_DEFAULT_AMBIGUITY_TERMS))
    if not isinstance(ambiguity_terms, list) or not all(isinstance(term, str) and term.strip() for term in ambiguity_terms):
        raise ValueError("adaptive_routing.ambiguity_terms must be a list of non-empty strings")
    ambiguity_matches = [term for term in ambiguity_terms if term.casefold() in text]
    requirement_count = len(seed.get("requirements", [])) if isinstance(seed.get("requirements", []), list) else 0
    complex_requirement_count = routing.get("complex_requirement_count", 5)
    if not isinstance(complex_requirement_count, int) or complex_requirement_count < 1:
        raise ValueError("adaptive_routing.complex_requirement_count must be a positive integer")
    task_type = str(seed.get("task_type", "")).casefold()
    difficulty = str(seed.get("difficulty", "")).casefold()
    prompt = str(seed.get("user_request", ""))
    simple_task_types = routing.get("simple_task_types", ["question_answer", "code_explanation"])
    if not isinstance(simple_task_types, list) or not all(isinstance(item, str) and item for item in simple_task_types):
        raise ValueError("adaptive_routing.simple_task_types must be a list of non-empty strings")
    simple_task_types = [item.casefold() for item in simple_task_types]
    simple_max_requirements = routing.get("simple_max_requirements", 2)
    simple_max_characters = routing.get("simple_max_prompt_characters", 280)
    if not isinstance(simple_max_requirements, int) or simple_max_requirements < 0:
        raise ValueError("adaptive_routing.simple_max_requirements must be a non-negative integer")
    if not isinstance(simple_max_characters, int) or simple_max_characters < 1:
        raise ValueError("adaptive_routing.simple_max_prompt_characters must be a positive integer")

    reasons: list[str] = []
    if risks:
        reasons.extend(f"high_risk:{label}" for label in sorted(risks))
    if ambiguity_matches:
        reasons.extend(f"ambiguous:{term}" for term in sorted(ambiguity_matches))
    if difficulty == "advanced":
        reasons.append("advanced_difficulty")
    if requirement_count >= complex_requirement_count:
        reasons.append(f"many_requirements:{requirement_count}")
    if reasons:
        level = "complex"
    else:
        simple_signal = any(pattern.search(prompt) for pattern in _SIMPLE_PATTERNS)
        if (
            task_type in simple_task_types
            and difficulty == "beginner"
            and requirement_count <= simple_max_requirements
            and len(prompt) <= simple_max_characters
            and simple_signal
        ):
            level = "simple"
            reasons.append("narrow_low_risk_simple_signal")
        else:
            level = "normal"
            reasons.append("default_normal_route")
    profile = _profile(config, level)
    selected, skipped = _checks(seed, level, risks)
    return EffortRoute(
        level=level,
        reasons=tuple(reasons),
        risk_labels=tuple(sorted(risks)),
        selected_checks=selected,
        skipped_checks=skipped,
        reviewer_passes=int(profile["reviewer_passes"]),
        default_max_rounds=int(profile["default_max_rounds"]),
        maximum_rounds=int(profile["maximum_rounds"]),
        minimum_assistant_characters=int(profile["minimum_assistant_characters"]),
        code_book_context_limit=int(profile["code_book_context_limit"]),
    )
