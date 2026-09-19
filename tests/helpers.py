from __future__ import annotations

from typing import Any

from scripts.lib.schema import make_generated_record


def seed(seed_id: str = "train-test-001", prompt: str = "Show a secure server RemoteEvent pattern.") -> dict[str, Any]:
    return {
        "id": seed_id,
        "split": "train",
        "title": "Test seed",
        "task_type": "code_generation",
        "difficulty": "intermediate",
        "user_request": prompt,
        "requirements": ["Keep authority on the server"],
        "concepts": ["RemoteEvents", "security"],
        "expected_evidence": ["server validation"],
        "avoid": ["client trust"],
        "tags": ["remoteevents", "security"],
        "source": {"kind": "project_authored"},
    }


def good_answer() -> str:
    return """A RemoteEvent lets a client request an action, but the request is not proof that the action is allowed. Put the RemoteEvent in ReplicatedStorage so both sides can reference it, and keep the price and reward on the server. Never trust the client to supply a currency amount.

```luau
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local BuyRequest = ReplicatedStorage:WaitForChild("BuyRequest")
local PRICE = 25

BuyRequest.OnServerEvent:Connect(function(player, itemId)
    if itemId ~= "Potion" then
        return
    end
    local coins = player:GetAttribute("Coins")
    if type(coins) ~= "number" or coins < PRICE then
        return
    end
    player:SetAttribute("Coins", coins - PRICE)
    -- Grant a server-owned inventory entry here.
end)
```

The client can display the outcome, but it must not decide the price, balance, or grant."""


def reviewed_record(seed_id: str = "train-test-001", prompt: str = "Show a secure server RemoteEvent pattern.") -> dict[str, Any]:
    record = make_generated_record(seed(seed_id, prompt), good_answer(), generator={"kind": "test"}, variant=1)
    record["quality"]["static"] = {"status": "pass", "issues": [], "checked_at": "2026-09-19T00:00:00Z"}
    record["quality"]["llm_review"] = {
        "status": "complete",
        "decision": "accept",
        "review": {"scores": {"accuracy": 5, "security": 5, "pedagogy": 4}},
        "checked_at": "2026-09-19T00:00:00Z",
    }
    record["quality"]["deduplication"] = {
        "status": "unique",
        "duplicate_of": None,
        "similarity": None,
        "checked_at": "2026-09-19T00:00:00Z",
    }
    return record
