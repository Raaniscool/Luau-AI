# Code Book source policy

## Primary sources

Use official sources first:

- Roblox Creator Hub documentation and engine API reference: `https://create.roblox.com/docs/`
- Official Luau documentation surfaced through the Roblox Creator Hub.

Starter sources used by the proof of concept:

- [Remote events and callbacks](https://create.roblox.com/docs/scripting/events/remote)
- [RemoteEvent API reference](https://create.roblox.com/docs/reference/engine/classes/RemoteEvent)
- [RemoteFunction API reference](https://create.roblox.com/docs/reference/engine/classes/RemoteFunction)
- [Security and cheat mitigation tactics](https://create.roblox.com/docs/scripting/security/security-tactics)
- [Securing the client-server boundary](https://create.roblox.com/docs/scripting/security/client-server-boundary)
- [Luau type checking](https://create.roblox.com/docs/luau/type-checking)
- [Official Luau syntax reference](https://luau.org/syntax/)
- [Official Luau standard library reference](https://luau.org/library/)
- [Official Luau type-system reference](https://luau.org/types/)
- [Roblox events and connections](https://create.roblox.com/docs/scripting/events)
- [Implement player data and purchasing systems](https://create.roblox.com/docs/scripting/data/player-data-purchasing)
- [Developer products](https://create.roblox.com/docs/production/monetization/developer-products)

## Evidence rules

1. Cite the exact source(s) supporting a claim using source IDs in that card.
2. Mark project architecture advice as `engineering_pattern`, not as a Roblox API guarantee.
3. Include caveats when API behavior depends on execution context, replication, ownership,
   game design, Studio settings, or current platform documentation.
4. Re-check cards after major Roblox/Luau changes or when a Reviewer flags an API claim.
5. Do not elevate a community tutorial, generated answer, or model output to a source of
   truth without independently verifying it against official documentation and tests.
