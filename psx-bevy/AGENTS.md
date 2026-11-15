# psx-bevy: ours to change

This crate began as the psx toolkit's `psx-bevy` template (psx 0.3.8) and has been this
project's own since 2026-11-15: it no longer follows the template, and `psx template update`
leaves it alone (`.psx-template.toml` lists none of it). Change it as the game needs.

It holds the PS1 renderer (`src/psx/`: the PS1 material, its shaders, the screen), the
launcher, controls and the dev MCP. The PS1 look's machinery belongs here; what is specific to
this game's world still goes in `engine/` or `game/`.
