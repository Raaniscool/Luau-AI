# DukeOTR desktop application

## What this is (and is not)

**DukeOTR** is a native, local Windows-oriented desktop application for working with a model
served by Ollama. It is implemented with Python's standard-library `tkinter` UI, not a browser
page styled as an application. The implementation is intentionally dependency-light so it fits
this Python repository and can later be packaged for Windows without requiring Electron, a cloud
GPU, or a web server.

The default selected local model tag is **`qwen3:4b`**, displayed as **Qwen3-4B
(`qwen3:4b`)**. DukeOTR is the application name. It is **not** a claim that Qwen3-4B was
fine-tuned, renamed, or released as a DukeOTR model. No model weights, adapters, downloads, or
training commands are included in the desktop app.

The application was developed and tested with mocked provider responses in Arena. The Arena
Python runtime does not include `tkinter`, and Arena cannot reach a user's Windows Ollama
service, so a real native window, Ollama connection/model generation, Windows look-and-feel, and
a packaged `.exe` still need to be verified on the Windows machine.

## Windows installation and first run

Use a current CPython 3.10+ installation with **Tcl/Tk** enabled (the normal python.org Windows
installer includes it). In PowerShell, open the repository root:

```powershell
Set-Location "C:\path\to\your\Luau-AI-clone"

# Required human preflight: inspect the existing local installation before any model use.
ollama list

# Optional but recommended isolated Python environment. The desktop runtime itself uses no
# third-party package; this command does not install, pull, replace, or modify an Ollama model.
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Starts a real native desktop window.
python -m desktop_app
```

`ollama list` must show the exact existing `qwen3:4b` tag before selecting it. Do **not** run
`ollama pull` merely because the repository contains no weights. The desktop app itself uses the
local HTTP API only; it does not launch or scrape a terminal, and it never automatically
downloads a model.

If the connection indicator is red, make sure the local Ollama service is running and that its
normal local API is available at `http://127.0.0.1:11434`. The app's **Settings** dialog lets a
user change that endpoint deliberately **only to another local loopback URL**. Remote/LAN/cloud
endpoints are rejected so prompt history is not redirected to another machine. No cloud service
is configured by this project.

## Architecture

```text
Native DukeOTR tkinter UI
        |
        v
DukeOTRApplicationCore
  - visible modes and prompt construction
  - local settings and conversation persistence
  - model selection / generation lifecycle
        |
        v
abstract ModelProvider
        |
        +-- OllamaProvider (current)
                |
                v
          Ollama local HTTP API
                |
                v
      selected actual local model tag (for example qwen3:4b)
```

The relevant code is deliberately split at this boundary:

| Responsibility | Location |
| --- | --- |
| Native UI and background-thread streaming | `desktop_app/ui.py` |
| UI-independent orchestration | `desktop_app/core/app_core.py` |
| Provider contract/errors | `desktop_app/core/provider.py` |
| Ollama HTTP/JSONL implementation | `desktop_app/core/ollama_provider.py` |
| Settings and local conversation storage | `desktop_app/core/storage.py` |
| Mode-specific instructions | `desktop_app/core/modes.py` |
| Existing adaptive-routing planning bridge | `desktop_app/core/builder_adapter.py` |
| Future Studio boundary, deliberately unimplemented | `desktop_app/integrations/roblox_studio.py` |

The existing repository's `scripts.lib.ollama` client remains the dependency-free local HTTP
client for data/evaluation tooling. `OllamaProvider` uses the same local-HTTP design but is an
application-specific implementation because interactive UI streaming and cancellation need an
iterator rather than an aggregated pipeline response.

### Provider behavior

The current provider calls only these Ollama endpoints:

- `GET /api/tags` discovers actual installed model tags and drives the model selector.
- `POST /api/chat` sends the current local conversation and the selected tag. It requests
  JSON-line streaming and exposes chunks in the UI as they arrive.

Before generation, it verifies the selected tag against `/api/tags`; it does not guess that
`qwen3:4b` exists, report a fabricated connection, or pull a replacement. Connection errors,
absent model tags, malformed responses, empty responses, and interrupted streams get distinct,
actionable messages. The Stop button closes the app's receipt of the local stream. If text was
already received, the app preserves that local partial answer and labels it **interrupted**;
otherwise it saves no fake assistant answer.

All model traffic is sent only to the endpoint chosen in Settings. With the default setting, that
is the same computer's loopback Ollama API. The app does not upload history to a DukeOTR service,
call a cloud API, or host a browser-facing backend.

### Model/provider transition path

The UI and application core depend on the abstract `ModelProvider` contract rather than on
Ollama-specific widgets. A future `DukeOTRProvider`, a real released `dukeotr` Ollama tag, or a
different local provider can implement `status`, `list_models`, `validate_model`, and
`stream_chat` without rewriting the chat UI, storage, or modes.

That transition must be evidence-led: a `dukeotr` tag must not be selected or described as a
DukeOTR-trained model until an actual adapter/model artifact and independent held-out evaluation
evidence exist. Until then, the selector must continue to identify `qwen3:4b` truthfully.

## Local data and privacy

By default on Windows, the app stores only its own JSON files under:

```text
%LOCALAPPDATA%\DukeOTR\
  settings.json
  conversations.json
```

On non-Windows development systems it uses `$XDG_DATA_HOME/dukeotr` or
`~/.local/share/dukeotr`. Prompt text, code, generated answers, selected model settings, and
interrupted-answer markers stay in those files. Writes are atomic where the filesystem supports
it. If a JSON file is corrupt, it is preserved as a neighboring `.corrupt` file rather than
silently overwritten.

The project does not save app history inside the Git checkout by default. Treat the local files
as sensitive because they may contain source code and project details. Deleting a conversation
from the sidebar removes it from this local history on the next atomic save; deleting files
manually is also possible while the app is closed.

## Workspace modes

The left rail provides visible mode-specific instruction context:

- **Chat** — direct local assistance.
- **Code** — generation, explanation, improvement, fixing, or refactoring, with an optional
  language/code field.
- **Review** — structured review with findings, changes, and a manual test plan.
- **Debug** — observed error, expected/actual behavior, and optional code.
- **Builder** — a first-pass implementation plan with assumptions/security/test expectations.
- **Security** — defensive trust-boundary and server-authority review for work the user is
  authorized to protect.

Responses with fenced code have native **Copy code** buttons. The responsive layout keeps the
conversation list, model/connection status, transcript, mode fields, and composer usable as the
window is resized.

Mode instructions explicitly avoid false claims about code execution, Studio access, training,
or model specialization. They also exclude held-out evaluation prompts, rubrics, answers, and
score artifacts from application generation context.

### Existing Builder / Reviewer / Fixer architecture

Builder mode does not duplicate or replace the repository's established
Builder → Reviewer/Tester → Fixer process. It reuses the existing deterministic
`classify_task()` routing policy from `scripts.lib.effort_routing` to display a **planning
summary** (risk labels, selected checks, and configured reviewer-pass count). This lookup does
not query Ollama, create a dataset record, expose Code Book/evaluation content, or claim a review
ran.

The full trace runner remains `python -m scripts.run_builder_reviewer_fixer` and must keep its
curated train-brief provenance, quality gates, Code Book controls, and held-out evaluation
isolation. An arbitrary desktop conversation is not silently converted into training material.
See `docs/FUTURE_BUILDER_VERIFIER_REVIEWER.md` and `training_factory/README.md` for that separate
workflow.

## Roblox Studio boundary

`desktop_app/integrations/roblox_studio.py` contains an abstract `RobloxStudioBridge` and an
explicitly unavailable default implementation. It is a seam for a future locally installed
Studio plugin/companion protocol, not an integration claim. Today the desktop app cannot inspect
a place, invoke Studio, insert scripts, read a DataModel, or write files into a Roblox project.

A future implementation should use an explicit, documented local protocol and require
per-operation user confirmation. It must show exactly what is sent/changed and preserve Studio's
security and permission boundaries. It must not expose a loopback bridge to arbitrary web pages
or pretend that a model response has been tested in Studio.

## Testing

The desktop core is deliberately testable without a display or Ollama service:

```powershell
python -m compileall -q desktop_app
python -m unittest tests.test_desktop_core -v
python -m unittest discover -v
```

`tests/test_desktop_core.py` uses mocked transports/providers and covers provider discovery,
request payload construction and JSONL parsing, missing models, unavailable/malformed streams,
cancellation, settings/history persistence and corrupt-file recovery, model switching, and each
mode's instructions. It does not fake a live Ollama success.

## Eventual Windows packaging

No `.exe` has been built, committed, or tested by this change. When a Windows maintainer is ready
to package after running the source version successfully, one candidate PyInstaller procedure is:

```powershell
# From the repository root, inside the chosen virtual environment.
python -m pip install pyinstaller
py -m PyInstaller --noconfirm --clean --windowed --name DukeOTR --paths . `
  --add-data "configs;configs" desktop_app\__main__.py

# Test the generated file manually on the same Windows machine.
.\dist\DukeOTR\DukeOTR.exe
```

The `configs` data inclusion keeps Builder-mode routing metadata available in a frozen app. Test
model discovery, a real `qwen3:4b` response, Stop, history persistence, settings, and all modes
before claiming the package works. Do not distribute `build/`, `dist/`, `.spec` artifacts,
Ollama model files, local conversations, adapters, or reports through Git by default.
