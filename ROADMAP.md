# DeepProfiler Roadmap

DeepProfiler is being redesigned from a training-and-profiling framework into a
**lean, pip-installable feature extractor** for microscopy images.
This document describes where the project is heading and what will be removed along the way.

## Vision

DeepProfiler's new focus is a single, well-defined job:

> **Take microscopy images (with optional cell masks) → extract deep learning features → output cytotable-compatible Parquet files.**

The package will be a thin layer over deep learning models, with seamless integration
with models hosted on [HuggingFace](https://huggingface.co/), and outputs that follow
[cytotable](https://github.com/cytomining/CytoTable) standards for downstream compatibility
with [pycytominer](https://github.com/cytomining/pycytominer) and the broader cytomining ecosystem.

---

## Milestones

### ✅ v0.3.x — Software gardening (current)
- Migrate to `pyproject.toml` and `uv`
- Publish to PyPI
- Add CI/CD via GitHub Actions
- Adopt cytomining community standards (CONTRIBUTING, CODE_OF_CONDUCT, CITATION.cff)
- Deprecation notices for functionality to be removed

### 🔜 v0.4.x — PyTorch rewrite (featurizer)
- Full rewrite in PyTorch (drop TensorFlow)
- Clean public API: `DeepProfiler.from_pretrained(...)` → `.profile(...)`
- Native HuggingFace model support (`transformers`, `timm`)
- Cell masking support: per-cell crops extracted from masks
- Cytotable-compatible Parquet output (`Metadata_*` + feature columns)
- Optimized inference: `torch.compile`, mixed precision (`torch.autocast`), multi-worker DataLoader

### 🔜 v0.5.x — Input flexibility
- Multiple input sources: numpy arrays, file paths, metadata CSVs, Parquet manifests
- Streaming large datasets without loading into memory

### 🔜 v1.0.0 — Stable release
- Stable public API
- Full documentation
- Pre-trained cytomining model weights hosted on HuggingFace

---

## What is being removed

| Functionality | Status | Removed in |
|---|---|---|
| Model training (`train`, `traintf2` commands) | ⚠️ Deprecated | v0.4.x |
| Single-cell export for training (`export-sc` command) | ⚠️ Deprecated | v0.4.x |
| Plugin system (models, crop generators, metrics) | ⚠️ Deprecated | v0.4.x |
| CometML experiment tracking | ⚠️ Deprecated | v0.4.x |
| TensorFlow backend | ⚠️ Deprecated | v0.4.x |

---

## Development philosophy: agentic AI infrastructure

This project is developed with the help of agentic AI tools — specifically
[Claude Code](https://claude.ai/code) — as a core part of the development workflow.
This is not about using AI to generate boilerplate. It is about having a capable collaborator
that understands the codebase, remembers decisions across sessions, and can execute
multi-step tasks (refactoring, writing tests, reviewing PRs) with human oversight at each step.

### Why this matters for contributors

Scientific software in biology is often maintained by small teams with limited engineering
bandwidth. Agentic AI infrastructure helps close that gap — allowing researchers to move
faster without sacrificing code quality or community standards.

Concretely, this means:

- **Reusable skills** — repeatable, reviewable AI tasks are encoded as
  [Claude Code skills](https://docs.anthropic.com/en/docs/claude-code/skills), so things
  like "review this PR", "run the test suite and summarize failures", or "check for
  deprecated API usage" can be invoked consistently by any maintainer.
- **Memory across sessions** — project context (decisions made, patterns to follow,
  known issues) is stored in structured memory files so the AI collaborator does not
  need to re-derive everything from scratch each session.
- **Hooks and automation** — routine tasks (linting before commit, updating changelogs,
  flagging deprecations) are automated via Claude Code hooks wired into the development
  environment, reducing the cognitive load on maintainers.
- **Transparent, reviewable output** — all AI-assisted changes go through the same
  PR review process as human contributions. The AI proposes; humans approve.

### Current setup

The following skills are actively used in this repository's development workflow:

| Skill | Purpose |
|---|---|
| `code-review` | Review PRs for correctness, simplification, and security |
| `security-review` | Audit changes for security issues before merge |
| `verify` | Run the app and confirm a change works end-to-end |
| `fewer-permission-prompts` | Keep the allowlist lean by auditing common read-only commands |

### Future plans for development tooling

As the codebase matures, we plan to add project-specific skills for:

- Validating that feature output shapes match cytotable schema expectations
- Running benchmark comparisons between model versions
- Automated deprecation audits across the codebase

---

## DeepProfiler as an agent-accessible tool

Beyond using AI to help *build* DeepProfiler, we want DeepProfiler itself to be
**usable by AI agents** — not just by humans at a command line.

### The problem it solves

Image-based profiling pipelines are long and multi-step: images come in, get
preprocessed, features get extracted, features get normalized, and results get
interpreted. Today, each of these steps requires a human to run a command, check the
output, and decide what to do next. This is slow and does not scale to the volume of
experiments modern biology demands.

AI agents can change this. An agent that can orchestrate a full profiling pipeline —
calling DeepProfiler to extract features, passing them to
[pycytominer](https://github.com/cytomining/pycytominer) for normalization, querying
[CytoTable](https://github.com/cytomining/CytoTable) for data transformation, and
surfacing a summary to a human for review — compresses days of work into minutes,
while keeping a human in the loop for decisions that matter.

### What we are building toward

For DeepProfiler to be useful to an AI agent, it needs to be more than just a
command-line tool. It needs:

- **A clean Python API** — so an agent can call `dp.profile(...)` programmatically
  without shelling out to a subprocess. This is a core goal of the v0.4.x rewrite.
- **Structured, predictable output** — cytotable-compatible Parquet files with a
  consistent schema mean an agent always knows what it is getting back, without
  needing to parse or guess.
- **A reusable skill** — we plan to publish a DeepProfiler skill that any AI agent
  can load and invoke. A skill is a self-contained, versioned description of how to
  use a tool: what inputs it expects, what it does, and what it returns. Skills are
  model-agnostic — the same skill definition can be used by Claude, and other agents
  that adopt the same skill standard. This means DeepProfiler becomes callable by an
  agent the same way a human calls a function: with clear inputs, clear outputs, and
  no ambiguity about what happened.

### The cytomining agent ecosystem

DeepProfiler is one piece of a larger cytomining toolchain. The long-term vision is
an agent that can orchestrate the full image-based profiling pipeline by invoking each
tool through its skill interface:

```
Images
  → DeepProfiler skill  (feature extraction)
  → CytoTable skill     (data transformation)
  → pycytominer skill   (normalization, aggregation)
  → Human review        (interpret results, approve next steps)
```

Each step produces structured output that the next tool consumes. The human stays in
the loop at meaningful decision points — not at every file conversion or parameter
choice. This is the difference between automation that replaces judgment and automation
that amplifies it.

### Human feedback as a first-class requirement

Automated pipelines in biology carry real risk: a silent error in feature extraction
can propagate into downstream conclusions without anyone noticing. We are designing
the agent interface with explicit human checkpoints — points where an agent pauses,
surfaces a summary of what it did and what it found, and waits for a human to confirm
before proceeding. This is not an afterthought; it is a design constraint we are
building in from the start.

---

## Feedback

If you depend on any functionality listed as deprecated, please
[open an issue](https://github.com/cytomining/DeepProfiler/issues) to let us know.
We want to understand active use cases before removing anything.
