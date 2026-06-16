# Agent Guide

This file provides guidance for AI agents working on or with DeepProfiler.
It covers two distinct scopes: agents that help *build and maintain* the codebase, and agents that *use* DeepProfiler as a tool in automated profiling workflows.


---

## Part 1: Agents working on this codebase

### How to work on this project

This project uses [uv](https://docs.astral.sh/uv/) for environment management and [ruff](https://docs.astral.sh/ruff/) for linting.
Always work within the uv-managed environment.

### Local commands

- Install dependencies: `uv sync --group dev`
- Run tests: `uv run pytest`
- Run lint: `uv run ruff check .`
- Run type checks (core non-deprecated code only): `uv run ty check deepprofiler/dataset/ deepprofiler/imaging/boxes.py deepprofiler/__main__.py --ignore unresolved-attribute --ignore no-matching-overload`
- Build package: `uv build`
- Run without environment setup: `uvx deepprofiler`

### Scope expectations

- **Do not modify deprecated code** unless fixing a bug that affects the current release.
  Deprecated paths are: `deepprofiler/learning/`, `deepprofiler/imaging/augmentations.py`, `deepprofiler/imaging/cropping.py`, `deepprofiler/dataset/sampling.py`, `plugins/`.
- **Do not anticipate the v0.6.x PyTorch rewrite** in v0.5.1 changes.
  Changes to non-deprecated code should be minimal and surgical.
- Type checking runs only on non-deprecated core code.
  Do not expand the `ty check` scope without discussion.
- All changes go through a pull request.
  The AI proposes; humans approve.


---

## Part 2: Agents using DeepProfiler as a tool

### Vision

Beyond using AI to help *build* DeepProfiler, we want DeepProfiler itself to be **usable by AI agents** — not just by humans at a command line.

Image-based profiling pipelines are long and multi-step: images come in, get preprocessed, features get extracted, features get normalized, and results get interpreted.
Today, each of these steps requires a human to run a command, check the output, and decide what to do next.
This is slow and does not scale to the volume of experiments modern biology demands.

AI agents can change this.
An agent that can orchestrate a full profiling pipeline — calling DeepProfiler to extract features, passing them to [pycytominer](https://github.com/cytomining/pycytominer) for normalization, querying [CytoTable](https://github.com/cytomining/CytoTable) for data transformation, and surfacing a summary to a human for review — compresses days of work into minutes, while keeping a human in the loop for decisions that matter.

### What we are building toward

For DeepProfiler to be useful to an AI agent, it needs to be more than just a command-line tool.
It needs:

- **A clean Python API** — so an agent can call `dp.profile(...)` programmatically without shelling out to a subprocess.
  This is a core goal of the v0.6.x PyTorch rewrite (`DeepProfiler.from_pretrained(...).profile(...)`).
- **Structured, predictable output** — cytotable-compatible Parquet files with a consistent schema mean an agent always knows what it is getting back, without needing to parse or guess.
- **A reusable skill** — we plan to publish a DeepProfiler skill that any AI agent can load and invoke.
  A skill is a self-contained, versioned description of how to use a tool: what inputs it expects, what it does, and what it returns.
  Skills are model-agnostic — the same skill definition can be used by Claude, and other agents that adopt the same skill standard.
  This means DeepProfiler becomes callable by an agent the same way a human calls a function: with clear inputs, clear outputs, and no ambiguity about what happened.

### The cytomining agent ecosystem

DeepProfiler is one piece of a larger cytomining toolchain.
The long-term vision is an agent that can orchestrate the full image-based profiling pipeline by invoking each tool through its skill interface:

```
Images
  → DeepProfiler skill  (feature extraction)
  → CytoTable skill     (data transformation)
  → pycytominer skill   (normalization, aggregation)
  → Human review        (interpret results, approve next steps)
```

Each step produces structured output that the next tool consumes.
The human stays in the loop at meaningful decision points — not at every file conversion or parameter choice.
This is the difference between automation that replaces judgment and automation that amplifies it.

### Human feedback as a first-class requirement

Automated pipelines in biology carry real risk: a silent error in feature extraction can propagate into downstream conclusions without anyone noticing.
We are designing the agent interface with explicit human checkpoints — points where an agent pauses, surfaces a summary of what it did and what it found, and waits for a human to confirm before proceeding.
This is not an afterthought; it is a design constraint we are building in from the start.
