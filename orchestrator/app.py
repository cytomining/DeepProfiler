"""Host-side Typer CLI that orchestrates the DeepProfiler pipeline.

This CLI runs on the *host*, not inside a container. CellProfiler and
DeepProfiler each run in their own container and share a mounted volume; this
orchestrator coordinates the handoff between them.

Design notes for future additions:

* Do NOT ``import deepprofiler`` here. That package is the DeepProfiler
  container runtime and pulls in heavy dependencies (TensorFlow, comet-ml, ...).
  The orchestrator stays lightweight and talks to the containers by shelling out
  to ``docker`` via :mod:`orchestrator.runner`.

* Grow by adding one sub-command module per concern and mounting it as a
  sub-Typer with ``app.add_typer(...)`` (see the CellProfiler / DeepProfiler
  stages below). Shared options (volume, mountpoint, dry-run) are resolved once
  in the root callback and handed to sub-commands via ``ctx.obj`` (an
  :class:`~orchestrator.context.AppState`).
"""

import typer

app = typer.Typer(
    help="DeepProfiler pipeline orchestrator (host-side).",
    no_args_is_help=True,
)

@app.command()
def version() -> None:
    """Print the orchestrator version."""
    typer.echo("dp-pipe 0.0.1")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
