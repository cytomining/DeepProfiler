# Contributing

First of all, thank you for contributing to DeepProfiler! 🎉 💯

This document contains guidelines on how to most effectively contribute to the codebase.

If you are stuck, please feel free to ask any questions or ask for help.

## Code of conduct

This project is governed by our [code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.
Please report unacceptable behavior to [cytodata.info@gmail.com](mailto:cytodata.info@gmail.com).

## Quick links

- Documentation: <https://cytomining.github.io/DeepProfiler-handbook/>
- Issue tracker: <https://github.com/cytomining/DeepProfiler/issues>
- Package dependencies: <https://github.com/cytomining/DeepProfiler/blob/main/pyproject.toml>

## Process

### Bug reporting

We love hearing about use-cases when our software does not work.
This provides us an opportunity to improve.
However, in order for us to fix a bug, you need to tell us exactly what went wrong.

When you report a bug, please tell us as much pertinent information as possible.
This information includes:

- The DeepProfiler version you're using
- The format and source of your input images
- Copy and paste two pieces of information: 1) your command and 2) the specific error message
- What you've tried to overcome the bug
- What environment you're running DeepProfiler in (OS, hardware, GPU, etc.)

Please provide this information as an issue in the repository: <https://github.com/cytomining/DeepProfiler/issues>

Please also search the issues (and documentation) for an existing solution.
It's possible we solved the bug already!
If you find an issue already describing the bug, please add a comment to the issue instead of opening a new one.

### Suggesting enhancements

We're committed to making DeepProfiler a simple, effective tool for image-based profiling.
This commitment requires open communication with our users.

We encourage you to propose enhancements by opening an issue in the repository.

First, check the documentation to see if your proposal is already implemented.
Next, check the issues (<https://github.com/cytomining/DeepProfiler/issues>) to see if someone else has already proposed it.
If you find an existing suggestion, please comment on it noting your interest.
If not, please open a new issue and clearly document the enhancement and why it would be helpful for your use case.

### Your first code contribution

Contributing code for the first time can be a daunting task.
However, in our community, we strive to be as welcoming as possible to newcomers, while ensuring rigorous software development practices.

We describe all future work as individual [GitHub issues](https://github.com/cytomining/DeepProfiler/issues).
For first time contributors we have specifically tagged [beginner issues](https://github.com/cytomining/DeepProfiler/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

If you want to contribute code that we haven't already outlined, please start a discussion in a new issue before writing any code.
A discussion will clarify the new code and reduce merge time.

### Pull requests

After you've decided to contribute code and have written it up, please file a pull request.
We follow a [forked pull request model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).
Please create a fork of the DeepProfiler repository, clone the fork, and then create a new, feature-specific branch.
Once you make the necessary changes on this branch, file a pull request to incorporate your changes into the main repository.

To ensure an efficient review process please:

1. Keep your pull request focused on _one_ specific feature or fix — small, focused pull requests move much faster than large ones.
1. Ensure your contribution passes all CI status checks (tests, linting).
1. Write a clear description of what your pull request does and why.

Pull request review and approval is required by at least one project maintainer to merge.
We will check for correctness, style, test coverage, and scope.

### Git commit messages

Please use a short phrase that describes the specific change.
For example, "Add illumination correction for 16-bit images" is much preferred to "fix stuff".
When appropriate, reference issues (via `#` plus number).

## Development

### Overview

DeepProfiler is written in Python with environments managed by [uv](https://docs.astral.sh/uv/).
We use [pytest](https://docs.pytest.org/) for local testing and [GitHub Actions](https://docs.github.com/en/actions) for automated CI.

### Getting started

1. [Install Python 3.10+](https://www.python.org/downloads/)
1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
1. Fork and clone the repository
1. Install the project with development dependencies:
   ```sh
   uv sync --extra dev
   ```

### Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Please run ruff before committing any code:

```sh
uv run ruff check .
uv run ruff format .
```

We use [Google Style Python Docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

### Testing

Run the test suite locally with:

```sh
uv run pytest
```

To generate an HTML coverage report:

```sh
uv run pytest --cov=deepprofiler tests/
```

Automated testing is performed via [GitHub Actions](https://docs.github.com/en/actions) on all pull requests.

## Publishing releases

### Versioning

We use [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning#Semantic_versioning) to distinguish between major, minor, and patch releases.

### Release process

1. Open a pull request with the changes for the release and label it appropriately.
1. On merging, update the version in `pyproject.toml` following semantic versioning.
1. Create and push a git tag matching the version: `git tag v<version> && git push origin v<version>`.
1. The [publish workflow](https://github.com/cytomining/DeepProfiler/actions/workflows/publish.yml) will automatically build and publish the package to [PyPI](https://pypi.org/project/deepprofiler/).

## Attribution

Portions of this contribution guide were adapted from [CytoTable](https://github.com/cytomining/CytoTable/blob/main/docs/source/contributing.md).
Many thanks go to the developers and contributors of that repository.

