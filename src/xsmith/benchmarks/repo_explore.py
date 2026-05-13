"""RepoExploreBench loader.

Upstream lists ~100 real-world Python modules across 9 popular libraries.
For v0 we ship a small, balanced subset (one or two modules per repo) so
the smoke benchmark runs quickly. The loader reads source via `inspect` —
the listed packages are all installed by `Dockerfile.runner`, and most are
also commonly present in any modern Python venv.

A target's `source` is the verbatim file text. `goals` is left empty —
the Explorer's evaluator populates it via `enumerate_goals()` before
exploration begins (and the smoke-test evaluator mounts that source into the
sandbox).
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

from xsmith.domain.target import Target


@dataclass(frozen=True)
class _Spec:
    module: str
    repo: str
    description: str


# A curated subset of upstream RepoExploreBench, chosen for: (a) installability
# without extra system deps, (b) variety, (c) small-enough that K=5 generators
# don't make the smoke test take forever. The order roughly mirrors upstream.
DEFAULT_TARGETS: list[_Spec] = [
    _Spec("click.types", "click", "Type system with conversion and validation."),
    _Spec("click.utils", "click", "File handling, lazy loading, env helpers."),
    _Spec("requests.utils", "requests", "URL parsing, encoding detection."),
    _Spec("requests.cookies", "requests", "Cookie jar with domain/path matching."),
    _Spec("flask.config", "flask", "Config object with env loading and defaults."),
    _Spec("flask.helpers", "flask", "URL generation, flashing, send_file."),
    _Spec("httpx._urls", "httpx", "URL parsing and manipulation."),
    _Spec("rich.style", "rich", "Style parsing and combining."),
    _Spec("jinja2.utils", "jinja2", "Template utilities."),
    _Spec("starlette.datastructures", "starlette", "Headers, FormData, MutableHeaders."),
]


class RepoExploreBench:
    name = "repo_explore"

    def __init__(self, specs: list[_Spec] | None = None):
        self.specs = specs or list(DEFAULT_TARGETS)

    def load(self, *, max_targets: int | None = None) -> list[Target]:
        specs = self.specs[:max_targets] if max_targets else list(self.specs)
        out: list[Target] = []
        for spec in specs:
            try:
                module = importlib.import_module(spec.module)
                source = inspect.getsource(module)
            except (ImportError, OSError, TypeError) as e:
                raise RuntimeError(
                    f"Failed to load source for {spec.module!r} ({spec.repo}): {e}. "
                    f"Install the package (e.g. `uv pip install {spec.repo}`)."
                ) from e

            out.append(
                Target(
                    target_id=f"{spec.repo}/{spec.module}",
                    module_path=spec.module,
                    source=source,
                    entrypoints=_top_level_names(module),
                )
            )
        return out


def _top_level_names(module) -> list[str]:
    """Return up to 25 top-level public names from `module`."""
    names = [n for n in dir(module) if not n.startswith("_")]
    return sorted(names)[:25]
