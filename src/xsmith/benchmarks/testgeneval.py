"""TestGenEval (Lite) loader.

Upstream uses the HuggingFace dataset `kjain14/testgeneval` (or the Lite
variant), where each row references a SWE-bench-testbed Docker image. Full
support requires those per-repo images (django, sympy, scikit-learn, …)
which are out of scope for v0.

This loader pulls rows from the dataset, captures the target's source code
(provided in the dataset under `code_src`), and emits Target instances. The
Docker image and working directory are stashed in `target.extra_files`
under sentinel keys so the runner can consume them, but full Docker config
support is left for a follow-up.
"""

from __future__ import annotations

from xsmith.domain.target import Target

HF_DATASET = "kjain14/testgeneval"
HF_SPLIT = "test"


class TestGenEvalBench:
    name = "testgeneval"

    def __init__(self, *, dataset: str = HF_DATASET, split: str = HF_SPLIT):
        self.dataset = dataset
        self.split = split

    def load(self, *, max_targets: int | None = None) -> list[Target]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "TestGenEval requires `datasets`. Install with "
                "`uv pip install datasets`."
            ) from e

        ds = load_dataset(self.dataset, split=self.split)
        if max_targets is not None:
            ds = ds.select(range(min(max_targets, len(ds))))

        out: list[Target] = []
        for row in ds:
            module = (
                row.get("module")
                or row.get("file_path")
                or row.get("code_file")
                or "target"
            )
            source = row.get("code_src") or row.get("source") or ""
            if not source:
                continue
            target_id = f"{row.get('repo', 'unknown')}/{module}"
            out.append(
                Target(
                    target_id=target_id,
                    module_path=_filepath_to_module(module),
                    source=source,
                    extra_files={
                        "_xsmith_meta.txt": (
                            f"docker_image={row.get('docker_image', '')}\n"
                            f"working_dir={row.get('working_dir', '')}\n"
                            f"setup_code={row.get('setup_code', '')}\n"
                        )
                    },
                )
            )
        return out


def _filepath_to_module(path: str) -> str:
    """Convert a filesystem-style path to a dotted module path.

    `foo/bar/baz.py` -> `foo.bar.baz`. Leaves dotted inputs unchanged.
    """
    if "/" not in path and "\\" not in path:
        return path.removesuffix(".py")
    p = path.replace("\\", "/").removesuffix(".py")
    return p.replace("/", ".")
