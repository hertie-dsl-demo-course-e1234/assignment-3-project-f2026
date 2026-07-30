"""HIDDEN structural checks for the group project - run faculty-side, once per team repo.

These are a completeness floor, not a grade: they confirm the submission has the required
parts, that the placeholders were replaced, and that the work is reproducible in principle.
Everything about whether the project is any *good* is hand-marked against the rubric in the
brief.

Two deliberate design choices:

- The checks run from the repository root (the grading runner sets the submission clone as
  the working directory), so they use plain relative paths.
- `src/pipeline.py` is inspected with `ast`, never imported. A real project imports pandas
  and scikit-learn; the grading runner has only pytest and nbconvert, so importing the
  submission would fail for reasons that have nothing to do with the student's work.

Seven test cases, one mark each (grading.yml: max_auto: 7).
"""

import ast
import re
from pathlib import Path

import pytest

REPORT = Path("REPORT.md")
SRC = Path("src")
REQUIRED_HEADINGS = ["Question", "Data", "Method", "Results", "Limitations",
                     "Model card", "Contributions"]
MODEL_CARD_FIELDS = ["Intended use", "Out-of-scope use", "Data", "Performance",
                     "Known limitations", "Monitoring"]
PLACEHOLDER = re.compile(r"<[^<>\n]{3,80}>|\bTODO\b|\bFIXME\b|\bTBD\b|_your answer_")
MIN_WORDS = 800


@pytest.fixture(scope="module")
def report_text():
    if not REPORT.is_file():
        pytest.fail("REPORT.md is missing from the repository root")
    return REPORT.read_text(encoding="utf-8", errors="replace")


def source_files():
    if not SRC.is_dir():
        return []
    return [p for p in SRC.rglob("*") if p.suffix.lower() in {".py", ".r", ".rmd"}]


def test_report_exists_and_has_substance(report_text):
    """A report of at least MIN_WORDS words - the brief asks for 2,000-2,500."""
    words = len(report_text.split())
    assert words >= MIN_WORDS, f"REPORT.md has {words} words; expected at least {MIN_WORDS}"


def test_report_has_every_required_heading(report_text):
    """The seven sections from the brief, by name."""
    missing = [h for h in REQUIRED_HEADINGS
               if not re.search(rf"^#{{1,4}}\s*\d*\.?\s*{re.escape(h)}", report_text,
                                re.MULTILINE | re.IGNORECASE)]
    assert not missing, f"missing section heading(s): {missing}"


def test_report_placeholders_replaced(report_text):
    """No `<like this>`, `_your answer_`, TODO or TBD left in the submitted report."""
    leftovers = sorted(set(PLACEHOLDER.findall(report_text)))
    assert not leftovers, f"unreplaced placeholders in REPORT.md: {leftovers[:5]}"


def test_model_card_fields_present(report_text):
    """Every model-card field the brief lists, and an interval somewhere in the report."""
    missing = [f for f in MODEL_CARD_FIELDS if f.lower() not in report_text.lower()]
    assert not missing, f"model card is missing: {missing}"
    assert re.search(r"\[\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*\]", report_text), \
        "no interval of the form [lo, hi] found - the brief requires one next to the metric"


def test_source_present_and_defines_run_pipeline():
    """src/ has code, and something in it defines a `run_pipeline` entry point."""
    files = source_files()
    assert files, "no .py / .R / .Rmd files found under src/"

    python_files = [p for p in files if p.suffix == ".py"]
    if not python_files:
        # An R project: look for a function assignment instead.
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)
        assert re.search(r"run_pipeline\s*(<-|=)\s*function", text), \
            "no `run_pipeline <- function(...)` found in src/"
        return

    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            pytest.fail(f"{path} does not parse as Python: {exc}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                    node.name == "run_pipeline":
                if _is_stub(node):
                    pytest.fail(f"{path}: `run_pipeline` is still the starter stub "
                                "(its body only raises NotImplementedError)")
                return
    pytest.fail("no `def run_pipeline(...)` found in any src/*.py file")


def _is_stub(node) -> bool:
    """True when a function body is nothing but a docstring and `raise NotImplementedError`."""
    body = [n for n in node.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str))]
    return len(body) == 1 and isinstance(body[0], ast.Raise) and (
        "NotImplementedError" in ast.dump(body[0]))


def test_dependencies_are_pinned():
    """A manifest exists, and a Python one pins at least one version."""
    manifests = [Path(name) for name in
                 ("requirements.txt", "environment.yml", "renv.lock", "pyproject.toml")
                 if Path(name).is_file()]
    assert manifests, ("no dependency manifest found (requirements.txt, environment.yml, "
                       "renv.lock or pyproject.toml)")
    if Path("requirements.txt").is_file():
        lines = [ln.strip() for ln in Path("requirements.txt").read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        assert lines, "requirements.txt is empty"
        assert any(re.search(r"[=><~]=?\s*\d", ln) for ln in lines), \
            "no pinned versions in requirements.txt - an unpinned environment is not reproducible"


def test_a_seed_is_fixed():
    """A fixed seed somewhere in the source - reproducibility is 3 of the 25 marks."""
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in source_files())
    assert re.search(r"random_state|default_rng|SEED|set\.seed|np\.random\.seed", text), \
        "no fixed random seed found in src/ (random_state=, default_rng(), set.seed())"
