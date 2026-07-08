# Contributing to usv-playpen

See the [Scientific Python Developer Guide][spc-dev-intro] for a general
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

This guide has two halves: **development setup** (how to run the tooling —
environment, pre-commit, tests, docs) and **conventions** (how to write code and
docs so new work reads as one system with the rest of the repo). Two meta-rules
override everything below:

1. **When editing an existing file, match its local style** even where it
   differs from these defaults (e.g. a module that uses the numpydoc
   `name : type` docstring form — see
   [Writing a function](#writing-a-function-signature--docstring)). These
   conventions describe the _dominant_ repo style for new files and settle ties;
   the neighbouring code is the tie-breaker.
2. **Never summarize or compress existing code.** Preserve original docstrings,
   whitespace, and variable names when editing. When asked to change a specific
   block, change only that block.

---

# Development setup

The project uses **uv** for everything; you don't need any other task runner.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (pip,
pipx, brew, or a single downloaded binary), then work through the `.venv` it
manages.

## Environment

```bash
uv sync                 # runtime + dev deps into .venv
uv sync --group docs    # add the Sphinx toolchain (only needed for docs work)
```

Combine extras carefully: `uv sync --extra gpu --group docs`. A bare
`uv sync --extra gpu` **prunes** the `docs` group.

## Dependencies, the lockfile, and console commands

**Adding a dependency.** Use `uv add`, which edits `pyproject.toml` _and_
re-locks in one step — don't hand-edit the dependency list and lock separately:

```bash
uv add <package>                 # runtime dep -> [project] dependencies
uv add --group test <package>    # dev-only tool -> [dependency-groups] test/dev/docs
uv add --optional gpu <package>  # an extra -> [project.optional-dependencies]
uv remove <package>              # drop one
```

Where each kind of dependency belongs:

- **`[project] dependencies`** — runtime requirements. Leave them **unpinned**
  (bare `"package"`) unless a specific version is genuinely required; pin with
  `==` for a known-good release (e.g. `spikeinterface==0.104.3`) or `>=` for a
  floor (e.g. `paramiko>=4.0.0`).
- **`[dependency-groups]`** (`test` / `dev` / `docs`; `dev` includes `test`) —
  tooling never shipped to end users. Test/dev/docs packages go here, **not** in
  `[project] dependencies`.
- **`[project.optional-dependencies]`** — extras a user opts into
  (`gpu = ["jax[cuda12]"]`), installed with `--extra gpu`.
- **A package from git rather than PyPI** goes in **`[tool.uv.sources]`** (e.g.
  `sam-2`, `sleap-anipose`).
- **Security floors for _transitive_ deps** (a Dependabot alert on something you
  don't import directly) go in **`[tool.uv] constraint-dependencies`** as
  `"<pkg>>=<patched>"`, with a dated comment saying why and when to prune —
  follow the block already there. Don't add a transitive package to
  `[project] dependencies` just to bump it.

**Updating `uv.lock`.** The lockfile **is committed** and must stay in step with
`pyproject.toml`. `uv add` / `uv remove` / `uv sync` update it automatically; to
re-resolve by hand:

```bash
uv lock                          # re-resolve after a manual pyproject edit
uv lock --upgrade-package <pkg>  # bump one package to its newest allowed version
uv lock --upgrade                # re-resolve everything (e.g. to check whether a constraint pin is now redundant)
```

Commit `pyproject.toml` and `uv.lock` **together, in the same change** — a
lockfile out of step with the manifest is a bug. (End users just run `uv sync`;
the README covers version-pinned checkouts.)

Re-syncing repeatedly grows `~/.cache/uv` (tens of GB with the torch/jax/CUDA
deps); `uv cache clean` reclaims it safely — matters on quota-limited home
directories like the cluster.

**Registering a console command.** CLI entry points live in
**`[project.scripts]`** as `kebab-name = "import.path:function"`; after
`uv sync` the name is on `PATH`:

```toml
[project.scripts]
generate-usv-masks = "usv_playpen.processing.generate_masks:generate_masks_cli"
```

The command name is **kebab-case**, the target is `module:function`, and the
function is a `click` command named `*_cli` (see
[Module & pipeline patterns](#module--pipeline-patterns)) — the only exceptions
are the GUI (`usv-playpen = "usv_playpen.usv_playpen_gui:main"`) and the
standalone `npx-meta-to-coords`.

## Pre-commit

Prepare pre-commit so commits pass the required checks:

```bash
uv tool install pre-commit   # or: brew install pre-commit on macOS
pre-commit install           # installs the commit hook
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` (check everything) without installing the hook.
**Always commit with the hooks active** so the CI **Format** job passes (see
[Git & release workflow](#git--release-workflow) for the push policy).

The hooks that actually run are formatting/hygiene, not linting: **prettier**
(`--prose-wrap=always`, so markdown/yaml is re-wrapped), **blacken-docs** (black
on python code-blocks in docs/docstrings), **nbstripout** (strips notebook
outputs), the three RST pygrep hooks (double-backticks, `::` on directives,
inline-code spacing), `name-tests-test` (test files must be `test_*.py`;
`_`-prefixed helpers exempt), `debug-statements` (no stray `breakpoint()`),
end-of-file / trailing-whitespace / merge-conflict / large-file guards, and the
`pyproject`/workflow/readthedocs schema validators. **The `ruff` and `mypy`
hooks are commented out**, so pre-commit does _not_ lint or type-check — run
those yourself: `uv run ruff check .` and `uv run mypy` (the config under
`[tool.ruff]` / `[tool.mypy]` in `pyproject.toml` is the source of truth for
both).

There are also **pre-push hooks** (a Format check + `pytest -x`) that need a
one-time opt-in: `pre-commit install --hook-type pre-push`. They mirror CI
before every push; bypass with `git push --no-verify` (see
[Git & release workflow](#git--release-workflow)).

## Testing

Run the unit checks with pytest:

```bash
uv run pytest                        # the suite
uv run pytest --cov=usv_playpen      # with a coverage report (as CI does)
```

`tests/` mirrors `src/` (see [Repository layout](#repository-layout)); a new
source module gets a test module in the mirrored directory. **Run the full suite
green before pushing** — see [Git & release workflow](#git--release-workflow).

Two settings shape how you write tests (details in
[Writing tests](#writing-tests)): pytest runs with **warnings-as-errors**
(`filterwarnings = ["error"]`) and `xfail_strict`, so new code must not emit
warnings; and there is **no enforced coverage threshold** (the Codecov upload is
non-blocking). CI runs the suite on **Python 3.12 and 3.13**, on ubuntu + macos
(Windows is excluded — torch/jax don't load their native libraries there);
**3.12–3.13** is the declared support surface.

## Building the docs

Documentation is reStructuredText under `docs/`, built with Sphinx
(`docs/conf.py`, theme `sphinx_rtd_theme` with `default_dark_mode`), notebooks
rendered by **nbsphinx** (copied to `docs/notebooks/`, gitignored, at build
time), and published to ReadTheDocs. The Sphinx toolchain is the **`docs`
dependency group** in `pyproject.toml`: `sphinx>=7.0`, `myst_parser`,
`sphinx_rtd_theme`, `sphinx-rtd-dark-mode`, `sphinx_copybutton`,
`sphinx_autodoc_typehints`, `furo`, `nbsphinx`.

```bash
uv sync --group docs
uv run python -m sphinx -b html docs docs/_build/html    # open docs/_build/html/index.html
uv run python -m sphinx -b dirhtml docs /tmp/_chk        # dirhtml builder (warning check; also what the esbonio preview uses)
```

For live in-editor preview, see the esbonio note under
[Documentation conventions](#documentation-conventions).

**Published build (ReadTheDocs)** is driven by `.readthedocs.yaml`: it installs
`uv`, runs `uv sync --group docs`, then
`python -m sphinx -T -b html docs $READTHEDOCS_OUTPUT/html`. `fail_on_warning`
is currently off, but **treat build warnings as bugs** — a clean build has none.

---

# Conventions

## Python code

- **All imports at the top of the file.** Never place `import` inside a
  function, method, or conditional; hoist any mid-file imports you find. Every
  module **must** begin with `from __future__ import annotations` — ruff
  enforces it via `isort.required-imports`.
- **Type-hint every signature.** Use modern builtin generics
  (`list[dict[str, Any]]`, `tuple[np.ndarray, np.ndarray]`, `dict | None`), not
  `typing.List` / `typing.Optional`. mypy runs `strict` over `src` + `tests`,
  and full typing is required for `usv_playpen.*` code; every `# type: ignore`
  must carry a code (`# type: ignore[arg-type]`).
- **Read dict parameters directly by key** (`d['key']`), and **never** use
  `.get()` with a default unless explicitly required — a missing key should fail
  loud, not silently substitute.
- **No `print` in library code** (ruff `T20`). User-facing output goes through
  the `message_output` callable (see
  [Module & pipeline patterns](#module--pipeline-patterns)); `print` is allowed
  only in `tests/**`.
- **Use `pathlib.Path`, not `os.path`** (ruff `PTH`); `os` is reserved for the
  few things `pathlib` can't do (`os.replace`, `os.environ`, `os.getpid`).
- **Fail loud with a specific built-in exception** (`ValueError`,
  `FileNotFoundError`, `KeyError`, `RuntimeError`, `TypeError`) — there are no
  custom exception classes. The message names the offending path / key /
  pattern, and is **assigned to a variable first, not inlined in the
  constructor** (ruff `EM`).
- **Colors are hex strings only.** Every matplotlib `color=` argument is a hex
  string (`"#202020"`), routed through a palette constant where one exists. No
  named colors (`"red"`, `"k"`, …).
- **Seed randomness explicitly** with the modern generator:
  `np.random.default_rng(seed)`, the seed passed as an argument — never global
  `np.random.seed`.
- **No dashed-banner section dividers** (`# -------- #` style). Remove them when
  found; a normal one-line comment is fine when a divider is genuinely needed.
- **Reuse what's already imported** before adding a dependency — e.g. the repo
  uses `polars` for CSV I/O; don't reach for `pandas`/`csv` when `polars` will
  do.
- **Minimize file footprint.** Don't default to new files; prefer extending an
  existing module / settings block, and justify explicitly when a new file is
  truly the right call. New source belongs in the subpackage that owns its
  concern (see [Repository layout](#repository-layout)).
- **Proactive hygiene + dead-code hunting.** During any review pass, apply
  low-risk cleanups (subprocess timeouts, import hoisting, docstring fixes)
  without asking, and sweep for dead / vestigial code (`vulture` + `ruff` +
  `grep`). Reserve questions for genuine correctness / architecture choices.
- **Performance rewrites that claim to be equivalent must be verified
  empirically** (numpy equality on representative data). Floating-point
  reordering at ~1e-16 is acceptable and should be called out.

ruff's default **line length is 88** and it selects a wide family set
(`ARG B C4 EM G I ICN NPY PD PGH PIE PL PT PTH RET RUF SIM T20 UP YTT` on top of
`E`/`F`) — canonical import aliases (`ICN`, e.g. `import numpy as np`), sorted
imports (`I`), no-`print` (`T20`), pathlib (`PTH`), and so on. Treat the
`[tool.ruff]` config as the authority rather than memorising the list.

## Writing a function (signature + docstring)

**Every function, method, and class gets a detailed docstring** with an explicit
`Description` / `Parameters` / `Returns` structure. Verbose is fine — do **not**
trim docstrings for brevity unless explicitly asked.

The canonical form (dominant across the repo — `name (type)`, type in
parentheses):

```python
def pool_group_count_matrices(
    sessions: list[dict[str, Any]],
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    One or more full-prose sentences describing what the function does and any
    non-obvious behaviour. Cross-reference other callables with
    :func:`extract_snippet_matrix` and name settings keys / dataframe columns /
    flags in ``double backticks``.

    Parameters
    ----------
    sessions (list[dict[str, Any]])
        What the argument is and how it is used.
    window_s (float)
        What it is; note the meaning of any default.

    Returns
    -------
    pooled_a, pooled_b (tuple[np.ndarray, np.ndarray])
        What is returned, and any edge-case behaviour (e.g. empty input).
    """
```

Precise rules:

- **Section headers are underlined with a row of hyphens the same length as the
  word:** `Description` → 11, `Parameters` → 10, `Returns` → 7. One blank line
  separates the three sections.
- **Parameter/return entries are `name (type)` on one line, the description
  indented beneath it.** The type may be prose where that reads better
  (`(list of pathlib.Path)`).
- **Keep the `Parameters` and `Returns` headers even when empty** — a
  no-argument function still writes `Parameters` + underline + a blank line; a
  `None`-returning function writes `Returns` + underline + `None`.
- **Module docstring** at the top of every file, opening on line 1 (`"""`) with
  an `@author: <you>` line, then a one-line purpose; keep the line when editing
  an existing file. **Exactly one blank line separates the closing `"""` from
  the first import** (`from __future__ import annotations`). Vendored /
  clean-room subtrees (`processing/masks`, `processing/qlvm_training`,
  `other/…`, `neuropixels/sglx_meta_to_coords.py`) are exempt.
- The minority `name : type` (numpydoc) form appears in `modeling/` and
  `other/cluster/`. Don't convert those files; match whichever form the file
  already uses.

## Writing tests

- **Layout & header.** `tests/` mirrors `src/`; a test file is `test_*.py`
  mirroring its source module. The module docstring is `@author: <you>`, a first
  line naming the module under test, and a `Coverage:` block; the file opens
  with `from __future__ import annotations`.
- **Names.** Functions read `test_<subject>_<expected_behavior>`; group related
  cases with plain `TestPascalCase` classes (no `__init__`, no inheritance).
  `_`-prefixed modules/helpers are **not collected** by pytest (e.g.
  `tests/modeling/_synth.py`).
- **Test docstrings are short** — a one-line _why the behaviour matters_, not
  the src `Description/Parameters/Returns` block. (Reusable `_`-prefixed helpers
  and the invariant guards keep the full docstring form.)
- **Warnings are errors** (`filterwarnings = ["error"]`). Demote an _expected_
  warning with a **narrow**
  `@pytest.mark.filterwarnings("ignore:<msg>:<Category>")` on the test/class —
  never by broadening the global filter. No custom markers exist
  (`--strict-markers` is on, so an unregistered marker errors).
- **Write only under the `tmp_path` fixture — never into `src/usv_playpen`.**
  This is enforced: `tests/test_src_integrity.py` SHA-snapshots the package and
  runs last, failing on any in-tree write. If you must exercise a package-write
  path, patch the writer.
- **Determinism.** Seed with `np.random.default_rng(seed)`. Force the headless
  matplotlib backend at import time — `import matplotlib` /
  `matplotlib.use("Agg")` then `# noqa: E402` on the following imports — and add
  an `autouse` teardown `plt.close("all")` in figure-heavy modules (keeps the
  ">20 figures" warning from tripping warnings-as-errors).
- **Isolation.** Use `monkeypatch` / the `mocker` fixture, patching the
  _module-local_ reference (e.g. `dispatcher.json.load`), and redirect
  settings/JSON loads rather than editing shipped config.
- **Skip, don't fail, on a missing external binary or optional dep.** Detect
  with `shutil.which(...)` + `@pytest.mark.skipif(reason=...)`, or
  `pytest.importorskip`.
- **Synthetic data** lives in a `_`-prefixed non-test module: deep-copy the
  shipped JSON and override only the tiny-data knobs, write the _exact_ on-disk
  contract the production code expects, and return the created paths.
- **Assertions.** `pytest.approx` for floats, `numpy.testing` / `np.allclose`
  for arrays, and `pytest.raises(..., match=<regex>)` for errors.
- **Keep the four invariant guards green:** `test_src_integrity` (no test
  mutates the package), `test_docs_notebooks` (every notebook is referenced in
  `Notebooks.rst`), `test_notebooks_static` (every notebook cell parses and has
  no undefined names), and `test_package` (installed metadata version ==
  `usv_playpen.__version__`).

## Notebooks

- **Cell 1 = all imports. Cell 2 = all settings / parameters** — _unless_ a
  notebook deliberately disperses per-section parameters (each section cell
  defining its own knobs inline at its top). Centralize when sections share one
  loaded object; disperse when sections are independent. Either way, nothing
  downstream silently redefines a setting. (In marimo, `_`-prefixed names are
  cell-local.)
- Notebooks are normalized by **nbstripout** on commit (outputs stripped). Run
  `pre-commit` before committing a notebook so the CI Format gate stays green.
- Package notebooks live in `src/usv_playpen/notebooks/` and are catalogued in
  `docs/Notebooks.rst`; `tests/test_docs_notebooks.py` enforces every notebook
  is referenced, and `tests/test_notebooks_static.py` statically checks that
  every code cell parses and references no undefined name. Keep both green.

## Repository layout

New code goes into the subpackage that owns its concern — don't create a sibling
top-level module for something an existing package covers.

```
src/usv_playpen/
  analyses/            analysis engines (e.g. neuronal_coactivity_engine, mixture_model_utils)
  modeling/            modeling pipelines, consolidators, SLURM dispatchers
  neuropixels/         Neuropixels / histology helpers (anatomy_converter, sglx_meta_to_coords)
  processing/          data processing (audio, SAM2 masks, acoustic features)
  recording/           acquisition (Windows-only rig control)
  visualizations/      figure / render code (make_*.py, plot_style, modeling_plots)
  notebooks/           the Jupyter + marimo notebooks
  other/               cluster scripts (HPSS, SLURM), kilosort runner
  _parameter_settings/ the shipped *_settings.json (analyses/visualizations/modeling/…)
  _config/             behavioral_experiments_settings.toml
  fonts/  img/         static assets
  cli_utils.py  os_utils.py  time_utils.py  yaml_utils.py  send_email.py
  usv_playpen_gui.py   the GUI entry point
```

- **`tests/` mirrors `src/`** one-to-one: `tests/analyses/`, `tests/modeling/`,
  `tests/neuropixels/`, `tests/processing/`, `tests/recording/`,
  `tests/visualizations/`, plus `tests/foundation/` (os/time/yaml utils) and the
  repo-level guards `test_docs_notebooks.py`, `test_notebooks_static.py`,
  `test_package.py`, `test_src_integrity.py`. A new source module gets a test
  module in the mirrored directory.
- **Settings are data, not code.** The `*_settings.json` files under
  `_parameter_settings/` are the single source of truth for tunables; code reads
  them, docs mirror them (see
  [Documentation conventions](#documentation-conventions)).

## Module & pipeline patterns

The house shape to match when writing a new pipeline or CLI command:

- **User messaging via `message_output: Callable = print`.** Thread a
  `message_output` parameter through classes/functions and store it as
  `self.message_output`; emit all user-facing text through it — not `logging`,
  not a bare `print` (this is the `T20` escape hatch). Email notifications go
  through `send_email.Messenger`, and a failed notification is reported through
  `message_output`, never raised (a broken email can't abort a run).
- **Worker-init signature.** Pipeline classes take
  `(<name>_settings_dict: dict = None, root_directory(ies) = None, message_output=None)`;
  when the settings dict is `None`, load the JSON itself via
  `pathlib.Path(__file__).parent.parent / '_parameter_settings/<name>.json'` —
  never a hardcoded absolute path. Settings basenames are passed as bare strings
  (`'analyses_settings'`); the loader appends the extension.
- **One verb-named public method** per pipeline class (`calculate_*`,
  `save_*_to_file`, `create_*`) — not a generic `.run()`; there is no shared
  base class.
- **Cross-OS share paths go through `os_utils.configure_path(...)`** (cluster
  submission: `to_cluster_path(...)`). Never hardcode a mount root
  (`/mnt/falkner`, `F:\`, …) — they come from the TOML `lab_shares` table.
  Canonical call form: `pathlib.Path(configure_path(p))`.
- **Publish precious outputs atomically** via `os_utils.atomic_output_path`
  (temp sibling + `os.replace`), not by writing straight to the final path.
- **"Pick one file" uses `os_utils.first_match_or_raise` /
  `newest_match_or_raise`** (deterministic sort + a named `FileNotFoundError`),
  not `sorted(glob)[0]`.
- **CLI = `click`.** Entry points are functions named `*_cli`, registered in
  `pyproject.toml [project.scripts]` as `kebab-name = "module:func_cli"`;
  options default to `None` / `required=False` so unset flags fall through to
  the JSON defaults, and command-line overrides are merged with
  `modify_settings_json_for_cli(ctx, provided_params, settings_dict='<name>')`.
  Edit nested settings through `cli_utils.find_nested_key_paths` /
  `set_nested_value_by_path` (which raise on a typo'd path), not manual dict
  digging.

Reference implementations: `analyses/analyze_data.py`, `cli_utils.py`,
`os_utils.py`, `send_email.py`.

## Documentation conventions

_(To build the docs, see [Building the docs](#building-the-docs).)_ The docs are
**lab-internal first** — assume the reader has the rig / cluster / falkner
mounts and knows the domain terms. Genuinely lab-specific operational detail
(hardware, mounts, account paths) is allowed, but should read as "operational"
rather than as a universal requirement.

**Page structure & RST hygiene:**

- **One H1 per page** (the title). Consistent underline hierarchy across all
  pages: `=` H1, `-` H2, `~` H3, `^` H4. Every underline is **at least as long
  as** the title text (equal length is the norm).
- **User Guide order must match the `index.rst` toctree:** Requirements → Record
  → Process → Analyze → Visualize → Neuropixels → Modeling → Notebooks → CLI.
  Each page is a step-by-step guide to that GUI tab / pipeline stage, with the
  relevant settings / CLI shown **in context**. Keep the `index.rst` intro's
  prose list of sections in sync with the toctree.
- **No numbers in section / subsection headings.** Titles are plain noun / verb
  phrases (`Pre-alignment export`, not `3. Pre-alignment export`) — Sphinx +
  page order convey sequence, and baked-in numbers drift. **The one exception**
  is the `Notebooks.rst` per-notebook walkthroughs (below): there, inline **bold
  run-in labels** carry numbers that mirror each notebook's own cell numbering
  so the reader can map doc ↔ notebook.
- **RST hygiene (pre-commit-enforced):** two-backtick ` `code` ` for anything
  code-like, directives end with `::`, correct inline-code spacing. Build
  warning-clean before pushing.

**`Notebooks.rst` (per-notebook walkthroughs):** the single detailed home for
the package notebooks; topical pages link here rather than restating. Per
notebook:

- **One H2 section**, titled in sentence case for what the notebook does. An
  intro paragraph names the `**notebook_name.ipynb**` in bold and says what it
  produces and from which inputs.
- **Walk the cells as bold run-in labels** — `**Label.** prose …` — rather than
  H3 sub-headings (which clutter the sidebar). Number the labels
  (`**3. Pre-alignment export.**`) to mirror the notebook's own cell numbering;
  leave setup cells unnumbered (`**Imports.**`).
- **Mirror each cell as a `.. code-block:: python`** with the parameters shown.
  `blacken-docs` runs `black` on every python code-block, so it must be valid
  Python: **no IPython magics** (`%load_ext autoreload`, `%autoreload`) inside a
  code-block — describe them in prose instead.
- **Gloss every abbreviation at first use** (`periaqueductal gray (PAG)`), then
  use the short form. **British spelling in prose** (`behavioural`), but keep
  code identifiers as written (`behavioral`).
- **End the section with a `Source:` link** to the notebook on GitHub.

**Settings keys, CLI flags, experimenter paths:**

- Name every settings key / CLI flag / dataframe column / function in ` `double
  backticks` `.
- A documented settings block must **match** the shipped `*_settings.json` (same
  keys, representative values). When a key/flag is renamed in code, fix the docs
  in the **same** change — a stale key/flag is a doc _bug_, not a style nit.
- **Processing-pipeline CLI commands expose their _whole_ settings block, and
  document every flag.** Every key in a command's `processing_settings.json`
  block is a `click.option` whose second positional arg is the bare setting key
  (e.g. `@click.option('--n-epochs', 'n_epochs', ...)`), and the command passes
  `block='<its block>'` to `modify_settings_json_for_cli` so an override always
  writes to **that** command's block — never the first global key match (keys
  like `n_epochs` / `batch_size` / `latent_dim` recur across blocks, so an
  unscoped write lands in the wrong one). Adding a settings key means adding its
  flag in the same change. In `CLI.rst`, document such a command as `usage` →
  `required arguments` → `optional arguments` with **every** `--flag` described
  (the `usage` line enumerates them all, wrapped like the `generate-viz` block);
  do **not** enumerate the settings-JSON keys separately, list their defaults,
  or add a `(full parameters under processing_settings.json -> ...)` pointer —
  every parameter is a real flag, so it is simply listed as one.
- **Experimenter paths:** the shipped `*_settings.json` carry a literal
  experimenter directory (`/mnt/falkner/Bartul/...`), and the code re-keys it to
  the active experimenter at runtime (`os_utils.rebase_experimenter_in_paths` —
  GUI selection / TOML `experimenter` / cluster `EXPERIMENTER_ID`). **Write
  example paths the way the settings ship them — literal
  `/mnt/falkner/Bartul/...`** — and state **once** (where experimenter paths
  first appear) that they auto-re-key to the reader's own experimenter. Session
  **root** / **arena** dirs are user-entered — show them as literal example
  paths too. Use the linux mount form (`/mnt/falkner/...`), consistent with the
  settings files.
- **Media:** screenshots / gifs live in `docs/media/`, named
  `<page>_step_<n>.<ext>`, referenced with a consistent image/figure directive.

**Local live preview (esbonio, optional):** in-editor live preview uses the
**Esbonio** VS Code extension (`swyddfa.esbonio`), which is in the `docs`
dependency group (`uv sync --group docs` installs it into `.venv`). Point
esbonio + Python at `.venv` in `.vscode/settings.json` (gitignored):

```json
{
  "python.defaultInterpreterPath": "/ABSOLUTE/PATH/TO/usv-playpen/.venv/bin/python",
  "esbonio.sphinx.pythonCommand": [
    "/ABSOLUTE/PATH/TO/usv-playpen/.venv/bin/python"
  ],
  "python.terminal.activateEnvironment": false,
  "python.terminal.activateEnvInCurrentTerminal": false
}
```

- **Use an ABSOLUTE path, never `${workspaceFolder}`, in
  `esbonio.sphinx.pythonCommand`** — esbonio's own resolver doesn't know
  `${workspaceFolder}` and crashes with `Undefined variable: 'workspaceFolder'`,
  blanking the preview.
- **Blank / unstyled / light preview** is almost always a stale esbonio build
  cache (`~/Library/Caches/esbonio/<hash>/`):
  **`rm -rf ~/Library/Caches/esbonio` then Reload Window.** Confirm a clean CLI
  build is healthy first with the `dirhtml` command above, then
  `grep -c 'theme.css\|dark_mode' /tmp/_chk/index.html`.
- **Known limitation:** `sphinx_rtd_dark_mode` is unmaintained; dark mode
  **renders** (via `default_dark_mode = True`) but its jQuery-based toggle
  button does nothing under `sphinx_rtd_theme` 3.x. A theme migration (e.g.
  `furo`) would remove this.

## Git & release workflow

- **Commit _with_ pre-commit hooks active** (see [Pre-commit](#pre-commit)) so
  the CI **Format** job passes. **Push with `--no-verify`** — the pre-push hook
  runs the full pytest suite and hangs in sandboxed environments.
- **Run the full test suite before pushing**
  (`python -m pytest -p no:cacheprovider -q`) and push only when green.
- **Stage explicitly** (`git add -u` / named paths) — **never `git add -A`**;
  never commit scratch / `REVIEW_*` / draft files.
- **Release titles are bare version strings**
  (`gh release create --title "vX.Y.Z"`); all narrative goes in the body.

## Project-specific domain conventions

- **Experimenter-scoped paths.** The `*_settings.json` files ship literal
  `Bartul` paths; `os_utils.rebase_experimenter_in_paths` re-keys them to the
  active experimenter in both the **GUI** (front-page selection) and the
  **headless CLI** (`cli_utils.modify_settings_json_for_cli`, keyed off
  `os_utils._host_experimenter`). Session **root** / **arena** dirs stay literal
  (typed per run). `modeling_settings.json` is **not** auto-re-keyed (edit its
  paths per checkout). Cluster scripts set `EXPERIMENTER_ID` at the top and
  export it into the generated SLURM job; `_host_experimenter` reads that env
  var in preference to the TOML `experimenter`, so it drives both the bash
  scratch paths and the CLI path re-keying with no per-experimenter TOML edit.
- **Mixture-model naming.** Use `mixture_model_*` for things that handle
  **both** Gaussian and Student-t mixtures (settings keys, the
  inter-USV-interval sweep, the `mixture_model_fits` HDF5 group). Reserve the
  `gmm_*` prefix for the Gaussian-only helpers in
  `analyses/mixture_model_utils.py`, a deliberate parallel surface to the
  `t_mixture_*` functions (`gmm_icl` ↔ `t_mixture_icl`, etc.) — do not rename
  those. Mirror this in prose: "mixture model" for the gauss-or-t machinery,
  "GMM" / "Gaussian" only for the genuinely Gaussian-only path.
- **Behavioral videos must be ≤ 4096 px** (width or height × dpi) for
  PowerPoint; wider frames glitch the PPT hardware decoder.
- **Taxonomy is column-driven, not hardcoded.** Category-id settings (e.g.
  `onset_target_category`) are bare integers; the taxonomy/column they index
  into comes from `usv_category_column_name` (`df[column] == id`).
