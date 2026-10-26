"""
@author: bartulem
Cohort consolidation: the false-discovery correction across units, the conjunction test and the unit
categories.

Every per-unit result file holds metrics and p-values and no decisions, because a correction across units
cannot be made from one unit. This module makes those decisions once, over the whole cohort:

- it collects one p-value per test per unit (quiet encoding, transfer with its slope, vocal occurrence,
  vocalization identity, nested decoding, vocal gating), treating a test that did not run as missing,
  never as failed;
- it applies the Benjamini-Hochberg (or Benjamini-Yekutieli) procedure at ``significance.fdr_q`` to each
  test over the units where it ran, within each brain area unless ``data.pool_areas`` pools them;
- it forms the conjunction p-value ``max(p_quiet, p_transfer, p_identity, p_nested)`` and corrects it
  once;
- it assigns each unit one of six categories by priority, with the transformation category defined by
  the conjunction test and a positive transfer slope, and records everything else as modifiers;
- it reports, per area, the floor requirement of each test, how many units each test could not use, the
  share of units each requirement of the conjunction removes, and the correlation of the tests' z-scores.

Units recorded in one session (tested on blocks of it) are excluded from every correction unless
``significance.include_single_session_units`` admits them; they are listed either way.

Exports :func:`unit_row`, :func:`fdr_reject`, :func:`classify_unit`, :func:`correlation_matrix`,
:func:`consolidate` and the
command-line entry :func:`main`.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np
import polars as pls

from ..os_utils import atomic_output_path
from .main_neural_encoding_dispatcher import load_settings
from .neural_artifacts import build_provenance, read_unit_artifact
from .neural_cohort import build_cohort_manifest
from .neural_significance import benjamini_hochberg
from .shift_null_inference import bh_floor_requirement

#: The tests corrected across units, in report order, with the section and key each p-value is read from.
TESTS = ("quiet", "transfer", "occurrence", "identity", "nested", "gating")

#: The components of the conjunction test. Occurrence and gating are reported but not required.
CONJUNCTION_TESTS = ("quiet", "transfer", "identity", "nested")

#: The six primary categories, in priority order.
CATEGORIES = ("Transformation", "Relay", "Regime-switching", "Vocal", "Kinematic", "None")

#: The false-discovery procedures implemented.
FDR_METHODS = ("bh", "by")


def null_z(observed: float, null) -> float:
    """
    Description
    -----------
    The observed statistic's distance from the mean of its null, in null standard deviations.

    Parameters
    ----------
    observed (float)
        The observed statistic.
    null (np.ndarray | None)
        Its null draws; ``None`` or an empty or constant null gives NaN.

    Returns
    -------
    z (float)
        The z-score, or NaN when it is undefined.
    """

    if null is None:
        return float("nan")
    values = np.asarray(null, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or not np.isfinite(observed) or float(np.std(values)) == 0.0:
        return float("nan")
    return float((observed - np.mean(values)) / np.std(values))


def _section(container: dict, key: str):
    """
    Description
    -----------
    A section (or entry) of a unit's file, or None when the file does not hold it.

    A missing section is a legitimate state here -- a test that never ran for the unit -- so absence is
    returned explicitly rather than raised.

    Parameters
    ----------
    container (dict)
        A unit's file, or one of its sections.
    key (str)
        The section or entry name.

    Returns
    -------
    value (Any)
        The stored value, or None when absent.
    """

    return container[key] if key in container else None


def _ran(section) -> bool:
    """
    Description
    -----------
    Whether a stored section holds a result, as opposed to being absent or written as not run.

    Parameters
    ----------
    section (dict | None)
        One section of a unit's file.

    Returns
    -------
    ran (bool)
        False for an absent section or one carrying ``ran: False``.
    """

    return isinstance(section, dict) and not ("ran" in section and section["ran"] is False)


def _number(section: dict, key: str) -> float:
    """
    Description
    -----------
    A stored scalar as a float, NaN where it is missing or not finite.

    Parameters
    ----------
    section (dict)
        One section of a unit's file.
    key (str)
        The scalar's key.

    Returns
    -------
    value (float)
        The scalar, or NaN.
    """

    if key not in section or section[key] is None:
        return float("nan")
    value = float(section[key])
    return value if np.isfinite(value) else float("nan")


def unit_row(record: dict, artifact: dict) -> dict:
    """
    Description
    -----------
    One unit's inputs to the correction: its p-values, z-scores, effect sizes and the reason any test is
    missing.

    A test is missing when its section is absent, was written as not run, or holds no finite p-value (a
    kinematic encoding with no scoreable fold, or a transfer with no model). Missing is never failed:
    the unit then drops out of that test's correction and of anything that needs it.

    Parameters
    ----------
    record (dict)
        The unit's cohort-manifest record (``unit_uid``, ``brain_area``, ``block_split`` and the rest).
    artifact (dict)
        The unit's result file, as :func:`neural_artifacts.read_unit_artifact` returns it; empty when the
        unit was never run.

    Returns
    -------
    row (dict)
        Flat per-unit inputs: ``p_<test>``, ``z_<test>`` and ``reason_<test>`` for every test in
        :data:`TESTS`, plus the transfer slope, the occurrence response sign, the identity target, the
        gating feature and the effect sizes.
    """

    row = {"unit_uid": record["unit_uid"], "brain_area": record["brain_area"],
           "single_session": record["block_split"] is not None,
           "n_courtship_sessions": int(record["n_courtship_sessions"]),
           "n_vocal_sessions": int(record["n_vocal_sessions"]), "has_results": bool(artifact)}
    for test in TESTS:
        row[f"p_{test}"], row[f"z_{test}"], row[f"reason_{test}"] = float("nan"), float("nan"), None

    encoding = _section(artifact, "kinematic_encoding")
    if _ran(encoding):
        for test, prefix in (("quiet", "quiet"), ("transfer", "transfer")):
            p_value = _number(encoding, f"{prefix}_p")
            row[f"p_{test}"] = p_value
            row[f"z_{test}"] = null_z(_number(encoding, f"{prefix}_score"), _section(encoding, f"{prefix}_null"))
            if not np.isfinite(p_value):
                row[f"reason_{test}"] = ("no_scoreable_fold" if _section(encoding, "no_scoreable_fold")
                                         else ("no_model" if _section(encoding, "no_model") else "no_p_value"))
        row["quiet_score"] = _number(encoding, "quiet_score")
        row["transfer_score"] = _number(encoding, "transfer_score")
        row["transfer_slope"] = _number(encoding, "transfer_slope")
    else:
        row["reason_quiet"] = row["reason_transfer"] = "not_run"
        row["quiet_score"] = row["transfer_score"] = row["transfer_slope"] = float("nan")

    for test, name, effect in (("occurrence", "vocal_occurrence", "gain"),
                               ("identity", "vocalization_identity", "gain"),
                               ("nested", "nested_vocal_manifold_position_decoding", "added")):
        section = _section(artifact, name)
        if _ran(section):
            row[f"p_{test}"] = _number(section, "p")
            row[f"z_{test}"] = _number(section, "z")
            row[f"{test}_effect"] = _number(section, effect)
            if not np.isfinite(row[f"p_{test}"]):
                row[f"reason_{test}"] = "no_p_value"
        else:
            row[f"{test}_effect"] = float("nan")
            row[f"reason_{test}"] = (section["reason"] if isinstance(section, dict) and "reason" in section
                                     else "not_run")
    occurrence = _section(artifact, "vocal_occurrence")
    row["response_sign"] = occurrence["response_sign"] if _ran(occurrence) and "response_sign" in occurrence else None
    identity = _section(artifact, "vocalization_identity")
    row["identity_target"] = identity["target"] if _ran(identity) and "target" in identity else None

    gating = _section(artifact, "vocal_gating")
    if _ran(gating):
        row["p_gating"] = _number(gating, "p_unit")
        row["gating_feature"] = gating["p_unit_feature"] if "p_unit_feature" in gating else None
        row["gating_at_floor"] = bool(gating["p_unit_at_floor"]) if "p_unit_at_floor" in gating else False
        if not np.isfinite(row["p_gating"]):
            row["reason_gating"] = "no_p_value"
    else:
        row["gating_feature"] = None
        row["gating_at_floor"] = False
        row["reason_gating"] = (gating["reason"] if isinstance(gating, dict) and "reason" in gating
                                else "not_run")
    return row


def fdr_reject(p_values: np.ndarray, q: float, method: str) -> np.ndarray:
    """
    Description
    -----------
    The false-discovery rejections of one test across units.

    ``"bh"`` is the Benjamini-Hochberg step-up procedure; ``"by"`` is Benjamini-Yekutieli, the same
    procedure at ``q / sum_{i<=m} 1/i``, valid under any dependence between the units' p-values. NaN
    p-values are carried through as non-rejections and never counted in ``m``.

    Parameters
    ----------
    p_values (np.ndarray)
        One p-value per unit, NaN where the test did not run.
    q (float)
        Target false-discovery rate.
    method (str)
        One of :data:`FDR_METHODS`.

    Returns
    -------
    rejected (np.ndarray)
        Boolean mask, True where the null is rejected.
    """

    if method not in FDR_METHODS:
        msg = f"significance.fdr_method must be one of {list(FDR_METHODS)}; got {method!r}."
        raise ValueError(msg)
    values = np.asarray(p_values, dtype=np.float64)
    m = int(np.isfinite(values).sum())
    level = q if method == "bh" else q / float(np.sum(1.0 / np.arange(1, max(m, 1) + 1)))
    return benjamini_hochberg(values, level)[0]


def classify_unit(passes: dict, slope: float, conjunction: bool) -> tuple[str, dict]:
    """
    Description
    -----------
    One unit's primary category and its modifiers.

    ``passes`` maps each test to True (rejected after correction), False (tested, not rejected) or None
    (not tested). A missing test is neither a pass nor a failure, so a rule that needs a test to pass, or
    to fail, is not met by a missing one. The rules, first match wins:

    1. Transformation -- the conjunction test is rejected and the transfer slope is positive.
    2. Relay -- T+ (quiet and transfer pass, positive slope), occurrence or identity passes, nested fails.
    3. Regime-switching -- T- (quiet and transfer pass, negative slope).
    4. Vocal -- occurrence, identity or nested passes.
    5. Kinematic -- quiet passes.
    6. None.

    Transformation is defined by the conjunction rather than by each test passing on its own, so the
    category count equals the headline count. A unit that passes the four tests separately but not the
    conjunction falls through to a later rule and is flagged.

    Parameters
    ----------
    passes (dict)
        ``{test: True | False | None}`` for every test in :data:`TESTS`.
    slope (float)
        The transfer's pooled calibration slope.
    conjunction (bool)
        Whether the conjunction p-value was rejected.

    Returns
    -------
    category (str)
        One of :data:`CATEGORIES`.
    modifiers (dict)
        ``encoding_state`` (``T+``, ``T-``, ``kinematic_only`` or ``none``), ``vocal_tests_passed``,
        ``nested``, ``transfers``, ``gated`` and ``separate_pass_conjunction_fail``.
    """

    quiet, transfer = passes["quiet"] is True, passes["transfer"] is True
    t_plus = quiet and transfer and np.isfinite(slope) and slope > 0
    t_minus = quiet and transfer and np.isfinite(slope) and slope < 0
    vocal_pass = [test for test in ("occurrence", "identity") if passes[test] is True]
    if conjunction and np.isfinite(slope) and slope > 0:
        category = "Transformation"
    elif t_plus and vocal_pass and passes["nested"] is False:
        category = "Relay"
    elif t_minus:
        category = "Regime-switching"
    elif vocal_pass or passes["nested"] is True:
        category = "Vocal"
    elif quiet:
        category = "Kinematic"
    else:
        category = "None"
    separately = t_plus and passes["identity"] is True and passes["nested"] is True
    modifiers = {"encoding_state": ("T+" if t_plus else "T-" if t_minus else "kinematic_only" if quiet else "none"),
                 "vocal_tests_passed": "+".join(vocal_pass) if vocal_pass else "none",
                 "nested": passes["nested"], "transfers": transfer, "gated": passes["gating"],
                 "separate_pass_conjunction_fail": bool(separately and category != "Transformation")}
    return category, modifiers


def _passes(p_values: np.ndarray, rejected: np.ndarray) -> list:
    """
    Description
    -----------
    Per-unit pass state of one test: True, False, or None where it did not run.

    Parameters
    ----------
    p_values (np.ndarray)
        The test's p-values, NaN where it did not run.
    rejected (np.ndarray)
        The correction's rejections.

    Returns
    -------
    states (list)
        One of True, False or None per unit.
    """

    return [None if not np.isfinite(p) else bool(r) for p, r in zip(p_values, rejected, strict=True)]


def _draw_counts(artifacts: dict) -> dict:
    """
    Description
    -----------
    The most common number of null draws behind each test, for its floor requirement.

    Parameters
    ----------
    artifacts (dict)
        ``{unit_uid: result file}``.

    Returns
    -------
    draws (dict)
        ``{test: draws}``, absent for a test no unit ran.
    """

    found: dict = {test: [] for test in TESTS}
    for artifact in artifacts.values():
        encoding = _section(artifact, "kinematic_encoding")
        if _ran(encoding):
            for test in ("quiet", "transfer"):
                null = _section(encoding, f"{test}_null")
                if null is not None and np.size(null):
                    found[test].append(int(np.size(null)))
        for test, name, key in (("occurrence", "vocal_occurrence", "n_shuffles"),
                                ("identity", "vocalization_identity", "n_permutations"),
                                ("nested", "nested_vocal_manifold_position_decoding", "n_permutations"),
                                ("gating", "vocal_gating", "n_draws")):
            section = _section(artifact, name)
            if _ran(section) and key in section:
                found[test].append(int(section[key]))
    return {test: int(np.bincount(values).argmax()) for test, values in found.items() if values}


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Pearson correlation between the columns of ``values``, NaN wherever a column has no spread.

    The cross-test correlation of z-scores is how the conjunction's stringency is reported, so a test whose
    z-scores do not vary across the group gives an undefined correlation rather than a division by zero.

    Parameters
    ----------
    values (np.ndarray)
        ``(n_units, n_tests)`` z-scores, all finite.

    Returns
    -------
    correlation (np.ndarray)
        ``(n_tests, n_tests)`` correlation matrix.
    """

    centred = values - values.mean(axis=0)
    spread = np.sqrt(np.sum(centred ** 2, axis=0))
    correlation = np.full((values.shape[1], values.shape[1]), np.nan)
    varying = spread > 0
    if varying.any():
        normalised = centred[:, varying] / spread[varying]
        correlation[np.ix_(varying, varying)] = normalised.T @ normalised
    return correlation


def _area_report(frame: pls.DataFrame, q: float, draws: dict) -> dict:
    """
    Description
    -----------
    The cohort report of one correction group (one area, or the pooled cohort).

    Parameters
    ----------
    frame (pls.DataFrame)
        The group's per-unit table after correction.
    q (float)
        Target false-discovery rate.
    draws (dict)
        The most common number of draws per test.

    Returns
    -------
    report (dict)
        Per test: units tested, units rejected, units excluded by reason and the floor requirement (for
        gating, read off the Bonferroni-adjusted floor its units reach, ``floor_p``); the
        conjunction's count; the share of quiet-encoding units each conjunction requirement removes; the
        correlation matrix of the tests' z-scores; and the category counts.
    """

    report: dict = {"n_units": frame.height, "tests": {}}
    for test in [*TESTS, "conjunction"]:
        tested = frame.filter(pls.col(f"p_{test}").is_not_nan())
        entry = {"n_tested": tested.height, "n_rejected": int(frame[f"pass_{test}"].fill_null(False).sum())}
        if test != "conjunction":
            reasons = frame.filter(pls.col(f"p_{test}").is_nan())[f"reason_{test}"].fill_null("unknown")
            entry["excluded"] = dict(zip(*np.unique(reasons.to_numpy().astype(str), return_counts=True),
                                         strict=True)) if reasons.len() else {}
            entry["excluded"] = {str(k): int(v) for k, v in entry["excluded"].items()}
            if test in draws and tested.height:
                entry["n_draws"] = draws[test]
                entry["floor_requirement"] = bh_floor_requirement(tested.height, q, draws[test])
            if test == "gating":
                # A unit's gating p is Bonferroni-adjusted over its features, so it floors at
                # F / (N + 1) rather than 1 / (N + 1); the requirement is read off the floor reached.
                floored = frame.filter(pls.col("gating_at_floor"))["p_gating"].drop_nans()
                entry["floor_p"] = float(floored.min()) if floored.len() else None
                entry["floor_requirement"] = (tested.height * entry["floor_p"] / q
                                              if entry["floor_p"] is not None else None)
        report["tests"][test] = entry

    encoded = frame.filter(pls.col("pass_quiet").fill_null(False))
    if encoded.height:
        transfers = (encoded["pass_transfer"].fill_null(False) & (encoded["transfer_slope"].fill_nan(0.0) > 0))
        report["removed_by_requirement"] = {
            "n_quiet_encoding": encoded.height,
            "transfer": float(1.0 - transfers.mean()),
            "identity": float(1.0 - encoded["pass_identity"].fill_null(False).mean()),
            "nested": float(1.0 - encoded["pass_nested"].fill_null(False).mean())}
    columns = [f"z_{test}" for test in ("quiet", "transfer", "occurrence", "identity", "nested")]
    z = frame.select(columns).to_numpy().astype(np.float64)
    complete = np.isfinite(z).all(axis=1)
    report["z_correlation"] = {"tests": [c[2:] for c in columns], "n_units": int(complete.sum()),
                               "pearson": (correlation_matrix(z[complete]).tolist() if complete.sum() > 2 else None)}
    report["categories"] = {name: int((frame["category"] == name).sum()) for name in CATEGORIES}
    report["separate_pass_conjunction_fail"] = int(frame["separate_pass_conjunction_fail"].sum())
    return report


def consolidate(cohort: list, artifacts: dict, settings: dict) -> tuple[pls.DataFrame, dict]:
    """
    Description
    -----------
    Correct every test across units, test the conjunction, and classify each unit.

    Units are grouped by brain area unless ``data.pool_areas`` pools them, and each group is corrected on
    its own, so one area's count never depends on how many units of another were run. Single-session
    units enter the correction only when ``significance.include_single_session_units`` admits them; when
    excluded they are still listed, with every pass state and category left empty.

    Parameters
    ----------
    cohort (list)
        The cohort manifest (:func:`neural_cohort.build_cohort_manifest`).
    artifacts (dict)
        ``{unit_uid: result file}``; a unit without one is listed as having no results.
    settings (dict)
        Full neural-modeling settings; reads ``significance.fdr_q``, ``significance.fdr_method``,
        ``significance.include_single_session_units`` and ``data.pool_areas``.

    Returns
    -------
    table (pls.DataFrame)
        One row per unit: inputs, the per-test and conjunction pass states, the category and the modifiers.
    report (dict)
        ``{correction group: report}`` (see :func:`_area_report`), plus ``excluded_single_session``.
    """

    significance = settings["significance"]
    q, method = float(significance["fdr_q"]), significance["fdr_method"]
    include_single = bool(significance["include_single_session_units"])
    rows = [unit_row(record, artifacts[record["unit_uid"]] if record["unit_uid"] in artifacts else {})
            for record in cohort]
    for row in rows:
        conjunction = [row[f"p_{test}"] for test in CONJUNCTION_TESTS]
        row["p_conjunction"] = float(max(conjunction)) if np.isfinite(conjunction).all() else float("nan")
        row["corrected"] = include_single or not row["single_session"]
        row["group"] = "pooled" if settings["data"]["pool_areas"] else row["brain_area"]

    draws = _draw_counts(artifacts)
    out, report = [], {}
    for group in sorted({row["group"] for row in rows}):
        members = [row for row in rows if row["group"] == group and row["corrected"]]
        if not members:
            continue
        for test in [*TESTS, "conjunction"]:
            p_values = np.array([row[f"p_{test}"] for row in members], dtype=np.float64)
            for row, state in zip(members, _passes(p_values, fdr_reject(p_values, q, method)), strict=True):
                row[f"pass_{test}"] = state
        for row in members:
            category, modifiers = classify_unit({test: row[f"pass_{test}"] for test in TESTS},
                                                row["transfer_slope"], row["pass_conjunction"] is True)
            row.update({"category": category, **modifiers})
        out.extend(members)
        frame = pls.DataFrame(members, infer_schema_length=None)
        report[group] = _area_report(frame, q, draws)

    excluded = [row for row in rows if not row["corrected"]]
    for row in excluded:
        row.update({f"pass_{test}": None for test in [*TESTS, "conjunction"]})
        row.update({"category": None, "encoding_state": None, "vocal_tests_passed": None, "nested": None,
                    "transfers": None, "gated": None, "separate_pass_conjunction_fail": False})
    report["excluded_single_session"] = [row["unit_uid"] for row in excluded]
    table = pls.DataFrame(out + excluded, infer_schema_length=None)
    return table, report


def main() -> int:
    """
    Description
    -----------
    Command-line entry: build the cohort manifest from the settings, read every unit's result file from
    ``--results-directory``, consolidate, and write ``neural_cohort_table.csv`` and
    ``neural_cohort_report.pkl`` (the report, the provenance of this step and the distinct settings hashes
    found in the unit files) into ``--output-directory``.

    Parameters
    ----------
    None

    Returns
    -------
    status (int)
        Process exit status.
    """

    parser = argparse.ArgumentParser(description="Cohort-level correction and unit classification.")
    parser.add_argument("--results-directory", dest="results_directory", required=True)
    parser.add_argument("--output-directory", dest="output_directory", required=True)
    parser.add_argument("--settings-path", dest="settings_path", default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings_path)
    cohort = build_cohort_manifest(settings)
    artifacts = {record["unit_uid"]: read_unit_artifact(args.results_directory, record["unit_uid"])
                 for record in cohort}
    table, report = consolidate(cohort, artifacts, settings)
    hashes = sorted({artifact["provenance"]["settings_sha256"] for artifact in artifacts.values()
                     if artifact and "provenance" in artifact})
    output = pathlib.Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    table.drop([c for c in table.columns if table[c].dtype == pls.Object]).write_csv(output / "neural_cohort_table.csv")
    with atomic_output_path(output / "neural_cohort_report.pkl") as temporary, \
            pathlib.Path(temporary).open("wb") as handle:
        pickle.dump({"report": report, "provenance": build_provenance(settings),
                     "unit_settings_sha256": hashes}, handle)
    for group, entry in report.items():
        if group != "excluded_single_session":
            print(f"{group}: {entry['n_units']} units | categories {entry['categories']}")
    if len(hashes) > 1:
        print(f"[!] the unit files were written under {len(hashes)} different settings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
