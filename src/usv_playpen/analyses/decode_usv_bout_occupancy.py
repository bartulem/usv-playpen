"""
@author: bartulem
Decode whether a moment sits inside a vocal bout from simultaneously recorded spiking.

The session is cut into bins of `bin_seconds`. Bouts are found on the USV sequence FIRST, as
maximal runs of the emitter's USVs whose consecutive silences all stay under the inter-bout
threshold, keeping runs of at least `min_vocalizations` USVs. Bins then follow the bouts rather
than the other way round:

* a POSITIVE bin is laid from a bout's own first USV start, stepping forward by `bin_seconds`
  while it stays inside that bout's extent, and contributes only its SILENT time -- so no window
  the decoder reads ever contains a vocalization. It needs at least `min_kept_seconds` of silent
  time to count at all; without that floor a bin can enter the population vector carrying a couple
  of milliseconds of counting time, which is noise.
* a NEGATIVE bin comes from the global lattice and must hold no vocalization and touch no bout.

Bouts are grouped from the emitter's pure USVs, but SILENCE is not class-specific: the time a
positive bin keeps and the bins a negative may use must be free of every vocalization of the
emitter -- pure USVs, pure squeaks, overlap segments (a USV and a squeak in one segment) and
unscored segments alike -- and of every vocalization assigned to no animal. Checking silence
against pure USVs only would let a squeak, or an overlap segment, which holds a USV, sit inside a
"silent" negative bin or inside a positive bin's kept time.
* everything else is DROPPED. A bin holding a single USV cannot be assigned: a 120 ms USV
  followed by a 110 ms silence is one USV per bin and squarely inside a bout, so a USV count
  cannot say whether such a bin is bout time or not.

Each bin becomes a population vector of raw spike counts, three contiguous thirds rotate as train
and test, and a test bin takes the label of the training bin its vector correlates with most, ties
broken at random. The sweep decodes with increasingly large random subsets of the area's units, so
the curve says how much of the population the signal needs rather than only whether the whole
population carries it. Accuracy is balanced across the two classes, since the negatives outnumber
the positives several times over and a raw proportion would be carried by the majority.

PER-UNIT Z-SCORING MUST NOT BE INTRODUCED HERE. Under the shift null a positive bin's counts are a
fixed fraction of a negative bin's for EVERY unit, because it contributes less counting time.
Pearson correlation is scale-invariant, so that uniform factor is invisible to this decoder and the
unequal exposure costs nothing. Standardizing each unit first turns the uniform factor into a
per-unit offset that varies across units, which correlation CAN read: measured on one session it
lifted the 415-unit null from 0.509 to 0.751. The exposure difference is harmless; only z-scoring
makes it harmful.

The inter-bout threshold is the same construction `build_naturalistic_usv_repository` uses -- the
emitter sex's log-interval mixture component at `ibi_z_score` standard deviations -- and is NOT the
`bout_quiet_seconds` of `compute_neuronal_tuning_curves`. That one is a VMI baseline-window
parameter whose value also happens to group USVs; this one describes the respiratory rhythm. They
are different objects and both are correct for their own analysis.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import datetime

import numpy as np
import polars as pls
from tqdm import tqdm

from ..os_utils import atomic_output_path, resolve_experimenter_path
from ..time_utils import is_gui_context, smart_wait
from ._usv_io import unique_match_or_none
from .compute_neuronal_tuning_curves import (
    _circular_shift_spike_times,
    ibi_bout_extents,
    ibi_bout_threshold,
    load_vocal_inputs,
)


def free_intervals(lo: float, hi: float, starts: np.ndarray, stops: np.ndarray) -> list:
    """
    Description
    -----------
    The sub-intervals of ``[lo, hi)`` that no vocalization covers.

    Parameters
    ----------
    lo, hi (float)
        Bin bounds in seconds.
    starts, stops (np.ndarray)
        Vocalization onsets and offsets in seconds, sorted by onset.

    Returns
    -------
    free (list)
        The uncovered ``(start, stop)`` sub-intervals, in time order.
    """

    hits = np.flatnonzero((stops > lo) & (starts < hi))
    free, cursor = [], lo
    for i in hits:
        if starts[i] > cursor:
            free.append((cursor, min(float(starts[i]), hi)))
        cursor = max(cursor, float(stops[i]))
        if cursor >= hi:
            break
    if cursor < hi:
        free.append((cursor, hi))
    return [(a, b) for a, b in free if b > a]


def bout_occupancy_bins(bin_seconds: float, starts: np.ndarray, stops: np.ndarray,
                        silence_starts: np.ndarray, silence_stops: np.ndarray,
                        duration_seconds: float, ibi_threshold: float, min_vocalizations: int,
                        min_kept_seconds: float) -> tuple[list, np.ndarray, np.ndarray,
                                                          np.ndarray]:
    """
    Description
    -----------
    Per bin, the sub-intervals whose spikes count, its label, its midpoint and its bout position.

    Positives are anchored to each bout's own first USV start and keep only their silent time;
    negatives come from the global lattice and must hold no vocalization and touch no bout;
    anything else is dropped. Bouts come from ``starts`` / ``stops`` (the emitter's pure USVs);
    silence is judged against ``silence_starts`` / ``silence_stops`` (every vocalization of the
    emitter, any class, plus those assigned to no animal). See the module docstring for why a
    single-USV bin cannot be assigned.

    Parameters
    ----------
    bin_seconds (float)
        Bin width in seconds.
    starts, stops (np.ndarray)
        Emitter USV onsets and offsets in seconds, sorted by onset; bouts are grouped from them.
    silence_starts, silence_stops (np.ndarray)
        Onsets and offsets in seconds, sorted by onset, of every vocalization silence must be
        free of (the emitter's of any class and the unassigned ones); a superset of ``starts``.
    duration_seconds (float)
        Session length in seconds.
    ibi_threshold (float)
        A silence at least this long ends a bout, in seconds.
    min_vocalizations (int)
        Fewest USVs a run needs to count as a bout.
    min_kept_seconds (float)
        Fewest seconds of silent time a positive bin needs to count.

    Returns
    -------
    spans, labels, mids, positions (tuple[list, np.ndarray, np.ndarray, np.ndarray])
        Per surviving bin, its kept ``(start, stop)`` intervals, its 0/1 label, its midpoint in
        seconds, and its one-based index within its bout (0 for negatives).
    """

    bout_lo, bout_hi = ibi_bout_extents(starts, stops, ibi_threshold, min_vocalizations)
    spans, labels, mids, positions = [], [], [], []

    for lo_b, hi_b in zip(bout_lo, bout_hi):
        step = 0
        while lo_b + (step + 1) * bin_seconds <= hi_b + 1e-9:
            lo = lo_b + step * bin_seconds
            free = free_intervals(lo, lo + bin_seconds, silence_starts, silence_stops)
            if sum(b - a for a, b in free) >= min_kept_seconds:
                spans.append(free)
                labels.append(1)
                mids.append(lo + 0.5 * bin_seconds)
                positions.append(step + 1)
            step += 1

    for k in range(int(duration_seconds / bin_seconds)):
        lo, hi = k * bin_seconds, (k + 1) * bin_seconds
        if np.any((silence_stops > lo) & (silence_starts < hi)):
            continue
        if bout_lo.size and np.any((bout_hi > lo) & (bout_lo < hi)):
            continue
        spans.append([(lo, hi)])
        labels.append(0)
        mids.append(0.5 * (lo + hi))
        positions.append(0)

    return (spans, np.asarray(labels, dtype=int), np.asarray(mids, dtype=float),
            np.asarray(positions, dtype=int))


def flatten_spans(spans: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Kept sub-intervals as flat arrays, with the bin each belongs to.

    Counting over flat arrays is what makes the null affordable -- two `searchsorted` calls per
    unit per draw rather than a loop over bins.

    Parameters
    ----------
    spans (list)
        Per bin, its list of kept ``(start, stop)`` intervals.

    Returns
    -------
    starts, ends, owner (tuple[np.ndarray, np.ndarray, np.ndarray])
        Interval starts and ends, and the bin index each belongs to.
    """

    starts, ends, owner = [], [], []
    for i, gaps in enumerate(spans):
        for a, b in gaps:
            starts.append(a)
            ends.append(b)
            owner.append(i)
    return (np.asarray(starts, dtype=float), np.asarray(ends, dtype=float),
            np.asarray(owner, dtype=int))


def count_matrix(flat: tuple, n_bins: int, trains: list, duration_seconds: float,
                 lag_seconds: float = None) -> np.ndarray:
    """
    Description
    -----------
    Raw spike count per bin per unit over the kept sub-intervals.

    One `lag_seconds` shifts every unit by the SAME amount, which preserves the correlations
    between units and breaks only their alignment to vocalization. Shifting each unit
    independently would additionally destroy those correlations, leaving the shuffled population
    less structured than the real one and the null correspondingly too easy to beat.

    Counts are raw. Nothing is divided by time and nothing is standardized -- see the module
    docstring.

    Parameters
    ----------
    flat (tuple)
        The output of `flatten_spans`.
    n_bins (int)
        Number of bins.
    trains (list)
        Sorted spike-time arrays in seconds, one per unit.
    duration_seconds (float)
        Session length in seconds.
    lag_seconds (float)
        Common circular shift in seconds, or None for the observed data.

    Returns
    -------
    counts (np.ndarray)
        Bins by units.
    """

    starts, ends, owner = flat
    out = np.zeros((n_bins, len(trains)), dtype=float)
    for j, train in enumerate(trains):
        moved = (train if lag_seconds is None
                 else _circular_shift_spike_times(train, duration_seconds, lag_seconds))
        counts = (np.searchsorted(moved, ends) - np.searchsorted(moved, starts)).astype(float)
        np.add.at(out[:, j], owner, counts)
    return out


def nearest_neighbour(train: np.ndarray, train_y: np.ndarray, test: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Description
    -----------
    Label each test bin by the training bin it correlates with most, classes balanced first.

    Each population vector is centred across units and normalised, which is Pearson correlation
    between bins. Units are NOT standardised first -- see the module docstring. Among training
    bins tied at the maximum one is drawn at random rather than taking the first, which would
    otherwise hand every tie to whichever bin comes earliest.

    Parameters
    ----------
    train, test (np.ndarray)
        Bins by units, for the training and test sets.
    train_y (np.ndarray)
        Training labels.
    rng (np.random.Generator)
        Source of the class subsample and the tie-breaking draws.

    Returns
    -------
    predicted (np.ndarray)
        A label per test bin.
    """

    smallest = min(int(np.sum(train_y == c)) for c in (0, 1))
    if smallest == 0:
        msg = ("A cross-validation fold has no training examples of one class; the decoder cannot "
               "balance it. Check n_folds against the number of in-bout bins.")
        raise ValueError(msg)
    pick = np.concatenate([rng.choice(np.flatnonzero(train_y == c), smallest, replace=False)
                           for c in (0, 1)])
    train, train_y = train[pick], train_y[pick]
    a, b = train.astype(float), test.astype(float)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    corr = b @ a.T
    best = corr.max(axis=1, keepdims=True)
    return np.array([train_y[rng.choice(np.flatnonzero(corr[i] == best[i, 0]))]
                     for i in range(b.shape[0])])


def balanced_accuracy(matrix: np.ndarray, labels: np.ndarray, folds: np.ndarray, n_folds: int,
                      rng: np.random.Generator) -> float:
    """
    Description
    -----------
    Mean of the two class recalls, over contiguous time folds.

    Parameters
    ----------
    matrix (np.ndarray)
        Bins by units.
    labels (np.ndarray)
        The 0/1 label per bin.
    folds (np.ndarray)
        Which contiguous fold each bin falls in.
    n_folds (int)
        Number of folds.
    rng (np.random.Generator)
        Passed through to the classifier.

    Returns
    -------
    accuracy (float)
        Balanced accuracy over all folds.
    """

    predicted = np.zeros_like(labels)
    for fold in range(n_folds):
        held_out = folds == fold
        predicted[held_out] = nearest_neighbour(matrix[~held_out], labels[~held_out],
                                                matrix[held_out], rng)
    return float(np.mean([np.mean(predicted[labels == c] == c) for c in (0, 1)]))



def _null_draw_mean(flat: np.ndarray, labels: np.ndarray, folds: np.ndarray, n_folds: int,
                    trains: list, count: int, duration_seconds: float, lag_seconds: float,
                    runs_per_count: int, rng: np.random.Generator) -> float:
    """
    Description
    -----------
    One null draw of the unit-ladder statistic: every unit is shifted by the same
    ``lag_seconds`` (one circular shift of the whole session, so the units keep their
    correlations and lose only their alignment to vocalization), and the balanced accuracy is
    averaged over ``runs_per_count`` random subsets of ``count`` units, exactly as the observed
    value averages its runs. A null of single runs would spread wider than a mean of runs and
    leave the p-value conservative.

    Parameters
    ----------
    flat (tuple)
        The bins' flattened spans, as ``flatten_spans`` returns them.
    labels (np.ndarray)
        ``(n_bins,)`` 1 for in-bout, 0 for silent bins.
    folds (np.ndarray)
        ``(n_bins,)`` fold index of each bin.
    n_folds (int)
        Number of contiguous folds.
    trains (list)
        Sorted spike-time arrays of the area's units, in seconds.
    count (int)
        Units per subset (one rung of the ladder).
    duration_seconds (float)
        Session duration, the circular-shift period.
    lag_seconds (float)
        The common shift of this draw.
    runs_per_count (int)
        Random subsets averaged per draw.
    rng (np.random.Generator)
        Draws the subsets and breaks the decoder's ties.

    Returns
    -------
    accuracy (float)
        Mean balanced accuracy of the draw.
    """

    return float(np.mean([
        balanced_accuracy(
            count_matrix(flat, labels.size, [trains[j] for j in rng.choice(len(trains), count, replace=False)],
                         duration_seconds, lag_seconds=lag_seconds),
            labels, folds, n_folds, rng)
        for _ in range(runs_per_count)]))

class BoutOccupancyDecoder:
    """
    Description
    -----------
    Per-session decode of vocal-bout occupancy (whether a moment lies inside a
    USV bout) from population spiking, run separately for every brain area of
    the session that has enough units. Sessions are independent here and
    nothing is pooled across them.

    The constructor takes keyword arguments only and stores each as an
    attribute of the same name: ``root_directory``,
    ``decoding_parameters_dict`` (the ``decode_usv_bout_occupancy`` analyses
    settings, including the ``emitter`` choice ``self`` / ``partner`` / ``all``)
    and ``message_output``; any other key raises ``TypeError``. It also sets
    ``app_context_bool`` (GUI vs. CLI context). The entry method is
    ``decode_usv_bout_occupancy``, which writes one ``<session>_<area>.npz`` per
    area plus ``<session>_bout_decoding.parquet``. It is constructed by
    ``analyses/analyze_data.py``.
    """

    def __init__(self, **kwargs) -> None:
        """
        Description
        -----------
        Initialize the per-session bout-occupancy decode, validating and stashing the keyword
        arguments as attributes and recording GUI-vs-CLI execution context.

        Parameters
        ----------
        **kwargs (dict)
            The expected keys are `root_directory`, `decoding_parameters_dict` and
            `message_output`; each is set as an attribute and any other key raises ``TypeError``.

        Returns
        -------
        None
        """

        expected_kwargs = {"root_directory", "decoding_parameters_dict", "message_output"}
        unexpected_kwargs = set(kwargs) - expected_kwargs
        if unexpected_kwargs:
            msg = (f"{type(self).__name__}() got unexpected keyword argument(s) "
                   f"{', '.join(map(repr, sorted(unexpected_kwargs)))}; expected only "
                   f"{', '.join(map(repr, sorted(expected_kwargs)))}.")
            raise TypeError(msg)
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val
        self.app_context_bool = is_gui_context()

    def _emitter_names(self, bundle: dict, focal_mouse: str) -> list:
        """
        Description
        -----------
        Which track names count as the emitter, under the `self` / `partner` / `all` vocabulary.

        `all` returns an empty list, meaning NO filter, deliberately: a session's USV table can
        carry rows whose emitter was never assigned, and a USV nobody was credited with is still
        a USV the animal heard.

        Parameters
        ----------
        bundle (dict)
            Output of `load_vocal_inputs`.
        focal_mouse (str)
            The recorded animal, from the unit catalog.

        Returns
        -------
        names (list)
            Track names to keep, or an empty list for no filter.
        """

        emitter = self.decoding_parameters_dict["emitter"]
        track_names = bundle["track_names"]
        if emitter == "all":
            return []
        if emitter == "self":
            if focal_mouse not in track_names:
                msg = (f"Recorded animal {focal_mouse!r} is not among this session's tracks "
                       f"{track_names}; cannot resolve emitter 'self'.")
                raise ValueError(msg)
            return [focal_mouse]
        if emitter == "partner":
            partners = [n for n in track_names if n != focal_mouse]
            if not partners:
                msg = f"No partner track beside {focal_mouse!r} in {track_names}."
                raise ValueError(msg)
            return partners
        msg = f"emitter must be 'self', 'partner' or 'all'; got {emitter!r}."
        raise ValueError(msg)

    def decode_usv_bout_occupancy(self) -> None:
        """
        Description
        -----------
        Decode bout occupancy from every brain area of this session that clears the unit floor.

        Areas are taken from the unit catalog rather than named in advance, so a session
        contributes whatever it recorded; areas seen but held back by the floor are reported by
        name, so an absent area is never ambiguous. Each area writes its own compressed archive
        carrying every observed and every null draw, and the session writes one summary table.
        Assembling across sessions is deliberately left to the caller -- sessions are independent
        here and nothing is pooled.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Writes one ``<session>_<area>.npz`` per area plus ``<session>_bout_decoding.parquet``.
        """

        message_output = self.message_output if hasattr(self, "message_output") else print
        params = self.decoding_parameters_dict
        message_output(f"Decoding USV bout occupancy started at: "
                       f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:"
                       f"{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        root = pathlib.Path(self.root_directory)
        session_id = root.name
        bundle = load_vocal_inputs(root)
        if bundle is None:
            message_output("  skipped: missing *_usv_summary.csv or tracking .h5 in this session.")
            return

        catalog = pls.read_csv(str(resolve_experimenter_path(params["catalog_path"])),
                               infer_schema_length=20000)
        in_session = [session_id in (ast.literal_eval(s) if isinstance(s, str) else s)
                      for s in catalog["rec_sessions"].to_list()]
        block = catalog.filter(pls.Series(in_session)
                               & pls.col("cluster_group").is_in(params["cluster_group"])
                               & pls.col("brain_area").is_not_null())
        if bool(params["somatic_bool"]):
            block = block.filter(pls.col("somatic"))
        if block.height == 0:
            message_output(f"  skipped: no catalog units for {session_id} under the quality "
                           f"filters.")
            return
        focal_mouse = block["mouse_id"][0]

        keep = self._emitter_names(bundle, focal_mouse)
        # Bouts are grouped from PURE USVs only (`is_usv`: usv & ~squeak), so pure squeaks,
        # overlap segments and unscored nulls never join or split a bout. Noise was already
        # dropped by the loader. Silence, though, is judged against every vocalization of the
        # emitter (any class) and every one assigned to no animal, each spanning the full extent of
        # its sound (the loader's occupancy times: a squeak-bearing row covers its segment and its
        # squeak extent), so neither a negative bin nor a positive bin's kept time can hold a squeak
        # or an overlap segment.
        if keep:
            emitter_any = np.isin(bundle["emitters"], keep)
        else:
            emitter_any = np.ones(bundle["starts"].size, dtype=bool)
        emitter_rows = bundle["is_usv"] & emitter_any
        order = np.argsort(bundle["starts"][emitter_rows], kind="mergesort")
        usv_starts = bundle["starts"][emitter_rows][order]
        usv_stops = bundle["stops"][emitter_rows][order]
        occupancy_emitters = bundle["occupancy_emitters"]
        if keep:
            occupancy_any = np.isin(occupancy_emitters, keep)
        else:
            occupancy_any = np.ones(bundle["occupancy_starts"].size, dtype=bool)
        unassigned = np.array([name is None for name in occupancy_emitters], dtype=bool)
        silence_rows = occupancy_any | unassigned
        silence_order = np.argsort(bundle["occupancy_starts"][silence_rows], kind="mergesort")
        silence_starts = bundle["occupancy_starts"][silence_rows][silence_order]
        silence_stops = bundle["occupancy_stops"][silence_rows][silence_order]

        # The threshold belongs to whoever is vocalizing, so 'partner' reads the partner's
        # mixture, not the recorded animal's. Under 'all' the emitter set is mixed (and includes
        # unassigned rows), so there is no single right sex and the recorded animal's is used.
        emitter_sexes = {bundle["animal_sex"][name] for name in keep}
        sex = (emitter_sexes.pop() if len(emitter_sexes) == 1
               else bundle["animal_sex"][focal_mouse])
        threshold = ibi_bout_threshold(sex, float(params["ibi_z_score"]),
                                       int(params["ibi_component_index"]))
        spans, labels, mids, positions = bout_occupancy_bins(
            float(params["bin_seconds"]), usv_starts, usv_stops, silence_starts, silence_stops,
            bundle["duration_seconds"], threshold, int(params["min_vocalizations"]),
            float(params["min_kept_seconds"]))
        n_positive, n_negative = int((labels == 1).sum()), int((labels == 0).sum())
        # The sex that picked the mixture is not logged: CodeQL reads `animal_sex` as personal data
        # (py/clear-text-logging-sensitive-data), and the bout gap it sets is printed instead.
        message_output(f"  {session_id}: emitter {params['emitter']} ({focal_mouse}), "
                       f"bout gap {1000 * threshold:.1f} ms, {n_positive} in-bout bins / "
                       f"{n_negative} silent bins")
        if n_positive == 0 or n_negative == 0:
            message_output("  skipped: one of the two classes is empty.")
            return

        n_folds = int(params["n_folds"])
        flat = flatten_spans(spans)
        folds = np.minimum((mids / bundle["duration_seconds"] * n_folds).astype(int), n_folds - 1)

        rosters: dict = {}
        for unit_id, area in zip(block["unit_id"].to_list(), block["brain_area"].to_list()):
            rosters.setdefault(area, []).append(unit_id)
        floor = int(params["min_units_per_area"])
        requested = params["brain_areas"]
        if requested is not None:
            missing = sorted(set(requested) - set(rosters))
            if missing:
                message_output(f"  requested areas absent from this session: {', '.join(missing)}")
            rosters = {a: u for a, u in rosters.items() if a in requested}
        held_back = sorted((a, len(u)) for a, u in rosters.items() if len(u) < floor)
        if held_back:
            message_output("  below the " + str(floor) + "-unit floor, not decoded: "
                           + ", ".join(f"{a} ({n})" for a, n in held_back))
        rosters = {a: sorted(u) for a, u in rosters.items() if len(u) >= floor}
        if not rosters:
            message_output("  skipped: no area clears the unit floor.")
            return

        out_dir = pathlib.Path(resolve_experimenter_path(params["output_directory"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, n_shuffles = [], int(params["n_shuffles"])
        guard = float(params["shuffle_guard_seconds"])

        for area, roster in sorted(rosters.items()):
            spike_files = []
            for u in roster:
                spike_file = unique_match_or_none(list((root / "ephys").rglob(f"cluster_data/{u}.npy")),
                                                  label=f"cluster_data/{u}.npy under {root / 'ephys'}")
                if spike_file is None:
                    msg = f"No cluster_data/{u}.npy under {root / 'ephys'}."
                    raise FileNotFoundError(msg)
                spike_files.append(spike_file)
            trains = [np.sort(np.load(p)[0].astype(float)) for p in spike_files]
            ladder = [c for c in params["unit_ladder"] if c < len(roster)] + [len(roster)]
            rng = np.random.default_rng(int(params["shuffle_seed"]))
            observed_all, null_all = {}, {}
            for count in tqdm(ladder, desc=f"{area} unit ladder"):
                observed = np.asarray([
                    balanced_accuracy(
                        count_matrix(flat, labels.size,
                                     [trains[j] for j in rng.choice(len(trains), count,
                                                                    replace=False)],
                                     bundle["duration_seconds"]),
                        labels, folds, n_folds, rng)
                    for _ in range(int(params["runs_per_count"]))])
                # Each null draw is the same statistic as the observed value: the mean over
                # runs_per_count random unit subsets, here all decoded from ONE common circular
                # shift of the session. Comparing the observed mean of runs with single-run null
                # draws (which spread wider) made the p-value conservative and mis-scaled.
                null = np.asarray([
                    _null_draw_mean(flat, labels, folds, n_folds, trains, count,
                                    bundle["duration_seconds"],
                                    float(rng.uniform(guard, bundle["duration_seconds"] - guard)),
                                    int(params["runs_per_count"]), rng)
                    for _ in range(n_shuffles)])
                observed_all[str(count)] = observed
                null_all[f"null_{count}"] = null
                rows.append({"session": session_id, "mouse": focal_mouse, "area": area,
                             "cells": count, "n_units": len(roster),
                             "n_positive": n_positive, "n_negative": n_negative,
                             "ibi_threshold_seconds": threshold,
                             "accuracy": float(observed.mean()),
                             "accuracy_sd": float(observed.std()),
                             "null": float(null.mean()), "null_sd": float(null.std()),
                             "null_p995": float(np.percentile(null, 99.5)),
                             "p": (1 + int(np.sum(null >= observed.mean()))) / (n_shuffles + 1)})
                message_output(f"    {area:6}{count:>5} cells  {observed.mean():.4f}  "
                               f"null {null.mean():.4f}  p {rows[-1]['p']:.4f}")
            # written through a handle because `np.savez_compressed` appends '.npz' to a path
            # that does not already end in it, which would leave the atomic rename with nothing
            # to move
            with atomic_output_path(out_dir / f"{session_id}_{area}.npz") as tmp_path:
                with pathlib.Path(tmp_path).open("wb") as archive:
                    np.savez_compressed(archive, labels=labels, positions=positions,
                                        ladder=np.asarray(ladder), **observed_all, **null_all)

        with atomic_output_path(out_dir / f"{session_id}_bout_decoding.parquet") as tmp_path:
            pls.DataFrame(rows).write_parquet(tmp_path)
        message_output(f"Decoding USV bout occupancy finished at: "
                       f"{datetime.now().hour:02d}:{datetime.now().minute:02d}:"
                       f"{datetime.now().second:02d}")
