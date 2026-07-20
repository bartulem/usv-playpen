"""
@author: bartulem
Key-coverage guard for settings blocks that feed a shared consumer.

Several functions are fed by more than one settings block: they read keys off a
dictionary the caller supplies, and different callers supply different blocks.
Nothing in the type system or the test suite tied those blocks to the keys the
consumer actually reads, so adding a key to one block and not its sibling was a
silent, one-sided change that only surfaced as a runtime ``KeyError`` on a
user's machine.

That is exactly how ``frequency_shift_audio_segment`` broke: it is reached both
from ``analyses_settings.json -> frequency_shift_audio_segment`` and from
``visualizations_settings.json -> make_behavioral_videos.pitch_shifted_audio_specs``,
and when ``fs_compand_transfer`` / ``fs_noise_reduction_std_threshold`` /
``fs_sinc_upper_cutoff_hz`` moved out of the source into settings, only the
analyses block gained them. The suite could not see it: every test holding the
real settings mocks the consumer, and the only test running the real consumer
hand-builds its dictionary, so the shipped block and the real reader never met.

This guard closes that gap without un-mocking anything. For each registered
contract it derives the key set **from the consumer's source** (so the
expectation cannot drift out of step with the code) and asserts every feeding
block supplies it. Keys a block legitimately lacks — because the caller injects
them at runtime, or because the read is guarded by which block is in play — are
declared per block, so an intentional asymmetry is recorded rather than
silently tolerated.
"""

from __future__ import annotations

import ast
import json
import pathlib
from dataclasses import dataclass, field

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "usv_playpen"
SETTINGS_ROOT = SRC_ROOT / "_parameter_settings"


@dataclass(frozen=True)
class FeedingBlock:
    """
    Description
    -----------
    One settings block that is passed to a consumer at some call site.

    Parameters
    ----------
    settings_file (str)
        File name under ``_parameter_settings/`` holding the block.
    path (tuple of str)
        Dotted path to the block within that file, outermost key first.
    exempt (frozenset of str)
        Keys the consumer reads that this block is *not* required to supply,
        each of which must be justified in ``reason``.
    reason (str)
        Why the exempt keys are absent (runtime injection, guarded read, ...).

    Returns
    -------
    None
    """

    settings_file: str
    path: tuple[str, ...]
    exempt: frozenset[str] = field(default_factory=frozenset)
    reason: str = ""

    @property
    def label(self) -> str:
        """Human-readable "<file>::<dotted.path>" identifier for assertions."""
        return f"{self.settings_file}::{'.'.join(self.path)}"


@dataclass(frozen=True)
class SettingsContract:
    """
    Description
    -----------
    A consumer that reads literal keys off a settings dictionary, together with
    every block known to be passed to it.

    Parameters
    ----------
    module (str)
        Path of the consuming module, relative to ``src/usv_playpen``.
    function (str)
        Name of the consuming function or method.
    dict_expression (str)
        Source text of the expression the keys are read from (e.g.
        ``self.freq_shift_settings_dict``), matched against unparsed AST.
    blocks (tuple of FeedingBlock)
        The settings blocks that reach this consumer.

    Returns
    -------
    None
    """

    module: str
    function: str
    dict_expression: str
    blocks: tuple[FeedingBlock, ...]

    @property
    def label(self) -> str:
        """Human-readable "<module>::<function>::<expr>" identifier."""
        return f"{self.module}::{self.function}::{self.dict_expression}"


# Every consumer fed by more than one settings block. Register a contract here
# whenever a function starts reading its configuration from a caller-supplied
# dictionary that more than one block can fill.
CONTRACTS: tuple[SettingsContract, ...] = (
    SettingsContract(
        module="analyses/generate_audio_files.py",
        function="frequency_shift_audio_segment",
        dict_expression="self.freq_shift_settings_dict",
        blocks=(
            FeedingBlock(
                settings_file="analyses_settings.json",
                path=("frequency_shift_audio_segment",),
            ),
            FeedingBlock(
                settings_file="visualizations_settings.json",
                path=("make_behavioral_videos", "pitch_shifted_audio_specs"),
                exempt=frozenset({"fs_sequence_start", "fs_sequence_duration"}),
                reason=(
                    "the behavioral-video caller injects the window from the "
                    "video's own video_start_time / video_duration"
                ),
            ),
        ),
    ),
    SettingsContract(
        module="modeling/modeling_metadata.py",
        function="build_run_metadata",
        dict_expression="jax_block",
        blocks=(
            FeedingBlock(
                settings_file="modeling_settings.json",
                path=("hyperparameters", "jax_linear", "bivariate"),
                exempt=frozenset({
                    "focal_loss_gamma",
                    "balance_predictions_bool",
                    "balance_train_bool",
                }),
                reason="read only under the `jax_kind == 'multinomial_logistic'` guard",
            ),
            FeedingBlock(
                settings_file="modeling_settings.json",
                path=("hyperparameters", "jax_linear", "multinomial_logistic"),
            ),
        ),
    ),
    SettingsContract(
        module="modeling/modeling_metadata.py",
        function="build_run_metadata",
        dict_expression="tp",
        blocks=(
            FeedingBlock(
                settings_file="modeling_settings.json",
                path=("hyperparameters", "jax_linear", "bivariate",
                      "tune_regularization_params"),
            ),
            FeedingBlock(
                settings_file="modeling_settings.json",
                path=("hyperparameters", "jax_linear", "multinomial_logistic",
                      "tune_regularization_params"),
            ),
        ),
    ),
)


def _literal_keys_read(module_relpath: str, function_name: str, dict_expression: str) -> frozenset[str]:
    """
    Description
    -----------
    Parse a module and collect every string-literal subscript key read off
    ``dict_expression`` inside ``function_name``. Deriving the key set from the
    source (rather than restating it here) is what keeps this guard honest: a
    key added to the consumer is picked up automatically, so the contract
    cannot quietly fall out of step with the code it describes.

    Parameters
    ----------
    module_relpath (str)
        Module path relative to ``src/usv_playpen``.
    function_name (str)
        Name of the function or method whose body is scanned.
    dict_expression (str)
        Source text of the subscripted expression to match.

    Returns
    -------
    keys (frozenset of str)
        Every literal key read off that expression in that function.
    """

    module_path = SRC_ROOT / module_relpath
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    targets = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert targets, f"{module_relpath}: no function named {function_name!r}"

    keys: set[str] = set()
    for target in targets:
        for node in ast.walk(target):
            if not isinstance(node, ast.Subscript):
                continue
            if not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
                continue
            if ast.unparse(node.value) == dict_expression:
                keys.add(node.slice.value)
    return frozenset(keys)


def _block_keys(block: FeedingBlock) -> frozenset[str]:
    """
    Description
    -----------
    Resolve a feeding block in its shipped settings file and return its keys.

    Parameters
    ----------
    block (FeedingBlock)
        The block to resolve.

    Returns
    -------
    keys (frozenset of str)
        Keys present in that block as shipped.
    """

    node = json.loads((SETTINGS_ROOT / block.settings_file).read_text(encoding="utf-8"))
    for step in block.path:
        assert isinstance(node, dict), f"{block.label}: settings path is not a block at {step!r}"
        assert step in node, f"{block.label}: settings path does not resolve (missing {step!r})"
        node = node[step]
    assert isinstance(node, dict), f"{block.label}: settings path is not a block"
    return frozenset(node)


@pytest.mark.parametrize(
    ("contract", "block"),
    [(c, b) for c in CONTRACTS for b in c.blocks],
    ids=[f"{c.label}->{b.label}" for c in CONTRACTS for b in c.blocks],
)
def test_feeding_block_supplies_every_key_its_consumer_reads(contract, block):
    """A block reaching a consumer must supply every key that consumer reads."""
    required = _literal_keys_read(contract.module, contract.function, contract.dict_expression)
    assert required, (
        f"{contract.label}: no literal key reads found — the contract's "
        f"function/dict_expression is stale and this guard is now vacuous"
    )

    missing = required - block.exempt - _block_keys(block)
    assert not missing, (
        f"{block.label} is missing {sorted(missing)}, which "
        f"{contract.label} reads. Add the keys to that block, or declare them "
        f"exempt with a reason if the caller injects them."
    )


@pytest.mark.parametrize(
    "contract",
    CONTRACTS,
    ids=[c.label for c in CONTRACTS],
)
def test_declared_exemptions_are_still_warranted(contract):
    """An exemption must name a key the consumer reads and the block truly lacks."""
    required = _literal_keys_read(contract.module, contract.function, contract.dict_expression)
    for block in contract.blocks:
        assert block.exempt <= required, (
            f"{block.label}: exempts {sorted(block.exempt - required)}, which "
            f"{contract.label} does not read — drop the stale exemption."
        )
        redundant = block.exempt & _block_keys(block)
        assert not redundant, (
            f"{block.label}: exempts {sorted(redundant)} but ships them — drop "
            f"the exemption so the key stays covered."
        )
        assert not block.exempt or block.reason, (
            f"{block.label}: exemptions must carry a reason."
        )
