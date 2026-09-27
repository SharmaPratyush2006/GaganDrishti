"""Phase 5: geographic train / held-out separation, checked against reality.

A validation number only means "generalises to a new place" if the model never
saw that place. This module turns that sentence into a check.

What it does
------------
1. The **declared** split comes from ``configs/phase5.yaml`` (``dfc.split``): named
   training regions and named held-out regions, keyed by ``city`` -- the only
   region identifier DFC2019 carries (``JAX_004_007`` -> ``JAX``, read by the
   Phase 3 ``scene_id_regex``). Nothing here invents a region assignment.
2. The **actual** training set is rebuilt from the checkpoint itself: every
   Phase 3 checkpoint embeds its full config, including ``dataset.split``, and
   :func:`depthwizard.relative.data.split_pairs` is deterministic. So the pairs
   the model trained on are recomputed exactly, not assumed.
3. :func:`audit_region_split` refuses (:class:`SplitError`) when:

   * the checkpoint's split is a random mode (``random_scene`` /
     ``random_tile``) -- a random split can never stand in for a spatial one;
   * any held-out region appears among the checkpoint's training pairs
     (the leakage case);
   * the checkpoint trained on a region not declared as training, or a
     declared training region was never trained on (the report would then
     state a false split);
   * any geographic tile (scene id) is on both sides;
   * a pair has no region identifier, or the held-out regions have no pairs.

A same-city, tile-disjoint split (Phase 3's shipped ``per_city_scene``) trains on
every city, so it can never pass this audit for any held-out city. That is the
intended outcome: neighbouring tiles of one city are not geographically
independent.

Model-selection leakage
-----------------------
A checkpoint can be trained without a held-out city and still have been
*chosen* with it: when the held-out city is the checkpoint's validation side
(``scene_prefix`` with ``val_scene_prefixes: [OMA]``), ``best.pt`` is the epoch
with the lowest held-out validation loss. :func:`audit_model_selection` therefore
accepts, in that case, only the **final pre-set epoch** (``training.epochs - 1``,
i.e. ``epoch_009.pt`` for 10 epochs), whose weights no validation loss chose. It
reads only metadata every Phase 3 checkpoint already stores (``epoch``,
``best_epoch``, ``training.epochs``); a checkpoint missing them is refused,
because the selection cannot be verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from depthwizard.logging_setup import get_logger
from depthwizard.validation.config import RegionSplitConfig, ValidationError

__all__ = [
    "SplitError",
    "SplitAudit",
    "CheckpointSelection",
    "region_of",
    "select_regions",
    "audit_region_split",
    "audit_model_selection",
]

log = get_logger(__name__)


class SplitError(ValidationError):
    """The declared geographic split does not hold; the message says how."""


def region_of(pair: Any, region_key: str = "city") -> str:
    """The region a DFC2019 pair belongs to.

    Raises:
        SplitError: when the pair carries no region identifier.
    """
    if region_key != "city":
        raise SplitError(f"unsupported region key {region_key!r}")
    region = getattr(pair, "city", "")
    if not region:
        raise SplitError(
            f"pair {getattr(pair, 'stem', pair)!r} has no city identifier; the scene_id_regex "
            "must have a (?P<city>...) group for a geographic split"
        )
    return str(region)


def select_regions(pairs: Sequence[Any], regions: Sequence[str], region_key: str = "city") -> tuple[Any, ...]:
    """The pairs whose region is in ``regions``, in their given order."""
    wanted = set(regions)
    return tuple(p for p in pairs if region_of(p, region_key) in wanted)


def _count_by_region(pairs: Sequence[Any], region_key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        region = region_of(pair, region_key)
        counts[region] = counts.get(region, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class SplitAudit:
    """A split that passed :func:`audit_region_split`, with its evidence."""

    region_key: str
    train_regions: tuple[str, ...]
    heldout_regions: tuple[str, ...]
    checkpoint_split_mode: str
    #: Pairs the checkpoint trained on, per region (recomputed, not assumed).
    checkpoint_train_pairs_by_region: dict[str, int]
    heldout_pairs_by_region: dict[str, int]
    heldout_scene_ids: tuple[str, ...]
    rationale: str = ""
    #: Regions on the checkpoint's own VALIDATION side (where its validation
    #: loss, and hence ``best.pt``, came from). Used by :func:`audit_model_selection`.
    checkpoint_validation_regions: tuple[str, ...] = ()

    @property
    def heldout_pairs(self) -> int:
        return sum(self.heldout_pairs_by_region.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_key": self.region_key,
            "training_regions": list(self.train_regions),
            "heldout_regions": list(self.heldout_regions),
            "rationale": self.rationale,
            "checkpoint_split_mode": self.checkpoint_split_mode,
            "checkpoint_train_pairs_by_region": self.checkpoint_train_pairs_by_region,
            "checkpoint_validation_regions": list(self.checkpoint_validation_regions),
            "heldout_pairs_by_region": self.heldout_pairs_by_region,
            "heldout_geographic_tiles": len(self.heldout_scene_ids),
            "scene_overlap": 0,
            "random_split": False,
        }


def audit_region_split(
    declared: RegionSplitConfig,
    pairs: Sequence[Any],
    checkpoint_split: Any,
) -> SplitAudit:
    """Check a declared region split against the split a checkpoint trained on.

    Args:
        declared: ``dfc.split`` from the Phase 5 config.
        pairs: Every discovered DFC2019 pair (:func:`~depthwizard.relative.data.discover_pairs`).
        checkpoint_split: The :class:`~depthwizard.relative.config.SplitConfig`
            recorded **inside the checkpoint** -- not the one in any YAML file.

    Returns:
        A :class:`SplitAudit`.

    Raises:
        SplitError: on any of the conditions in the module docstring.
    """
    from depthwizard.relative.data import split_pairs

    key = declared.region_key
    if not pairs:
        raise SplitError("no DFC2019 pairs were discovered; nothing to split")
    for pair in pairs:
        region_of(pair, key)

    mode = str(getattr(checkpoint_split, "mode", ""))
    if not getattr(checkpoint_split, "is_spatially_separated", False):
        raise SplitError(
            f"the checkpoint was trained with split.mode={mode!r}, which is NOT spatially "
            "separated; a random split cannot replace a geographic one"
        )

    trained, validation = split_pairs(pairs, checkpoint_split)
    trained_by_region = _count_by_region(trained, key)
    trained_regions = set(trained_by_region)

    leaked = sorted(trained_regions & set(declared.heldout_regions))
    if leaked:
        detail = ", ".join(f"{r}: {trained_by_region[r]} training pair(s)" for r in leaked)
        raise SplitError(
            f"held-out region(s) {leaked} were used to train the checkpoint ({detail}; checkpoint "
            f"split.mode={mode!r}). A city the model trained on cannot be a held-out validation city. "
            "Real-world validation on this split is not yet measured."
        )
    undeclared = sorted(trained_regions - set(declared.train_regions))
    if undeclared:
        raise SplitError(
            f"the checkpoint trained on region(s) {undeclared}, which dfc.split does not declare as "
            "training regions; the report would state a false split"
        )
    unused = sorted(set(declared.train_regions) - trained_regions)
    if unused:
        raise SplitError(
            f"declared training region(s) {unused} contributed no training pair to the checkpoint"
        )

    heldout = select_regions(pairs, declared.heldout_regions, key)
    if not heldout:
        raise SplitError(f"no pairs found for held-out region(s) {list(declared.heldout_regions)}")
    missing = sorted(set(declared.heldout_regions) - {region_of(p, key) for p in heldout})
    if missing:
        raise SplitError(f"held-out region(s) {missing} have no pairs in the dataset")

    shared = sorted({p.scene_id for p in trained} & {p.scene_id for p in heldout})
    if shared:
        raise SplitError(f"geographic tile(s) on both sides of the split: {shared[:5]}")

    audit = SplitAudit(
        region_key=key,
        train_regions=tuple(declared.train_regions),
        heldout_regions=tuple(declared.heldout_regions),
        checkpoint_split_mode=mode,
        checkpoint_train_pairs_by_region=trained_by_region,
        heldout_pairs_by_region=_count_by_region(heldout, key),
        heldout_scene_ids=tuple(sorted({p.scene_id for p in heldout})),
        rationale=declared.rationale,
        checkpoint_validation_regions=tuple(sorted({region_of(p, key) for p in validation})),
    )
    log.info("geographic split verified against the checkpoint", extra=audit.to_dict())
    return audit


@dataclass(frozen=True)
class CheckpointSelection:
    """Which epoch a checkpoint holds, from the metadata Phase 3 stores in it."""

    #: Zero-based epoch whose weights the file holds (``payload["epoch"]``).
    epoch: int | None
    #: Epoch with the lowest validation loss so far (``payload["best_epoch"]``).
    best_epoch: int | None
    #: Pre-set number of epochs (``config.training.epochs``).
    epochs: int | None

    @property
    def final_epoch(self) -> int | None:
        return self.epochs - 1 if self.epochs is not None else None

    @classmethod
    def from_payload(cls, payload: Any) -> "CheckpointSelection":
        def as_int(value: Any) -> int | None:
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        training = (payload.get("config") or {}).get("training") or {}
        return cls(epoch=as_int(payload.get("epoch")), best_epoch=as_int(payload.get("best_epoch")),
                   epochs=as_int(training.get("epochs")))

    def to_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "best_epoch": self.best_epoch, "epochs": self.epochs,
                "final_epoch": self.final_epoch}


def audit_model_selection(selection: CheckpointSelection, audit: SplitAudit) -> dict[str, Any]:
    """Refuse a checkpoint that may have been chosen with the held-out region.

    Only applies when a held-out region is on the checkpoint's validation side;
    then the checkpoint must be the final pre-set epoch. A best-validation-loss
    checkpoint whose best epoch *is* the final epoch holds those same weights and
    is accepted.

    Raises:
        SplitError: on a non-final epoch, or metadata too incomplete to verify.
    """
    overlap = sorted(set(audit.checkpoint_validation_regions) & set(audit.heldout_regions))
    record = {**selection.to_dict(), "validation_side_heldout_regions": overlap}
    if not overlap:
        return {**record, "status": "not applicable: no held-out region on the checkpoint's validation side"}
    if selection.epoch is None or selection.final_epoch is None:
        raise SplitError(
            f"the checkpoint's validation side contains held-out region(s) {overlap}, but it records no "
            "epoch / training.epochs, so it cannot be verified that the held-out validation loss did not "
            "select it"
        )
    if selection.epoch != selection.final_epoch:
        why = ("it is the best-validation-loss epoch" if selection.epoch == selection.best_epoch
               else "it is not the final pre-set epoch")
        raise SplitError(
            f"model-selection leakage: the checkpoint holds epoch {selection.epoch} of a "
            f"{selection.epochs}-epoch run (final = {selection.final_epoch}); {why}, and its validation loss "
            f"was computed on held-out region(s) {overlap}. Use the final-epoch checkpoint "
            f"(epoch_{selection.final_epoch:03d}.pt). Real-world validation is not yet measured."
        )
    return {**record, "status": "PASSED: final pre-set epoch; no validation loss selected it"}
