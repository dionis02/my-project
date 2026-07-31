#!/usr/bin/env python3
"""Extract a molecule's selected ancestors using its ``source_scenario``.

For a source scenario ``s0 s1 ... sn``, ``s0`` identifies the initial-input
branch.  For every later token, ``s[k]`` is the zero-based molecule index in
the selected batch whose scenario is the prefix ``s0 ... s[k-1]``.  Thus the
ancestors available in ``all_selected.sdf`` are reconstructed from successively
longer prefixes.  The initial input molecule itself is not stored there.

Example:
    python generator/extract_scenario_ancestors.py all_selected.sdf \
        --scenario 1 2 3 1 7 1 0 1 2 7 1 0 0 \
        --output ancestors.sdf
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem


@dataclass
class SelectedRecord:
    file_record: int
    molecule: Chem.Mol
    scenario: tuple[int, ...]
    source_iter: int | None


def parse_scenario(values: list[str]) -> tuple[int, ...]:
    """Accept both one quoted string and multiple integer arguments."""
    tokens = " ".join(values).replace(",", " ").split()
    if not tokens:
        raise ValueError("source_scenario is empty")
    try:
        scenario = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError("source_scenario must contain only integers") from exc
    if any(index < 0 for index in scenario):
        raise ValueError("source_scenario indices cannot be negative")
    return scenario


def mol_scenario(mol: Chem.Mol) -> tuple[int, ...] | None:
    for prop in ("scenario", "source_scenario"):
        if mol.HasProp(prop):
            try:
                return parse_scenario([mol.GetProp(prop)])
            except ValueError:
                return None
    return None


def load_selected(path: Path) -> tuple[list[SelectedRecord], int]:
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    records = []
    unreadable = 0
    for file_record, mol in enumerate(supplier, start=1):
        if mol is None:
            unreadable += 1
            continue
        scenario = mol_scenario(mol)
        if scenario is None:
            continue
        source_iter = None
        if mol.HasProp("source_iter"):
            try:
                source_iter = int(mol.GetProp("source_iter"))
            except ValueError:
                pass
        records.append(SelectedRecord(file_record, mol, scenario, source_iter))
    return records, unreadable


def contiguous_batches(records: list[SelectedRecord], prefix: tuple[int, ...]) -> list[list[SelectedRecord]]:
    """Split repeated/resumed writes of the same scenario into file-order batches."""
    matches = [record for record in records if record.scenario == prefix]
    batches: list[list[SelectedRecord]] = []
    for record in matches:
        if not batches or record.file_record != batches[-1][-1].file_record + 1:
            batches.append([record])
        else:
            batches[-1].append(record)
    return batches


def reconstruct(
    records: list[SelectedRecord],
    source_scenario: tuple[int, ...],
    batch_policy: str,
) -> tuple[list[tuple[int, int, tuple[int, ...], SelectedRecord]], list[dict]]:
    ancestors = []
    trace = []
    # Token zero points to an initial input molecule, which all_selected lacks.
    for depth in range(1, len(source_scenario)):
        prefix = source_scenario[:depth]
        selected_index = source_scenario[depth]
        expected_iter = depth
        candidates = [
            record
            for record in records
            if record.scenario == prefix
            and (record.source_iter is None or record.source_iter == expected_iter)
        ]
        batches = contiguous_batches(candidates, prefix)
        if not batches:
            raise LookupError(
                f"No selected batch for prefix {' '.join(map(str, prefix))!r} "
                f"(expected source_iter={expected_iter})"
            )
        batch = batches[0] if batch_policy == "first" else batches[-1]
        if selected_index >= len(batch):
            raise LookupError(
                f"Prefix {' '.join(map(str, prefix))!r} has {len(batch)} selected "
                f"molecules in the {batch_policy} batch, but scenario requests index "
                f"{selected_index}"
            )
        record = batch[selected_index]
        ancestors.append((depth, selected_index, prefix, record))
        trace.append(
            {
                "depth": depth,
                "scenario_prefix": " ".join(map(str, prefix)),
                "selected_index": selected_index,
                "batch_policy": batch_policy,
                "batch_size": len(batch),
                "all_selected_record": record.file_record,
                "source_iter": record.source_iter,
                "name": record.molecule.GetProp("name") if record.molecule.HasProp("name") else "",
                "current_task": (
                    record.molecule.GetProp("current_task")
                    if record.molecule.HasProp("current_task")
                    else ""
                ),
            }
        )
    return ancestors, trace


def write_ancestors(
    output: Path,
    ancestors: list[tuple[int, int, tuple[int, ...], SelectedRecord]],
    source_scenario: tuple[int, ...],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(output)) as writer:
        for depth, selected_index, prefix, record in ancestors:
            mol = Chem.Mol(record.molecule)
            mol.SetProp("ancestor_depth", str(depth))
            mol.SetProp("ancestor_scenario_prefix", " ".join(map(str, prefix)))
            mol.SetProp("ancestor_selected_index", str(selected_index))
            mol.SetProp("ancestor_all_selected_record", str(record.file_record))
            mol.SetProp("target_source_scenario", " ".join(map(str, source_scenario)))
            writer.write(mol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("all_selected", type=Path, help="all_selected.sdf from the same run")
    parser.add_argument(
        "--scenario",
        nargs="+",
        required=True,
        help="source_scenario as separate integers or one quoted string",
    )
    parser.add_argument("--output", type=Path, required=True, help="output SDF, oldest ancestor first")
    parser.add_argument(
        "--batch",
        choices=("first", "last"),
        default="last",
        help="which contiguous batch to use if a resumed run repeated a scenario (default: last)",
    )
    parser.add_argument("--trace-json", type=Path, help="optional JSON reconstruction trace")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        scenario = parse_scenario(args.scenario)
        records, unreadable = load_selected(args.all_selected)
        ancestors, trace = reconstruct(records, scenario, args.batch)
    except (ValueError, LookupError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    write_ancestors(args.output, ancestors, scenario)
    result = {
        "all_selected": str(args.all_selected),
        "target_source_scenario": " ".join(map(str, scenario)),
        "output": str(args.output),
        "ancestors_written": len(ancestors),
        "unreadable_sdf_records": unreadable,
        "initial_input_note": (
            "The first scenario token selects an initial input molecule; "
            "that molecule is not present in all_selected.sdf."
        ),
        "trace": trace,
    }
    if args.trace_json:
        args.trace_json.parent.mkdir(parents=True, exist_ok=True)
        args.trace_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
