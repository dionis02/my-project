#!/usr/bin/env python3
"""Analyze broken molecules around aromatic-amide ortho correction.

The generator does not stamp ``source_current_task`` on candidates rejected
before scoring.  Therefore this utility reports two cohorts separately:

* ``explicit``: the SDF metadata proves that an ortho-correction task ran;
* ``inferred_last_atom``: the last atom is attached to a free/ortho position of
  a geometrically rotated aromatic amide.  This is a strong structural
  inference, not provenance proof.

Example:
    python generator/analyze_ortho_amide_failures.py all_brokens.sdf \
        --scored all_scored.sdf --output-dir ortho_amide_report
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


CORRECTION_TASKS = {
    "build_orthosubstituent_to_rotated_amide",
    "build_orthosubstituent_to_rotated_aniline_Nd0",
}


def properties(mol: Chem.Mol) -> dict[str, str]:
    return {name: mol.GetProp(name) for name in mol.GetPropNames()}


def atom_types(mol: Chem.Mol) -> list[str]:
    if not mol.HasProp("name"):
        return []
    return mol.GetProp("name").split()


def normalized_dihedral(value: float) -> float:
    value = abs(value) % 360.0
    return 360.0 - value if value > 180.0 else value


def generator_aromatic_ring(mol: Chem.Mol, ring: Iterable[int], names: list[str]) -> bool:
    ring = tuple(ring)
    if not ring or max(ring) >= len(names):
        return False
    allowed = {"Car", "C3r", "Nac", "Nd0", "Sul", "O_a"}
    double_bonds = 0
    for pos, idx in enumerate(ring):
        if names[idx] not in allowed:
            return False
        nxt = ring[(pos + 1) % len(ring)]
        bond = mol.GetBondBetweenAtoms(idx, nxt)
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
            double_bonds += 1
    return all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) or double_bonds >= len(ring) // 2


def aromatic_attachment(
    mol: Chem.Mol,
    center_idx: int,
    excluded_idx: int,
    names: list[str],
) -> list[tuple[int, tuple[int, int], tuple[int, ...]]]:
    result = []
    center = mol.GetAtomWithIdx(center_idx)
    for ipso in center.GetNeighbors():
        if ipso.GetIdx() == excluded_idx:
            continue
        for ring in mol.GetRingInfo().AtomRings():
            if ipso.GetIdx() not in ring or not generator_aromatic_ring(mol, ring, names):
                continue
            ring_set = set(ring)
            orthos = tuple(
                n.GetIdx()
                for n in ipso.GetNeighbors()
                if n.GetIdx() in ring_set and n.GetIdx() != center_idx
            )
            if len(orthos) == 2:
                result.append((ipso.GetIdx(), orthos, tuple(ring)))
    return result


def rotated_amide_contexts(mol: Chem.Mol, tolerance: float = 30.0) -> list[dict]:
    """Return rotated Ar-amide sides and their ortho positions."""
    names = atom_types(mol)
    if len(names) != mol.GetNumAtoms() or mol.GetNumConformers() == 0:
        return []
    contexts = []
    conf = mol.GetConformer()
    for n_idx, n_type in enumerate(names):
        if n_type != "Nd0":
            continue
        n_atom = mol.GetAtomWithIdx(n_idx)
        for carbonyl in n_atom.GetNeighbors():
            c_idx = carbonyl.GetIdx()
            if names[c_idx] != "Cs2":
                continue
            if not any(names[o.GetIdx()] == ".=O" for o in carbonyl.GetNeighbors()):
                continue
            for side, center_idx, excluded_idx, other_idx in (
                ("amide_N_side", n_idx, c_idx, c_idx),
                ("carbonyl_C_side", c_idx, n_idx, n_idx),
            ):
                for ipso_idx, orthos, ring in aromatic_attachment(
                    mol, center_idx, excluded_idx, names
                ):
                    try:
                        raw = rdMolTransforms.GetDihedralDeg(
                            conf, orthos[0], ipso_idx, center_idx, other_idx
                        )
                    except (RuntimeError, ValueError):
                        continue
                    dih = normalized_dihedral(raw)
                    if tolerance < dih < 180.0 - tolerance:
                        contexts.append(
                            {
                                "side": side,
                                "amide_n": n_idx,
                                "carbonyl_c": c_idx,
                                "ipso": ipso_idx,
                                "orthos": list(orthos),
                                "ring": list(ring),
                                "dihedral": round(dih, 3),
                            }
                        )
    return contexts


def rotated_aniline_contexts(mol: Chem.Mol, tolerance: float = 30.0) -> list[dict]:
    """Return rotated non-amide Ar-Nd0-X contexts (30-degree conservative rule)."""
    names = atom_types(mol)
    if len(names) != mol.GetNumAtoms() or mol.GetNumConformers() == 0:
        return []
    contexts = []
    conf = mol.GetConformer()
    for n_idx, n_type in enumerate(names):
        if n_type != "Nd0":
            continue
        n_atom = mol.GetAtomWithIdx(n_idx)
        heavy = [a for a in n_atom.GetNeighbors() if a.GetSymbol() != "H"]
        is_amide = any(
            names[a.GetIdx()] == "Cs2"
            and any(names[o.GetIdx()] == ".=O" for o in a.GetNeighbors())
            for a in heavy
        )
        if is_amide or len(heavy) < 2:
            continue
        for ipso_idx, orthos, ring in aromatic_attachment(mol, n_idx, -1, names):
            for opposite in heavy:
                if opposite.GetIdx() == ipso_idx:
                    continue
                try:
                    raw = rdMolTransforms.GetDihedralDeg(
                        conf, orthos[0], ipso_idx, n_idx, opposite.GetIdx()
                    )
                except (RuntimeError, ValueError):
                    continue
                dih = normalized_dihedral(raw)
                if tolerance < dih < 180.0 - tolerance:
                    contexts.append(
                        {
                            "side": "aniline_Nd0",
                            "amide_n": n_idx,
                            "carbonyl_c": "",
                            "ipso": ipso_idx,
                            "orthos": list(orthos),
                            "ring": list(ring),
                            "dihedral": round(dih, 3),
                        }
                    )
    return contexts


def rotated_correction_contexts(mol: Chem.Mol) -> list[dict]:
    return rotated_amide_contexts(mol) + rotated_aniline_contexts(mol)


def explicit_correction_tasks(mol: Chem.Mol) -> set[str]:
    found = set()
    for key in ("source_current_task", "current_task", "tasks"):
        if mol.HasProp(key):
            found.update(CORRECTION_TASKS.intersection(mol.GetProp(key).split()))
    for task in CORRECTION_TASKS:
        if mol.HasProp(task):
            found.add(task)
    return found


def last_atom_ortho_additions(mol: Chem.Mol, contexts: list[dict]) -> list[dict]:
    """Infer correction children using the generator's append-last convention."""
    if mol.GetNumAtoms() == 0:
        return []
    last_idx = mol.GetNumAtoms() - 1
    last = mol.GetAtomWithIdx(last_idx)
    additions = []
    for context in contexts:
        attached_orthos = [
            idx
            for idx in context["orthos"]
            if mol.GetBondBetweenAtoms(last_idx, idx) is not None
        ]
        if attached_orthos:
            additions.append(
                {
                    **context,
                    "new_atom": last_idx,
                    "new_atom_type": atom_types(mol)[last_idx],
                    "attached_ortho": attached_orthos[0],
                    "new_atom_degree": last.GetDegree(),
                    "new_atom_in_ring": last.IsInRing(),
                }
            )
    return additions


def load_sdf(path: Path) -> list[Chem.Mol]:
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    return [mol for mol in supplier if mol is not None]


def row_for(mol: Chem.Mol, record: int, cohort: str, contexts: list[dict], additions: list[dict]) -> dict:
    props = properties(mol)
    addition = additions[0] if additions else {}
    return {
        "record": record,
        "cohort": cohort,
        "reason_to_skip": props.get("reason_to_skip", "<missing>"),
        "explicit_tasks": " ".join(sorted(explicit_correction_tasks(mol))),
        "source_iter": props.get("source_iter", ""),
        "source_parent_mol": props.get("source_parent_mol", ""),
        "source_scenario": props.get("source_scenario", props.get("scenario", "")),
        "source_current_task": props.get("source_current_task", ""),
        "next_task": props.get("current_task", ""),
        "tasks": props.get("tasks", ""),
        "num_atoms": mol.GetNumAtoms(),
        "num_rotated_amide_contexts": len(contexts),
        "amide_side": addition.get("side", ""),
        "amide_dihedral": addition.get("dihedral", ""),
        "ortho_atom": addition.get("attached_ortho", ""),
        "added_atom": addition.get("new_atom", ""),
        "added_atom_type": addition.get("new_atom_type", ""),
        "added_atom_degree": addition.get("new_atom_degree", ""),
        "added_atom_in_ring": addition.get("new_atom_in_ring", ""),
        "name": props.get("name", ""),
    }


def analyze(path: Path) -> tuple[list[Chem.Mol], list[dict], dict]:
    mols = load_sdf(path)
    rows = []
    selected_mols = []
    for record, mol in enumerate(mols, start=1):
        contexts = rotated_correction_contexts(mol)
        additions = last_atom_ortho_additions(mol, contexts)
        explicit = explicit_correction_tasks(mol)
        if explicit:
            cohort = "explicit_ortho_correction"
        elif additions:
            cohort = "inferred_last_atom_ortho_addition"
        elif contexts:
            cohort = "rotated_geometry_without_provenance"
        else:
            continue
        rows.append(row_for(mol, record, cohort, contexts, additions))
        selected_mols.append(mol)

    by_cohort = Counter(row["cohort"] for row in rows)
    reason_by_cohort = {
        cohort: dict(
            Counter(
                row["reason_to_skip"] for row in rows if row["cohort"] == cohort
            ).most_common()
        )
        for cohort in by_cohort
    }
    inferred = [row for row in rows if row["cohort"] == "inferred_last_atom_ortho_addition"]
    summary = {
        "input": str(path),
        "total_records": len(mols),
        "matched_records": len(rows),
        "cohorts": dict(by_cohort),
        "reasons_by_cohort": reason_by_cohort,
        "inferred_ortho_added_atom_types": dict(
            Counter(row["added_atom_type"] for row in inferred).most_common()
        ),
        "inferred_next_tasks": dict(
            Counter(row["next_task"] or "<missing>" for row in inferred).most_common()
        ),
        "metadata_warning": (
            "Pre-scoring broken records lack source_current_task in the current generator. "
            "The inferred_last_atom cohort is structural inference, not definitive lineage."
        ),
    }
    return selected_mols, rows, summary


def scored_summary(path: Path) -> dict:
    mols = load_sdf(path)
    matched = [m for m in mols if explicit_correction_tasks(m)]
    return {
        "input": str(path),
        "total_records": len(mols),
        "explicit_ortho_correction_survivors": len(matched),
        "next_tasks": dict(
            Counter(
                m.GetProp("current_task") if m.HasProp("current_task") else "<missing>"
                for m in matched
            ).most_common()
        ),
        "added_atom_types": dict(
            Counter(atom_types(m)[-1] for m in matched if atom_types(m)).most_common()
        ),
    }


def write_outputs(output_dir: Path, mols: list[Chem.Mol], rows: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(row_for(Chem.Mol(), 0, "", [], []))
    with (output_dir / "matched_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with Chem.SDWriter(str(output_dir / "matched_records.sdf")) as writer:
        for mol, row in zip(mols, rows):
            copy = Chem.Mol(mol)
            copy.SetProp("analysis_cohort", row["cohort"])
            copy.SetProp("analysis_record", str(row["record"]))
            writer.write(copy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brokens", type=Path, help="all_brokens.sdf from the run")
    parser.add_argument("--scored", type=Path, help="optional all_scored.sdf from the same run")
    parser.add_argument("--output-dir", type=Path, default=Path("ortho_amide_report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mols, rows, summary = analyze(args.brokens)
    if args.scored:
        summary["scored"] = scored_summary(args.scored)
    write_outputs(args.output_dir, mols, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
