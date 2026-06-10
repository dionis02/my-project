# измененный код с фильтрацией дубликатных веток
# ruff: noqa: PLR2004, PLR0912, PLR0915, N802, N803, PLR0913, PLR0911, E741, S112, BLE001

import configparser
import os
import shutil
import time
import warnings
from collections import Counter, defaultdict
from enum import Enum
from importlib.resources import as_file, files
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors, rdmolops
from rdkit.Geometry import Point3D
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from torch_geometric.data import DataLoader

import ml_models.generator_module.dot_predictor_ensemble_v3 as dot_pred
import ml_models.generator_module.ref_mol
import ml_models.utils.utils as general_utils
from ml_models.generator_module.probe_minimizer import probe_minimizer, probe_docker_denovo
from ml_models.parser_pdb.pandas_pdb import PandasPdb

from ml_models.generator_module.gen3D_V4_systematic import generate_conformers_from_smiles
from collections import deque

# импорт класса для динамического времени с учетом таймаутов
from timeout_snapshot_manager import (
    TimeoutSnapshotManager,
    current_timeout_return_iter,
    parent_timeout_return_iter,
)


RDLogger.DisableLog("rdApp.*")

warnings.filterwarnings("ignore")

GLOBAL_TARGET_NAMES = ["Car", "O_a", "Cs3", "Nac", "Nd+", "Nd0", "Cs2", ".=O", "Hal", "O_d", "Csp", "Sul", "SO2"]
GLOBAL_TARGET_NAMES_RING = ["Car", "O_a", "Cs3", "C3r", "Nac", "Nd+", "Nd0", "Cs2", "C2r", ".=O", "Hal", "O_d", "Csp", "Sul", "SO2"]
ring_types = ["Car", "C2r", "C3r", "O_a", "Nd0", "Nd+", "Nac", "SO2", "Sul"]

def define_atom_type(atom):
    atom_type = "undefined"
    if atom.GetSymbol() == "C":
        if str(atom.GetHybridization()) == "SP2":
            if atom.GetIsAromatic():
                atom_type = "Car"
            elif atom.IsInRing():
                atom_type = "C2r"
            else:
                atom_type = "Cs2"
        elif str(atom.GetHybridization()) == "SP3":
            atom_type = "C3r" if atom.IsInRing() else "Cs3"
        elif str(atom.GetHybridization()) == "SP":
            atom_type = "Csp"
    elif atom.GetSymbol() == "N":
        has_h_neighb = 0
        for neighb in atom.GetNeighbors():
            if neighb.GetSymbol() == "H":
                has_h_neighb = 1
                break
        atom_type = ("Nd+" if atom.GetFormalCharge() == 1 else "Nd0") if has_h_neighb == 1 else "Nac"
    elif atom.GetSymbol() == "O":
        if len(atom.GetNeighbors()) == 1:
            atom_type = ".=O"
        else:
            has_h_neighb = 0
            for neighb in atom.GetNeighbors():
                if neighb.GetSymbol() == "H":
                    has_h_neighb += 1
                    break
            atom_type = "O_d" if has_h_neighb else "O_a"
    elif atom.GetSymbol() == "S":
        o_neighbs = 0
        for neighb in atom.GetNeighbors():
            if neighb.GetSymbol() == "O":
                o_neighbs += 1
        if o_neighbs == 2:
            atom_type = "SO2"
        elif o_neighbs == 0 and len(atom.GetNeighbors()) == 2:
            atom_type = "Sul"
    elif atom.GetSymbol() in ["Cl", "Br", "I"]:
        atom_type = "Hal"
    return atom_type


def mol_contains_macrocycles(mol):
    return any(len(ring) > 7 for ring in mol.GetRingInfo().AtomRings())


def new_dihedral(p):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def angular_diff_deg(a1, a2):
    diff = abs(a1 - a2) % 360.0
    return min(diff, 360.0 - diff)

def find_dihedral_atom_indices(mol):
    d_idx = mol.GetNumAtoms() - 1  # новый атом
    d_atom = mol.GetAtomWithIdx(d_idx)

    d_neighbors = [a.GetIdx() for a in d_atom.GetNeighbors()]
    c_idx = d_neighbors[0]
    c_atom = mol.GetAtomWithIdx(c_idx)
    b_candidates = [a for a in c_atom.GetNeighbors() if a.GetIdx() != d_idx]
    for b in b_candidates:
        a_candidates = [a for a in b.GetNeighbors() if a.GetIdx() != c_idx]
        if a_candidates:
           b_idx = b.GetIdx()
           a_idx = a_candidates[0].GetIdx()

    return a_idx, b_idx, c_idx, d_idx

def get_dihedral_for_mol(mol, atom_indices, dihedral_func):
    a_idx, b_idx, c_idx, d_idx = atom_indices
    coords = [get_atom_coords(mol, i) for i in (a_idx, b_idx, c_idx, d_idx)]
    return dihedral_func(*coords)

def calc_rmsd_no_alignment(mol1, mol2):
# calculate rmsd without positional alignment (as in Chem.GetBestRms) to sort out duplicates
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    atom_indices = range(mol1.GetNumAtoms())

    sq_dists = []
    for i in atom_indices:
        p1 = conf1.GetAtomPosition(i)
        p2 = conf2.GetAtomPosition(i)
        diff = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z], dtype=float)
        sq_dists.append(np.dot(diff, diff))

    return float(np.sqrt(np.mean(sq_dists)))

def filter_similar_dihedrals(group, atom_type):
    fin_list = []
    atom_indices_list = []
    atom_type_is_cycle = atom_type in ring_types
    mol1_counter = 0
    for mol1 in group:
        atom_indices = find_dihedral_atom_indices(mol1)
        if not fin_list:
            fin_list.append(mol1)
            atom_indices_list.append(atom_indices)
        else:
            mol1_pos = mol1.GetConformer().GetPositions()
            a_atom_position_1 = mol1_pos[atom_indices[0]]
            b_atom_position_1 = mol1_pos[atom_indices[1]]
            c_atom_position_1 = mol1_pos[atom_indices[2]]
            key_atom_position_1 = mol1_pos[atom_indices[3]]
            angle_1 = new_dihedral([a_atom_position_1,b_atom_position_1,c_atom_position_1,key_atom_position_1])

            mol_1_dihedral_diff_list = []
            mol_1_rmsd_list = []
            mol_1_angle_diff_list = []

            for mol2 in fin_list:

                mol_1_rmsd_list.append(calc_rmsd_no_alignment(mol1, mol2))

                mol2_pos = mol2.GetConformer().GetPositions()
                a_atom_position_2 = mol2_pos[atom_indices[0]]
                b_atom_position_2 = mol2_pos[atom_indices[1]]
                c_atom_position_2 = mol2_pos[atom_indices[2]]
                key_atom_position_2 = mol2_pos[atom_indices[3]]
                angle_2 = new_dihedral([a_atom_position_2,b_atom_position_2,c_atom_position_2,key_atom_position_2])
                mol_1_dihedral_diff_list.append(angular_diff_deg(angle_1, angle_2))
                mol_1_angle_diff_list.append(angular_diff_deg(calc_angle(b_atom_position_1,c_atom_position_1,key_atom_position_1)
                                                         ,calc_angle(b_atom_position_2,c_atom_position_2,key_atom_position_2)))
            if all(rmsd > 1 for rmsd in mol_1_rmsd_list):
                fin_list.append(mol1)
                atom_indices_list.append(atom_indices)
            else:
                if atom_indices not in atom_indices_list:
                    fin_list.append(mol1)
                    atom_indices_list.append(atom_indices)
                else:
                    indexes_with_low_rmsd = [i for i in range(len(fin_list)) if (mol_1_rmsd_list[i] <=2 and atom_indices_list[i] == atom_indices)]
                    mols_with_low_rmsd = [fin_list[i] for i in indexes_with_low_rmsd]
                    dihedral_diffs_with_low_rmsd = [mol_1_dihedral_diff_list[i] for i in indexes_with_low_rmsd]
                    angle_diffs_with_low_rmsd = [mol_1_angle_diff_list[i] for i in indexes_with_low_rmsd]
                    bond_types_of_bonds_with_low_rmsd = [str(mols_with_low_rmsd[i].GetBondBetweenAtoms(atom_indices[2], atom_indices[3]).GetBondType()) for i in range(len(mols_with_low_rmsd))]


                    if all(dih > 30 for dih in dihedral_diffs_with_low_rmsd):
                        fin_list.append(mol1)
                        atom_indices_list.append(atom_indices)
                    elif atom_type_is_cycle:
                        angle_diffs_with_low_rmsd_low_dih_diff = [angle_diffs_with_low_rmsd[i] for i in range(len(angle_diffs_with_low_rmsd)) if dihedral_diffs_with_low_rmsd[i] < 30]
                        if all(ang > 5 for ang in angle_diffs_with_low_rmsd_low_dih_diff):
                            fin_list.append(mol1)
                            atom_indices_list.append(atom_indices)
                        elif str(mol1.GetBondBetweenAtoms(atom_indices[2], atom_indices[3]).GetBondType()) not in bond_types_of_bonds_with_low_rmsd:
                            fin_list.append(mol1)
                            atom_indices_list.append(atom_indices)
        mol1_counter+=1
    return fin_list


def ring_is_planar(mol, ring):
    positions = mol.GetConformer().GetPositions()
    atoms_combos = [list(x) for x in combinations(ring, 4)]
    dihedral_score = 0
    if atoms_combos:
        for atoms_combo in atoms_combos:
            dihedral_score += abs(
                new_dihedral(
                    [
                        positions[atoms_combo[0]],
                        positions[atoms_combo[1]],
                        positions[atoms_combo[2]],
                        positions[atoms_combo[3]],
                    ]
                )
            )
        dihedral_score /= len(atoms_combos)
    return dihedral_score < 6


def all_aromatic_rings_are_planar(mol):
    mol_name = mol.GetProp("name").split()
    mol_rings = mol.GetRingInfo().AtomRings()
    for ring in mol_rings:
        atom_names = [mol_name[i] for i in ring]
        if ring_is_aromatic(ring, atom_names) and not ring_is_planar(mol, ring):
            return False
    return True


def all_dihedrals_of_new_atom_are_ok(mol, dihedrals_df):
    temp_name = mol.GetProp("name").split()
    positions = mol.GetConformer().GetPositions()
    atom1 = mol.GetAtomWithIdx(len(mol.GetAtoms()) - 1)
    idx1 = atom1.GetIdx()
    name1 = temp_name[idx1]
    pos1 = positions[idx1]
    for atom2 in atom1.GetNeighbors():
        idx2 = atom2.GetIdx()
        name2 = temp_name[idx2]
        pos2 = positions[idx2]
        for atom3 in [i for i in atom2.GetNeighbors() if i.GetIdx() != idx1]:
            idx3 = atom3.GetIdx()
            name3 = temp_name[idx3]
            pos3 = positions[idx3]
            for atom4 in [i for i in atom3.GetNeighbors() if i.GetIdx() != idx2]:
                idx4 = atom4.GetIdx()
                name4 = temp_name[idx4]
                pos4 = positions[idx4]
                full_name = f"{name1} {name2} {name3} {name4}"
                full_name_reverse = f"{name4} {name3} {name2} {name1}"
                dihedral = abs(new_dihedral([pos1, pos2, pos3, pos4]))
                dihedral = int(dihedral - dihedral % 10)
                if full_name in dihedrals_df.index:
                    if not dihedrals_df.loc[full_name, str(dihedral)]:
                        return False
                elif full_name_reverse in dihedrals_df.index:
                    if not dihedrals_df.loc[full_name_reverse, str(dihedral)]:
                        return False
                else:
                    return False
    return True


def intramolecular_clashes(mol):
    positions = mol.GetConformer().GetPositions()
    for atom1 in mol.GetAtoms():
        atom1_idx = atom1.GetIdx()
        env1 = [a.GetIdx() for a in atom1.GetNeighbors()] + [atom1.GetIdx()]
        symbol = atom1.GetSymbol()
        rms_coord = positions[atom1_idx]
        for atom2 in mol.GetAtoms():
            atom2_idx = atom2.GetIdx()
            env2 = [a.GetIdx() for a in atom2.GetNeighbors()] + [atom2.GetIdx()]
            if atom2_idx > atom1_idx and not set(env1).intersection(set(env2)):
                symbol2 = atom2.GetSymbol()
                rms_coord2 = positions[atom2_idx]
                dist = ((rms_coord - rms_coord2) ** 2).sum() ** 0.5
                combo = symbol + symbol2
                if (
                    combo in ["CC", "CN", "NC", "SH", "HS"]
                    and dist < 2.0
                    or combo in ["NN", "NO", "ON"]
                    and dist < 2.2
                    or combo in ["CO", "OC", "OO"]
                    and dist < 2.0
                    or combo in ["CS", "SC", "NS", "SN"]
                    and dist < 2.2
                    or combo in ["CCl", "ClC", "NCl", "ClN", "OCl", "ClO"]
                    and dist < 2.6
                    or combo in ["CH", "HC"]
                    and dist < 2.0
                    or combo in ["OH", "HO", "ClH", "HCl"]
                    and dist < 1.9
                    or combo in ["NH", "HN"]
                    and dist < 2.0
                    or combo in ["OS", "SO"]
                    and dist < 2.5
                    or combo == "HH"
                    and dist < 1.7
                    or combo in ["SS", "ClCl", "ClS", "SCl"]
                    and dist < 3.1
                    or combo in ["HBr", "BrH"]
                    and dist < 2.05
                    or combo in ["IH", "HI"]
                    and dist < 2.2
                    or combo in ["BrC", "CBr", "BrN", "NBr", "BrO", "OBr"]
                    and dist < 2.5
                    or combo in ["BrS", "SBr", "IC", "CI", "IO", "OI", "IN", "NI", "ClBr", "BrCl"]
                    and dist < 2.7
                    or combo in ["IS", "SI", "ICl", "ClI"]
                    and dist < 2.9
                    or combo in ["IBr", "BrI", "II"]
                    and dist < 3.1
                ):
                    return True
    return False


def intramolecular_clashes_building(mol, next_task):
    if next_task == "no_task":
        mol = Chem.AddHs(mol, addCoords=True)
    positions = mol.GetConformer().GetPositions()
    for atom1 in mol.GetAtoms():
        atom1_idx = atom1.GetIdx()
        env1 = [a.GetIdx() for a in atom1.GetNeighbors()] + [atom1.GetIdx()]
        symbol = atom1.GetSymbol()
        rms_coord = positions[atom1_idx]
        for atom2 in mol.GetAtoms():
            atom2_idx = atom2.GetIdx()
            env2 = [a.GetIdx() for a in atom2.GetNeighbors()] + [atom2.GetIdx()]
            if atom2_idx > atom1_idx and not set(env1).intersection(set(env2)):
                symbol2 = atom2.GetSymbol()
                rms_coord2 = positions[atom2_idx]
                dist = ((rms_coord - rms_coord2) ** 2).sum() ** 0.5
                combo = symbol + symbol2
                if (
                    combo in ["CC", "CN", "NC", "SH", "HS"]
                    and dist < 2.0
                    or combo in ["NN", "NO", "ON"]
                    and dist < 2.2
                    or combo in ["CO", "OC", "OO"]
                    and dist < 2.0
                    or combo in ["CS", "SC", "NS", "SN"]
                    and dist < 2.2
                    or combo in ["CCl", "ClC", "NCl", "ClN", "OCl", "ClO"]
                    and dist < 2.6
                    or combo in ["CH", "HC"]
                    and dist < 2.2
                    or combo in ["OH", "HO", "ClH", "HCl"]
                    and dist < 1.9
                    or combo in ["NH", "HN"]
                    and dist < 2.0
                    or combo in ["OS", "SO"]
                    and dist < 2.5
                    or combo == "HH"
                    and dist < 1.5
                    or combo in ["SS", "ClCl", "ClS", "SCl"]
                    and dist < 3.1
                    or combo in ["HBr", "BrH"]
                    and dist < 2.05
                    or combo in ["IH", "HI"]
                    and dist < 2.2
                    or combo in ["BrC", "CBr", "BrN", "NBr", "BrO", "OBr"]
                    and dist < 2.5
                    or combo in ["BrS", "SBr", "IC", "CI", "IO", "OI", "IN", "NI", "ClBr", "BrCl"]
                    and dist < 2.7
                    or combo in ["IS", "SI", "ICl", "ClI"]
                    and dist < 2.9
                    or combo in ["IBr", "BrI", "II"]
                    and dist < 3.1
                ):
                    return True
    return False


def critical_clashes(lig, protein_df, current_task):
    ser = pd.Series(
        {
            "C C": 1.50,
            "C N": 1.46,
            "C O": 1.38,
            "C S": 1.78,
            "C Cl": 1.74,
            "C H": 1.07,
            "C Br": 1.89,
            "C I": 2.08,
            "N N": 1.42,
            "N O": 1.34,
            "N S": 1.74,
            "N Cl": 1.70,
            "N H": 1.03,
            "N Br": 1.85,
            "N I": 2.04,
            "S S": 2.06,
            "S O": 1.66,
            "S Cl": 2.02,
            "S H": 1.36,
            "S Br": 2.17,
            "S I": 2.36,
            "O O": 1.26,
            "O Cl": 1.62,
            "O H": 0.95,
            "O Br": 1.77,
            "O I": 1.96,
            "Cl Cl": 1.98,
            "Cl H": 1.31,
            "Cl Br": 2.13,
            "Cl I": 2.32,
            "H H": 0.64,
            "H Br": 1.46,
            "H I": 1.65,
            "Br Br": 2.28,
            "Br I": 2.47,
            "I I": 2.66,
        }
    )
    used_atoms = ["C", "N", "O", "S", "H", "Cl", "Br", "I"]
    atom_name_replace = {"P": "S", "B": "C", "Se": "S"}

    if current_task == "no_task":
        lig = Chem.AddHs(lig, addCoords=True)
    lig_coords = lig.GetConformer().GetPositions()
    lig_atoms = [atom.GetSymbol() if atom.GetSymbol() in used_atoms else "H" for atom in lig.GetAtoms()]
    lig_prot_dists = euclidean_distances(lig_coords, lig_coords)
    dists_df = pd.DataFrame(lig_prot_dists)
    for idx1, _ in enumerate(lig.GetAtoms()):
        for idx2, _ in enumerate(lig.GetAtoms()):
            if idx2 > idx1 and not lig.GetBondBetweenAtoms(idx1, idx2):
                dist = dists_df[idx1][idx2]
                atom_atm = lig_atoms[idx1] + " " + lig_atoms[idx2]
                atm_atom = lig_atoms[idx2] + " " + lig_atoms[idx1]
                limit = ser[atom_atm] if atom_atm in ser else ser[atm_atom]
                if limit > dist:
                    return True

    prot_atoms = protein_df["atom_name"].apply(lambda x: x[0] if x[0].isalpha() else x[1])
    prot_atoms = [atom_name_replace.get(name, name) for name in prot_atoms]
    prot_atoms = ["H" if name not in used_atoms else name for name in prot_atoms]
    prot_coords = np.array(protein_df[["x_coord", "y_coord", "z_coord"]])

    lig_prot_dists = euclidean_distances(lig_coords, prot_coords)
    dists_df = pd.DataFrame(lig_prot_dists, index=lig_atoms, columns=prot_atoms)
    unique_probe_atoms, unique_prot_atoms = dists_df.index.unique(), dists_df.columns.unique()

    for code, limit in ser.items():
        atom_name_1, atom_name_2 = code.split(" ")
        if atom_name_1 in unique_probe_atoms and atom_name_2 in unique_prot_atoms and (dists_df.loc[atom_name_1, atom_name_2] < limit).astype(int).sum().sum() > 0:
            return True

        if (
            atom_name_2 in unique_probe_atoms
            and atom_name_1 in unique_prot_atoms
            and atom_name_2 != atom_name_1
            and (dists_df.loc[atom_name_2, atom_name_1] < limit).astype(int).sum().sum() > 0
        ):
            return True

    return False


def all_valencies_are_ok(frag):
    frag_name = frag.GetProp("name").split()
    for atom in frag.GetAtoms():
        num_of_not_single_bonds = 0
        num_neighbors = len(atom.GetNeighbors())
        atom_idx = atom.GetIdx()

        if (frag_name[atom_idx] in ["Nd0", "Sul"]) and sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) > 2:
            return False

        if (frag_name[atom_idx] == "Nac") and (num_neighbors == 3):
            for neighb in atom.GetNeighbors():
                bond_type = frag.GetBondBetweenAtoms(atom_idx, neighb.GetIdx()).GetBondType()
                if str(bond_type) == "DOUBLE":
                    return False

        if atom.GetSymbol() != "S" and num_neighbors > 1:
            for neighb in atom.GetNeighbors():
                bond_type = frag.GetBondBetweenAtoms(atom_idx, neighb.GetIdx()).GetBondType()
                if str(bond_type) != "SINGLE":
                    num_of_not_single_bonds += 1
        if num_of_not_single_bonds > 1:
            return False
        if frag_name[atom_idx] in ["C2r", "Cs2", "Car"] and num_neighbors > 3:
            return False

    return True


def calc_angle(rms_coord, rms_coord2, rms_coord3):
    a = np.array(rms_coord3)
    b = np.array(rms_coord2)
    c = np.array(rms_coord)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def angles_for_SP2_atoms_with_3_neighbors_are_ok(mol):
    res = True
    coords = mol.GetConformer().GetPositions()
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) == 3 and str(atom.GetHybridization()) == "SP2":
            angles = []
            hood_coords = coords[[a.GetIdx() for a in atom.GetNeighbors()] + [atom.GetIdx()]]
            angles.append(calc_angle(hood_coords[0], hood_coords[3], hood_coords[1]))
            angles.append(calc_angle(hood_coords[0], hood_coords[3], hood_coords[2]))
            angles.append(calc_angle(hood_coords[1], hood_coords[3], hood_coords[2]))
            res = all(angle < 160 for angle in angles)
            if not res:
                break
    return res


# функция для проверки дубликатов
def duplicates_check(new_mol3, compare):
    rms2_temp = 100
    final_local_difference = 100
    new_mol3_with_Rs = Chem.Mol(new_mol3)
    new_positions = new_mol3_with_Rs.GetConformer().GetPositions()
    for mmm in compare:
        if (
            new_mol3_with_Rs.HasSubstructMatch(mmm)
            and mmm.HasSubstructMatch(new_mol3_with_Rs)
            and sorted(new_mol3_with_Rs.GetProp("name").split()) == sorted(mmm.GetProp("name").split())
        ):
            atoms_list = mmm.GetSubstructMatches(new_mol3_with_Rs, uniquify=False)  #: List[int]
            mmm_copy = Chem.Mol(mmm)
            for atoms in atoms_list:
                mmm_renumbered = Chem.RenumberAtoms(mmm_copy, atoms)
                positions = mmm_renumbered.GetConformer().GetPositions()
                n_atom = 0
                rms2 = 0
                biggest_local_difference = 0
                fixed_atoms_count = 0
                while n_atom < len(new_mol3_with_Rs.GetAtoms()):
                    fixed_atoms_count += 1
                    rms_coord2 = positions[n_atom]
                    rms_coord = new_positions[n_atom]
                    temp_dist = ((rms_coord - rms_coord2) ** 2).sum() ** 0.5
                    biggest_local_difference = max((temp_dist) ** 0.5, biggest_local_difference)
                    rms2 = rms2 + temp_dist
                    n_atom += 1
                rms2 = (rms2 / fixed_atoms_count) ** 0.5
                if rms2 < rms2_temp:
                    rms2_temp = rms2
                    final_local_difference = biggest_local_difference
    return (rms2_temp, final_local_difference)


def duplicates_check_local(mols):
    result = []
    checked = set()
    for i, mol_1 in enumerate(mols):
        if i in checked:
            continue
        compare = [mol_1]
        checked.add(i)
        new_atom = mol_1.GetAtomWithIdx(len(mol_1.GetAtoms()) - 1)
        new_atom_type = mol_1.GetProp("name").split()[-1]
        if new_atom_type == "Hal":
            new_atom_type = new_atom.GetSymbol()
        new_atom_neighbs = sorted([n.GetIdx() for n in new_atom.GetNeighbors()])
        new_atom_bonds = [mol_1.GetBondBetweenAtoms(new_atom.GetIdx(), n).GetBondTypeAsDouble() for n in new_atom_neighbs]

        for j, mol_2 in enumerate(mols):
            if j in checked:
                continue
            atom = mol_2.GetAtomWithIdx(len(mol_2.GetAtoms()) - 1)
            atom_type = mol_2.GetProp("name").split()[-1]
            if atom_type == "Hal":
                atom_type = atom.GetSymbol()
            atom_neighbs = sorted([n.GetIdx() for n in atom.GetNeighbors()])
            atom_bonds = [mol_2.GetBondBetweenAtoms(atom.GetIdx(), n).GetBondTypeAsDouble() for n in atom_neighbs]
            if new_atom_type == atom_type and all(a == b for a, b in zip(new_atom_neighbs, atom_neighbs)) and all(a == b for a, b in zip(new_atom_bonds, atom_bonds)):
                checked.add(j)
                compare.append(mol_2)

        coords = [mol.GetConformer().GetPositions()[-1] for mol in compare]
        clustering = DBSCAN(eps=0.2, min_samples=1).fit(coords)
        labels = pd.Series(clustering.labels_)
        for ii in labels.unique():
            index = labels[labels == ii].index
            cluster = [compare[k] for k in index]
            if len(cluster) == 1:
                result.extend(cluster)
            else:
                cluster_coords = [coords[k] for k in index]
                point = np.array(cluster_coords).mean(axis=0)
                dists = [np.linalg.norm(point-temp_coords) for temp_coords in cluster_coords]
                needed_mol = cluster[dists.index(min(dists))]
                result.append(needed_mol)
    return result


def ligand_is_within_grid(coords1, coords2):
    dists = euclidean_distances(coords1, coords2)
    return (dists < 1.3).any(axis=1).all()


def lig_prot_repulsion(probe, protein_df, forbidden):
    # probe = Chem.AddHs(probe)
    probe_name = probe.GetProp("name").split()
    probe_positions = probe.GetConformer().GetPositions()
    lig_don_list = []
    lig_acc_list = []
    for atom in probe.GetAtoms():
        # if atom.GetSymbol() == "H":
        #     continue
        atom_idx = atom.GetIdx()
        if atom_idx not in forbidden:
            if probe_name[atom_idx] in ["Nd0", "Nd+"]:
                lig_don_list.append(probe_positions[atom_idx])
            elif probe_name[atom_idx] in ["Nac", "O_a", ".=O"]: #and not [a for a in atom.GetNeighbors() if a.GetSymbol() == "H"]:
                lig_acc_list.append(probe_positions[atom_idx])

    if lig_don_list:
        lig_don_array = np.array(lig_don_list)
        prot_don_mask = (
            (protein_df["atom_name"].isin(["N"]))
            | (protein_df["residue_name"].isin(["ASN"]) & protein_df["atom_name"].isin(["ND2"]))
            | (protein_df["residue_name"].isin(["GLN"]) & protein_df["atom_name"].isin(["NE2"]))
            | (protein_df["residue_name"].isin(["ARG"]) & protein_df["atom_name"].isin(["NE", "NH1", "NH2"]))
            | (protein_df["residue_name"].isin(["LYS"]) & protein_df["atom_name"].isin(["NZ"]))
            | (protein_df["residue_name"].isin(["TRP"]) & protein_df["atom_name"].isin(["NE1"]))
        )

        prot_don_array = protein_df[prot_don_mask][["x_coord", "y_coord", "z_coord"]]
        don_distances = euclidean_distances(lig_don_array, prot_don_array)
    else:
        don_distances = np.array(10)

    if lig_acc_list:
        lig_acc_array = np.array(lig_acc_list)
        prot_acc_mask = (
            (protein_df["atom_name"].isin(["O"]))
            | (protein_df["residue_name"].isin(["GLU"]) & protein_df["atom_name"].isin(["OE1", "OE2"]))
            | (protein_df["residue_name"].isin(["GLN"]) & protein_df["atom_name"].isin(["OD1"]))
            | (protein_df["residue_name"].isin(["ASP"]) & protein_df["atom_name"].isin(["OD1", "OD2"]))
            | (protein_df["residue_name"].isin(["ASN"]) & protein_df["atom_name"].isin(["OD1"]))
        )
        prot_acc_array = protein_df[prot_acc_mask][["x_coord", "y_coord", "z_coord"]]
        acc_distances = euclidean_distances(lig_acc_array, prot_acc_array)
    else:
        acc_distances = np.array(10)
    return (don_distances < 3.3).any() or (acc_distances < 2.8).any()  # don 2.9 for initial generation in case study


def make_cycle_where_possible(probe):
    probe_name = probe.GetProp("name").split()
    probe_positions = probe.GetConformer().GetPositions()
    forbidden_atoms = [".=O", "Hal", "Csp", "Cs2", "Cs3", "O_d"]
    for atom in probe.GetAtoms():
        atom_idx = atom.GetIdx()
        if probe_name[atom_idx] not in forbidden_atoms:
            num_neighbors = len(atom.GetNeighbors())
            if (
                (probe_name[atom_idx] == "Nac" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()), 3.0))
                or (probe_name[atom_idx] == "Nd0" and num_neighbors == 2)
                or (atom.GetSymbol() == "C" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()), 4.0))
                or (probe_name[atom_idx] == "Sul" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()), 2.0))
                or (probe_name[atom_idx] == "SO2" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()), 6.0))
                or (atom.GetSymbol() == "O" and num_neighbors == 2)
                or (atom.GetSymbol() == "N" and num_neighbors == 3)
            ):
                continue
            rms_coord = probe_positions[atom_idx]
            for atom2 in probe.GetAtoms():
                atom2_idx = atom2.GetIdx()
                if atom2_idx <= atom_idx:
                    continue
                if probe_name[atom2_idx] not in forbidden_atoms:
                    num_neighbors2 = len(atom2.GetNeighbors())
                    if (
                        (probe_name[atom2_idx] == "Nac" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom2.GetBonds()), 3.0))
                        or (probe_name[atom2_idx] == "Nd0" and num_neighbors2 == 2)
                        or (atom.GetSymbol() == "C" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()), 4.0))
                        or (probe_name[atom_idx] == "Sul" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()), 2.0))
                        or (probe_name[atom_idx] == "SO2" and np.isclose(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()), 6.0))
                        or (atom2.GetSymbol() == "O" and num_neighbors2 == 2)
                        or (atom2.GetSymbol() == "N" and num_neighbors2 == 3)
                    ):
                        continue
                    if not probe.GetBondBetweenAtoms(atom_idx, atom2_idx):
                        rms_coord2 = probe_positions[atom2.GetIdx()]
                        dist = ((rms_coord - rms_coord2) ** 2).sum() ** 0.5
                        if dist > 1.2 and dist < 1.8:
                            probe_copy = Chem.Mol(probe)
                            edmol = Chem.EditableMol(probe_copy)
                            if (
                                probe_name[atom_idx] == "Car"
                                and probe_name[atom2_idx] == "Car"
                                and not any(b.GetBondType() is Chem.BondType.DOUBLE for b in atom.GetBonds())
                                and not any(b.GetBondType() is Chem.BondType.DOUBLE for b in atom2.GetBonds())
                            ):
                                edmol.AddBond(atom_idx, atom2_idx, Chem.BondType.DOUBLE)
                            else:
                                edmol.AddBond(atom_idx, atom2_idx, Chem.BondType.SINGLE)
                            probe_new = edmol.GetMol()
                            probe_new.UpdatePropertyCache()
                            # if angles_are_ok(probe_new, probe_new.GetAtomWithIdx(atom.GetIdx())) and angles_are_ok(probe_new, probe_new.GetAtomWithIdx(atom2.GetIdx())):
                            probe_new = optimize_ring(atom.GetIdx(), atom2.GetIdx(), probe_new)
                            probe_new.ClearProp("initiate_ring_bisect")
                            probe_new.ClearProp("initiate_ring_110")
                            return probe_new
    return probe


def atoms_in_ring(idx1, idx2, mol):
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if idx1 in ring and idx2 in ring:
            out = tuple(ring)
            break
    return out


def rings_are_adjacent(ring1, ring2, min_common_atoms=2):
    return len(set(ring1) & set(ring2)) >= min_common_atoms


def collect_conjugated_ring_system_atoms(mol, seed_ring, min_common_atoms=2):
    """
    Собирает всю связную систему циклов:
    seed_ring + все соседние циклы + соседи соседей и т.д.
    """
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings()]
    seed_ring = tuple(seed_ring)

    visited = set()
    queue = deque([seed_ring])
    collected_atoms = set(seed_ring)

    def ring_key(r):
        return tuple(sorted(r))

    visited.add(ring_key(seed_ring))

    while queue:
        current_ring = queue.popleft()

        for ring in rings:
            rk = ring_key(ring)
            if rk in visited:
                continue

            if rings_are_adjacent(current_ring, ring, min_common_atoms=min_common_atoms):
                visited.add(rk)
                queue.append(ring)
                collected_atoms.update(ring)

    return sorted(collected_atoms)


def collect_fragment_bonds(mol, atom_ids):
    atom_set = set(atom_ids)
    bond_ids = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in atom_set and a2 in atom_set:
            bond_ids.append(bond.GetIdx())
    return bond_ids


def extract_fragment_with_parent_mapnums(mol, atom_ids):
    """
    Вырезает фрагмент и сохраняет в его атомах parent idx через AtomMapNum = parent_idx + 1.
    """
    bond_ids = collect_fragment_bonds(mol, atom_ids)
    if len(bond_ids) == 0:
        return None

    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    frag = Chem.PathToSubmol(mol_copy, bond_ids)
    return frag



def build_ref_to_parent_atom_map(ref_mol, frag_mol):
    """
    Строит ИТОГОВЫЙ atom_map:
    (ref_idx, parent_idx)
    """

    atom_map = []
    for ref_atom in ref_mol.GetAtoms():
        parent_idx = ref_atom.GetAtomMapNum() - 1
        atom_map.append((ref_atom.GetIdx(), parent_idx))

    return atom_map

def copy_coords_from_aligned_ref_to_parent(mol, ref_aligned, ref_to_parent_map):
    conf = mol.GetConformer()
    ref_coords = ref_aligned.GetConformer().GetPositions().astype(np.double)

    for ref_idx, parent_idx in ref_to_parent_map:
        x, y, z = ref_coords[ref_idx]
        conf.SetAtomPosition(parent_idx, Point3D(float(x), float(y), float(z)))

    return mol


def optimize_ring(idx1, idx2, mol, min_common_atoms=2):
    """
    Логика:
    1. Находим только что закрытый цикл
    2. Собираем всю связанную систему циклов
    3. Вырезаем фрагмент из mol
    4. Генерируем mapped SMILES
    5. Строим ref_mol из smiles
    6. Собираем итоговый map (ref_idx, parent_idx)
    7. Выравниваем ref_mol СРАЗУ на parent mol
    8. Копируем координаты назад
    """
    Chem.SanitizeMol(mol)
    candidate_ring = atoms_in_ring(idx1, idx2, mol)
    if len(candidate_ring) == 0:
        return mol

    fragment_atoms = collect_conjugated_ring_system_atoms(
        mol,
        candidate_ring,
        )

    frag_mol = extract_fragment_with_parent_mapnums(mol, fragment_atoms)

    # Для SMILES делаем локальную перенумерацию atom maps: local idx + 1
    smiles = Chem.MolToSmiles(frag_mol)
    ref_mol = generate_conformers_from_smiles(smiles, 1, 10, 0.25)[0]
    ref_mol = Chem.RemoveHs(ref_mol)


    ref_to_parent_map = build_ref_to_parent_atom_map(ref_mol, frag_mol)

    ref_copy = Chem.Mol(ref_mol)

    AllChem.GetBestRMS(ref_copy, mol, map=[ref_to_parent_map])


    res_mol = copy_coords_from_aligned_ref_to_parent(mol, ref_copy, ref_to_parent_map)

    return res_mol


def incorrect_arom_ring_closing(probe):
    probe_name = probe.GetProp("name").split()
    tasks = probe.GetProp("tasks").split()
    if all(t == "close_aromatic_ring" for t in tasks[-2:]):
        probe_positions = probe.GetConformer().GetPositions()
        atom1 = [atom for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] != ".=O"][-1]
        for atom2 in atom1.GetNeighbors():
            if probe_name[atom2.GetIdx()] in ["Car", "O_a", "Nac", "Nd0", "Sul", "C3r"] and not atom2.IsInRing():
                for atom3 in atom2.GetNeighbors():
                    if probe_name[atom3.GetIdx()] in ["Car", "O_a", "Nac", "Nd0", "Sul", "C3r"] and not atom3.IsInRing() and atom3.GetIdx() != atom1.GetIdx():
                        for atom4 in atom3.GetNeighbors():
                            if probe_name[atom4.GetIdx()] in ["Car", "O_a", "Nac", "Nd0", "Sul", "C3r"] and not atom4.IsInRing() and atom4.GetIdx() != atom2.GetIdx():
                                coords1 = probe_positions[atom1.GetIdx()]
                                coords2 = probe_positions[atom2.GetIdx()]
                                coords3 = probe_positions[atom3.GetIdx()]
                                coords4 = probe_positions[atom4.GetIdx()]
                                dihedral = abs(new_dihedral([coords1, coords2, coords3, coords4]))
                                ang1 = calc_angle(coords1, coords2, coords3)
                                ang2 = calc_angle(coords2, coords3, coords4)
                                if dihedral > 10 or (ang1 > 113 and ang2 < 113) or (ang1 < 113 and ang2 > 113):
                                    return True
    return not atom_is_planar(probe, probe.GetAtomWithIdx(len(probe.GetAtoms()) - 1).GetNeighbors()[0])


def incorrect_aliph_ring_closing(probe):
    probe_name = probe.GetProp("name").split()
    tasks = probe.GetProp("tasks").split()
    if all(t == "close_aliphatic_ring" for t in tasks[-2:]):
        probe_positions = probe.GetConformer().GetPositions()
        atom1 = list(probe.GetAtoms())[-1]
        for atom2 in atom1.GetNeighbors():
            if probe_name[atom2.GetIdx()] in ["C3r", "Nd+", "Nac", "O_a", "C2r", "SO2", "Nd0"] and not atom2.IsInRing():
                for atom3 in atom2.GetNeighbors():
                    if probe_name[atom3.GetIdx()] in ["C3r", "Nd+", "Nac", "O_a", "C2r", "SO2", "Nd0"] and not atom3.IsInRing() and atom3.GetIdx() != atom1.GetIdx():
                        for atom4 in atom3.GetNeighbors():
                            if probe_name[atom4.GetIdx()] in ["C3r", "Nd+", "Nac", "O_a", "C2r", "SO2", "Nd0"] and not atom4.IsInRing() and atom4.GetIdx() != atom2.GetIdx():
                                coords1 = probe_positions[atom1.GetIdx()]
                                coords2 = probe_positions[atom2.GetIdx()]
                                coords3 = probe_positions[atom3.GetIdx()]
                                coords4 = probe_positions[atom4.GetIdx()]
                                dihedral = abs(new_dihedral([coords1, coords2, coords3, coords4]))
                                if dihedral > 90:
                                    return True
        i = -1
        while True:
            if tasks[i] == "close_aliphatic_ring":
                i -= 1
            else:
                break
        ring_atoms = [a.GetIdx() for a in probe.GetAtoms()][i:]
        combos = combinations(ring_atoms, 2)
        for combo in combos:
            if ((probe_positions[combo[0]] - probe_positions[combo[1]]) ** 2).sum() ** 0.5 > 3.5:
                return True
    return False


def valencies_after_cyclization_are_ok(frag):
    return all(not (atom.GetSymbol() in ["C", "N"] and len(atom.GetNeighbors()) >= 5 or atom.GetSymbol() == "O" and len(atom.GetNeighbors()) >= 3) for atom in frag.GetAtoms())


def is_in_planar_ring(atom, mol):
    rings = mol.GetRingInfo().AtomRings()
    return any(atom.GetIdx() in ring and ring_is_planar(mol, ring) for ring in rings)


def atom_is_planar(mol, atom, border=0.125):
    positions = mol.GetConformer().GetPositions()
    if len(atom.GetNeighbors()) != 3:
        return True
    coords = [positions[atom.GetIdx()]] + [positions[atom2.GetIdx()] for atom2 in atom.GetNeighbors()]
    # measure dist between atom and plane of its neighbors
    A, B, C = np.cross(coords[2] - coords[1], coords[3] - coords[1])
    D = -np.dot(np.array(coords[1]), np.array([A, B, C]))
    dist = np.abs(np.dot(np.array([A, B, C]), coords[0]) + D) / np.sqrt(A**2 + B**2 + C**2)
    return dist <= border


def all_Cs2_are_planar(mol, forbidden):
    mol.GetConformer().GetPositions()
    name = mol.GetProp("name").split()
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in forbidden and atom.GetSymbol() != "H":
            atom_idx = atom.GetIdx()
            atom_type = name[atom_idx]
            if (atom_type in ["Car", "Cs2", "C2r"] or is_in_planar_ring(atom, mol)) and not atom_is_planar(mol, atom):
                return False
    return True


def ring_is_aromatic(ring, atom_names):
    return (
        Counter(atom_names)["C3r"] <= 1
        and len(ring) in [5, 6]  # удалить 7
        and all(n in ["Car", "C3r", "Nac", "Nd0", "Sul", "O_a"] for n in atom_names)
    )


def all_needed_N_are_planar(mol):
    atom_names_all = mol.GetProp("name").split()
    res = True
    ring_tuple = mol.GetRingInfo().AtomRings()
    for atom in mol.GetAtoms():
        if not res:
            break

        if atom.GetSymbol() == "N" and len(atom.GetNeighbors()) == 3:
            index = atom.GetIdx()

            neighbours = any(i in [atom_names_all[atom2.GetIdx()] for atom2 in atom.GetNeighbors()] for i in ["Car", "Cs2", "C2r"])

            in_ring = False

            if ring_tuple != ():
                for the_ring in [ring for ring in ring_tuple if (index in ring)]:
                    atom_names = [atom_names_all[i] for i in the_ring]
                    in_ring = ring_is_aromatic(the_ring, atom_names)

                    if in_ring:
                        break

            if neighbours or in_ring:
                res = atom_is_planar(mol, atom, border=0.18)
    return res


def count_chiral_centers(mol):
    name = mol.GetProp("name").split()
    counter = 0
    info = rdmolops.FindPotentialStereo(mol)
    if len(info):
        for line in info:
            if str(line.type) == "Atom_Tetrahedral" and name[line.centeredOn] in ["Cs3", "C3r"]:
                counter += 1
    return counter


def mol_has_CNdSO2_with_bad_geometry(mol, forbidden):
    decision = False
    if mol.HasSubstructMatch(Chem.MolFromSmiles("CNS(=O)=O")):
        name = mol.GetProp("name").split()
        positions = mol.GetConformer().GetPositions()
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in forbidden and atom.GetSymbol() == "S":
                atom_idx = atom.GetIdx()
                atom_type = name[atom_idx]
                if atom_type == "SO2" and len(atom.GetNeighbors()) >= 3:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == "N":
                            neighbor_idx = neighbor.GetIdx()
                            atom_type = name[neighbor_idx]
                            if atom_type == "Nd0" and len(neighbor.GetNeighbors()) == 2:
                                decision = True
                                for oxygen in atom.GetNeighbors():
                                    if oxygen.GetIdx() != neighbor_idx and oxygen.GetSymbol() == "O":
                                        for carbon in neighbor.GetNeighbors():
                                            if carbon.GetIdx() != atom_idx:
                                                coords1 = positions[carbon.GetIdx()]
                                                coords2 = positions[neighbor_idx]
                                                coords3 = positions[atom_idx]
                                                coords4 = positions[oxygen.GetIdx()]
                                                dihedral = abs(new_dihedral([coords1, coords2, coords3, coords4])) // 10
                                                if dihedral == 17:
                                                    decision = False
    return decision


def mol_has_acyclic_cis_amide(mol):
    name = mol.GetProp("name").split()
    combos = mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)N"))
    positions = mol.GetConformer().GetPositions()
    for atoms in combos:
        # skip cyclic amides
        if mol.GetAtomWithIdx(atoms[0]).IsInRing() and mol.GetAtomWithIdx(atoms[2]).IsInRing():
            continue
        if name[atoms[0]] in ["Car", "C2r"]:
            continue
        # skip sully substituted Ns
        if len(mol.GetAtomWithIdx(atoms[2]).GetNeighbors()) != 2:
            continue
        # skip carboxyls
        if len([n for n in mol.GetAtomWithIdx(atoms[0]).GetNeighbors() if n.GetSymbol() == "O"]) > 1:
            continue
        coords1 = positions[atoms[1]]
        coords2 = positions[atoms[0]]
        coords3 = positions[atoms[2]]
        coords4 = positions[next(n.GetIdx() for n in mol.GetAtomWithIdx(atoms[2]).GetNeighbors() if n.GetIdx() != atoms[0])]
        dihedral = abs(new_dihedral([coords1, coords2, coords3, coords4]))
        if dihedral > 45:
            return True
    return False


def mol_has_Cs2_with_3_neighbors_and_valence_3(mol):
    name = mol.GetProp("name").split()
    for atom in mol.GetAtoms():
        atom_type = name[atom.GetIdx()]
        if atom_type in ["Cs2", "C2r", "Car"] and len(atom.GetNeighbors()) == 3 and atom.GetExplicitValence() == 3:
            return True
    return False


def calculate_max_chain_length(name, mol, atom_type, atom, visited_atoms, current_length):
    max_length = current_length
    visited_atoms.add(atom.GetIdx())
    for neighbor in atom.GetNeighbors():
        atom_type = name[neighbor.GetIdx()]
        if neighbor.GetIdx() not in visited_atoms and not neighbor.IsInRing() and atom_type not in ["Car", "C3r", "C2r", "O_d", "Hal", ".=O"]:
            max_length = max(
                max_length,
                calculate_max_chain_length(name, mol, atom_type, neighbor, visited_atoms, current_length + 1),
            )
    return max_length


def mol_has_optimal_acyclic_chains_length(mol, max_len):
    name = mol.GetProp("name").split()
    max_chain_lengths = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_type = name[atom_idx]
        if not atom.IsInRing() and atom_type not in ["Car", "C3r", "C2r", "O_d", "Hal", ".=O"]:
            visited_atoms = set()
            max_chain_lengths.append(calculate_max_chain_length(name, mol, atom_type, atom, visited_atoms, 1))
    return max(max_chain_lengths) <= max_len if max_chain_lengths else True


def intramol_repulsion(mol):
    mol = Chem.AddHs(mol, addCoords=True)
    mol_name = mol.GetProp("name").split()
    conf = mol.GetConformer().GetPositions()
    for atom1 in mol.GetAtoms():
        if atom1.GetSymbol() != "H":
            if mol_name[atom1.GetIdx()] in [".=O", "Nac", "O_a"]:
                env1 = [a.GetIdx() for a in atom1.GetNeighbors()] + [atom1.GetIdx()]
                for atom2 in mol.GetAtoms():
                    if atom2.GetSymbol() != "H" and mol_name[atom2.GetIdx()] in [".=O", "Nac", "O_a"]:
                        env2 = [a.GetIdx() for a in atom2.GetNeighbors()] + [atom2.GetIdx()]
                        if not set(env1).intersection(set(env2)):
                            dist = (np.array(conf[atom1.GetIdx()] - conf[atom2.GetIdx()]) ** 2).sum() ** 0.5
                            if dist < 2.7:
                                return True
            if mol_name[atom1.GetIdx()] in ["Nd0", "Nd+"] and "H" in [a.GetSymbol() for a in atom1.GetNeighbors()]:
                env1 = [a.GetIdx() for a in atom1.GetNeighbors()] + [atom1.GetIdx()]
                for atom2 in mol.GetAtoms():
                    if atom2.GetSymbol() != "H" and mol_name[atom2.GetIdx()] in ["Nd0", "Nd+"] and "H" in [a.GetSymbol() for a in atom2.GetNeighbors()]:
                        env2 = [a.GetIdx() for a in atom2.GetNeighbors()] + [atom2.GetIdx()]
                        if not set(env1).intersection(set(env2)):
                            for h1 in atom1.GetNeighbors():
                                if h1.GetSymbol() == "H":
                                    for h2 in atom2.GetNeighbors():
                                        if h2.GetSymbol() == "H":
                                            dist = (np.array(conf[h1.GetIdx()] - conf[h2.GetIdx()]) ** 2).sum() ** 0.5
                                            if dist < 2.5:
                                                return True
    return False


def charged_N_with_arom_neighb(mol):
    mol_name = mol.GetProp("name").split()
    for atom1 in mol.GetAtoms():
        if mol_name[atom1.GetIdx()] == "Nd+":
            for atom2 in atom1.GetNeighbors():
                if mol_name[atom2.GetIdx()] == "Car":
                    return True
    return False


def gem_O_d_present(mol):
    mol_name = mol.GetProp("name").split()
    return any(len([neighb for neighb in atom.GetNeighbors() if mol_name[neighb.GetIdx()] == "O_d"]) > 1 for atom in mol.GetAtoms())


def two_O_a_in_one_ring(mol):
    mol_name = mol.GetProp("name").split()
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        O_a_count = 0
        for n in ring:
            if mol_name[n] == "O_a":
                O_a_count += 1
        if O_a_count > 1:
            return True
    return False


def parallel_arom_DB(mol):
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol)
    mol.GetProp("name").split()
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if len(ring) == 6 and (
            (mol.GetBondBetweenAtoms(ring[0], ring[1]).GetBondTypeAsDouble() == 2 and mol.GetBondBetweenAtoms(ring[3], ring[4]).GetBondTypeAsDouble() == 2)
            or (mol.GetBondBetweenAtoms(ring[1], ring[2]).GetBondTypeAsDouble() == 2 and mol.GetBondBetweenAtoms(ring[4], ring[5]).GetBondTypeAsDouble() == 2)
            or (mol.GetBondBetweenAtoms(ring[2], ring[3]).GetBondTypeAsDouble() == 2 and mol.GetBondBetweenAtoms(ring[5], ring[0]).GetBondTypeAsDouble() == 2)
        ):
            return True
    return False


def Nac_are_conjugated(mol):
    mol_name = mol.GetProp("name").split()
    for atom in mol.GetAtoms():
        if mol_name[atom.GetIdx()] == "Nac" and len(atom.GetNeighbors()) == 3:
            neighbs = [mol_name[neighb.GetIdx()] for neighb in atom.GetNeighbors()]
            if "Car" not in neighbs and "Cs2" not in neighbs and "C2r" not in neighbs and "Nac" not in neighbs and "Nd0" not in neighbs and "SO2" not in neighbs:
                return False
    return True


def non_ring_atom_in_ring(mol):
    mol_name = mol.GetProp("name").split()
    return any(mol_name[atom.GetIdx()] in ["Csp", "Cs2", "Cs3"] and atom.IsInRing() for atom in mol.GetAtoms())


def some_ring_substituted_by_more_than_2_Csp3_or_more_than_2_Nd0(mol, forbidden):
    mol_name = mol.GetProp("name").split()
    for ring in mol.GetRingInfo().AtomRings():
        substituents = []
        for index in ring:
            if index not in forbidden:
                for neighbor in mol.GetAtoms()[index].GetNeighbors():
                    substituent_index = neighbor.GetIdx()
                    if substituent_index not in ring:
                        substituent_name = mol_name[substituent_index]
                        substituents.append(substituent_name)
        if substituents.count("Nd0") > 2 or substituents.count("Cs3") > 2:
            return True
    return False


def ring_is_aliphatic(ring, mol):
    for number in ring:
        atom = mol.GetAtomWithIdx(number)
        if atom.GetIsAromatic() or (atom.GetSymbol() == "C" and str(atom.GetHybridization()) != "SP3"):
            return False
    return True


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def axial_groups_present(mol):
    rings = mol.GetRingInfo()
    conf = mol.GetConformer()
    for ring in rings.AtomRings():
        if ring_is_aliphatic(ring, mol) and len(ring) == 6:
            for ID in ring:
                atom = mol.GetAtomWithIdx(ID)
                if len(atom.GetNeighbors()) < 4:
                    for n in atom.GetNeighbors():
                        if n.GetIdx() not in ring and n.GetSymbol() != "H":
                            x1 = conf.GetPositions()[ID]
                            y1 = conf.GetPositions()[n.GetIdx()]
                            v1 = y1 - x1
                            for n2 in atom.GetNeighbors():
                                if n2.GetIdx() in ring:
                                    for n3 in n2.GetNeighbors():
                                        if n3.GetIdx() in ring and n3.GetIdx() != ID:
                                            x2 = conf.GetPositions()[n2.GetIdx()]
                                            y2 = conf.GetPositions()[n3.GetIdx()]
                                            v2 = y2 - x2
                                            angle = np.degrees(angle_between(v1, v2))
                                            if angle > 60 and angle < 120:
                                                return True
    return False


def aliphatic_rings_are_ok(mol):
    rings = mol.GetRingInfo()
    conf = mol.GetConformer()
    for ring in rings.AtomRings():
        if ring_is_aliphatic(ring, mol) and len(ring) == 6:
            atoms = {i: mol.GetAtoms()[ring[i]].GetIdx() for i in range(6)}
            # torsion angle atom indices
            torsions = [
                (atoms[0], atoms[1], atoms[2], atoms[3]),
                (atoms[1], atoms[2], atoms[3], atoms[4]),
                (atoms[2], atoms[3], atoms[4], atoms[5]),
                (atoms[3], atoms[4], atoms[5], atoms[0]),
                (atoms[4], atoms[5], atoms[0], atoms[1]),
                (atoms[5], atoms[0], atoms[1], atoms[2]),
            ]
            angs = np.array([abs(Chem.rdMolTransforms.GetDihedralDeg(conf, *t)) for t in torsions])
            # check if pos/neg values are alternating
            for ang in angs:
                if ang < 40 or ang > 80:
                    return False
    return True


def condensed_aliphatic_rings_present(mol):
    info = mol.GetRingInfo().AtomRings()
    for ring1 in info:
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring1):
            for ring2 in info:
                # if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring2):
                if ring1 != ring2 and len(set(ring1).intersection(set(ring2))) > 1:
                    return True
    return False


def angles_are_ok(frag, atom):
    conf = frag.GetConformer().GetPositions()
    p2 = conf[atom.GetIdx()]
    neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
    pairs = combinations(neighbors, 2)
    angles = [calc_angle(conf[pair[0]], p2, conf[pair[1]]) for pair in pairs]

    frag_name = frag.GetProp("name").split()
    type_ = frag_name[atom.GetIdx()]
    if atom.IsInRingSize(6) or atom.IsInRingSize(5):
        if type_ in ["Sul"]:
            return all(90 < ang < 140 for ang in angles)
        return all(100 < ang < 140 for ang in angles)
    if atom.IsInRingSize(4):
        return all(85 < ang < 95 or 100 < ang < 140 for ang in angles)
    if atom.IsInRingSize(3):
        return all(55 < ang < 65 or 100 < ang < 140 for ang in angles)

    if type_ in ["Csp"]:
        return all(ang > 170 for ang in angles)
    if type_ in ["Cs2"]:
        return all(110 < ang < 135 for ang in angles)
    if type_ in ["Car", "Nd0"]:
        return all(100 < ang < 135 for ang in angles)
    if type_ in ["Cs3"]:
        return all(100 < ang < 125 for ang in angles)
    if type_ in ["SO2", "Sul"]:
        return all(90 < ang < 125 for ang in angles)
    if type_ in ["C3r", "C2r", "Nd+", "O_a", "Nac"]:
        return all(55 < ang < 65 or 85 < ang < 95 or 100 < ang < 135 for ang in angles)
    return None


def paired_Nd0_present(probe):
    probe_name = probe.GetProp("name").split()
    for atom in probe.GetAtoms():
        if probe_name[atom.GetIdx()] == "Nd0":
            for neighb in atom.GetNeighbors():
                if probe_name[neighb.GetIdx()] == "Nd0":
                    return True
    return False


def pair_Nd0_Nac_present(probe):
    probe_name = probe.GetProp("name").split()
    for atom in probe.GetAtoms():
        if probe_name[atom.GetIdx()] == "Nd0":
            for neighb in atom.GetNeighbors():
                if probe_name[neighb.GetIdx()] == "Nac":
                    return True
    return False


def Nd0_with_3_neighbors_present(probe):
    probe_name = probe.GetProp("name").split()
    return any(atom for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Nd0" and len(atom.GetNeighbors()) == 3)


def Car_Nac_DB_dihedrals_are_ok(probe):
    probe_name = probe.GetProp("name").split()
    conf = probe.GetConformer().GetPositions()
    for atom in probe.GetAtoms():
        if probe_name[atom.GetIdx()] == "Nac":
            coords2 = conf[atom.GetIdx()]
            for neighb in atom.GetNeighbors():
                if probe.GetBondBetweenAtoms(atom.GetIdx(), neighb.GetIdx()).GetBondTypeAsDouble() == 2:
                    coords1 = conf[neighb.GetIdx()]
                    for neighb2 in atom.GetNeighbors():
                        if neighb2.GetIdx() != neighb.GetIdx() and probe_name[neighb2.GetIdx()] == "Car":
                            coords3 = conf[neighb2.GetIdx()]
                            for car_neighb in neighb2.GetNeighbors():
                                if car_neighb.GetIdx() != atom.GetIdx():
                                    coords4 = conf[car_neighb.GetIdx()]
                                    dihedral = abs(new_dihedral([coords1, coords2, coords3, coords4]))
                                    if not (dihedral < 10 or dihedral > 170):
                                        return False
                                    break
    return True


def Car_DB_Car_dihedrals_are_ok(probe):
    probe_name = probe.GetProp("name").split()
    conf = probe.GetConformer().GetPositions()
    checked_Cars = []
    for atom in probe.GetAtoms():
        if probe_name[atom.GetIdx()] == "Car" and atom.GetIdx() not in checked_Cars:
            coords_Car_1 = conf[atom.GetIdx()]
            checked_Cars.append(atom.GetIdx())
            for neighb in atom.GetNeighbors():
                if (
                    probe.GetBondBetweenAtoms(atom.GetIdx(), neighb.GetIdx()).GetBondTypeAsDouble() == 2
                    and probe_name[neighb.GetIdx()] == "Car"
                    and neighb.GetIdx() not in checked_Cars
                ):
                    coords_Car_2 = conf[neighb.GetIdx()]
                    checked_Cars.append(neighb.GetIdx())
                    for neighb1 in atom.GetNeighbors():
                        if neighb1.GetIdx() != neighb.GetIdx():
                            coords_1 = conf[neighb1.GetIdx()]
                            for neighb2 in neighb.GetNeighbors():
                                if neighb2.GetIdx() != atom.GetIdx():
                                    coords_2 = conf[neighb2.GetIdx()]
                                    dihedral = abs(new_dihedral([coords_1, coords_Car_1, coords_Car_2, coords_2]))
                                    if not (dihedral < 10 or dihedral > 170):
                                        return False
                                    break
    return True


def double_bonded_arom_rings(mol):
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol)
    name = mol.GetProp("name").split()
    rings = mol.GetRingInfo().AtomRings()
    for atom in mol.GetAtoms():
        idx1 = atom.GetIdx()
        if name[idx1] == "Car" and atom.IsInRing():
            for neighb in atom.GetNeighbors():
                idx2 = neighb.GetIdx()
                if name[idx2] == "Car" and neighb.IsInRing():
                    bond = mol.GetBondBetweenAtoms(idx1, idx2).GetBondTypeAsDouble()
                    if bond == 2:
                        same_ring = False
                        for ring in rings:
                            if idx1 in ring and idx2 in ring:
                                same_ring = True
                                break
                        if not same_ring:
                            return True
    return False


def double_bonded_Cs2_and_Car(mol):
    mol = Chem.Mol(mol)
    Chem.Kekulize(mol)
    name = mol.GetProp("name").split()
    for atom in mol.GetAtoms():
        idx1 = atom.GetIdx()
        if name[idx1] == "Car":
            for neighb in atom.GetNeighbors():
                idx2 = neighb.GetIdx()
                if name[idx2] == "Cs2":
                    bond = mol.GetBondBetweenAtoms(idx1, idx2).GetBondTypeAsDouble()
                    if bond == 2:
                        return True
                    break
    return False


def alkene_present(mol):
    name = mol.GetProp("name").split()
    for atom in mol.GetAtoms():
        idx1 = atom.GetIdx()
        if name[idx1] == "Cs2":
            for neighbor in atom.GetNeighbors():
                idx2 = neighbor.GetIdx()
                if name[idx2] == "Cs2" and mol.GetBondBetweenAtoms(idx1, idx2).GetBondTypeAsDouble() == 2:
                    return True
    return False


def atom_counters_checker(frag, name, NO_limit, elem_dct, types_dct):
    elems = [a.GetSymbol() for a in frag.GetAtoms()]
    probe_len = len(frag.GetAtoms())
    O = Counter(elems)["O"]
    N = Counter(elems)["N"]
    return not (
        NO_limit < N + O
        or any(Counter(elems)[key] > elem_dct[key] for key in elem_dct)
        or any(Counter(name.split())[key] > types_dct[key] for key in types_dct)
        or (O > 4 and O / probe_len > 0.33)
        or (N > 4 and N / probe_len > 0.33)
    )


def check_fragments(fragments: pd.DataFrame) -> tuple[list[Chem.Mol], list[int], list[int]]:
    fragments_names = fragments.index.astype(int).to_list()
    fragments_counts = fragments["count"].fillna(1).astype(int).to_list()
    mol_fragments = list(map(Chem.MolFromSmarts, fragments["smarts"]))

    return mol_fragments, fragments_names, fragments_counts


def passes_all_mcf(mol: Chem.Mol, mcf: list[Chem.Mol], mcf_counts: list[int], fragments_names: list | None = None, tolerance: int = 0, *, return_all_unpassed: bool = False):
    molecule = Chem.Mol(mol)
    Chem.SanitizeMol(molecule, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)

    result = {"passed": True, "mcf_reward": 1}
    if return_all_unpassed:
        result["all_unpassed"] = []

    # Iterate through each MCF fragment and check if it passes the filter
    for mcf_fragment_index, (mcf_fragment, count) in enumerate(zip(mcf, mcf_counts)):
        if len(molecule.GetSubstructMatches(mcf_fragment)) >= count:
            new_count = len(molecule.GetSubstructMatches(mcf_fragment)) + 1
            tolerance -= 1
            if tolerance < 0:
                if return_all_unpassed:
                    result["mcf_reward"] -= 1 / len(mcf)
                    result["all_unpassed"].append([fragments_names[mcf_fragment_index] if fragments_names else str(mcf_fragment_index), new_count])
                else:
                    result["mcf_reward"] = mcf_fragment_index / len(mcf)

                result["passed"] = False
                result["result"] = [fragments_names[mcf_fragment_index] if fragments_names else str(mcf_fragment_index), new_count]
                if not return_all_unpassed:
                    return result  # Return early if only the first unpassed fragment is needed

    if return_all_unpassed:
        result["result"] = result["all_unpassed"]

    return result


def lig_dat_extractor_reduced(lig):
    lig_idx = [atom.GetIdx() for atom in lig.GetAtoms()]
    idx_to_remove = [atom.GetIdx() for atom in lig.GetAtoms() if atom.GetSymbol() in ["H", "F"]]

    edmol = Chem.EditableMol(lig)
    for i in idx_to_remove:
        edmol.RemoveAtom(i)
    edmol = edmol.GetMol()

    bonds = defaultdict(list)
    for bond in edmol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bonds[begin].append(end)
        bonds[end].append(begin)

    pandas_sdfs_tmp = pd.DataFrame({"atm_index": lig_idx})
    pandas_sdfs_tmp["bonds"] = pandas_sdfs_tmp["atm_index"].apply(lambda i: bonds.get(i, []))

    if pandas_sdfs_tmp["bonds"].apply(len).eq(0).any():
        raise ValueError("Error finding bonds")

    return pandas_sdfs_tmp


def mol_subgraph_match(ref, lig):
    # Extract atom names
    name1 = ref.GetProp("name").split()
    name2 = lig.GetProp("name").split()

    # Extract bonds
    bonds1 = lig_dat_extractor_reduced(ref).bonds.values
    bonds2 = lig_dat_extractor_reduced(lig).bonds.values

    # Initialize graphs
    G1 = nx.Graph()
    G2 = nx.Graph()

    # Add nodes and edges for ref molecule
    for i, atom_name in enumerate(name1):
        G1.add_node(i, name=atom_name)
    for i, connected_atoms in enumerate(bonds1):
        for j in connected_atoms:
            if not G1.has_edge(i, j):
                G1.add_edge(i, j)

    # Add nodes and edges for lig molecule
    for i, atom_name in enumerate(name2):
        G2.add_node(i, name=atom_name)
    for i, connected_atoms in enumerate(bonds2):
        for j in connected_atoms:
            if not G2.has_edge(i, j):
                G2.add_edge(i, j)

    # Match subgraphs based on atom names
    GM = nx.algorithms.isomorphism.GraphMatcher(G1, G2, node_match=lambda n1, n2: n1["name"] == n2["name"])

    # Check if a subgraph match exists
    return GM.subgraph_is_isomorphic()


def probe_has_crucial_interaction(mol, crucial_coords, ligand_role):
    name = mol.GetProp("name").split()
    mol_positions = mol.GetConformer().GetPositions()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom.GetSymbol() in ["O", "N"]:
            atom_type = name[atom_idx]
            if (ligand_role == "donor" and atom_type in ["O_d", "Nd0", "Nd+"]) or (ligand_role == "acceptor" and atom_type in ["Nac", ".=O", "O_a", "O_d"]):
                rms_coord = mol_positions[atom_idx]
                dist = ((rms_coord - crucial_coords) ** 2).sum() ** 0.5
                if dist < 4:
                    return True
    return False


def data_preparation_for_dot_prediction(protein, coords, elem):
    coords = pd.DataFrame(data=np.array(coords), columns=["x_coord", "y_coord", "z_coord"])
    pandas_atoms_tmp = protein.copy()
    pandas_atoms_tmp["key"] = 1
    coords["key"] = 1
    coords["elem"] = elem
    coords["copy_index"] = coords.index.values
    merged = coords.merge(pandas_atoms_tmp, how="outer", on="key")
    merged["dists"] = np.sqrt(((merged.x_coord_x - merged.x_coord_y) ** 2) + ((merged.y_coord_x - merged.y_coord_y) ** 2) + ((merged.z_coord_x - merged.z_coord_y) ** 2))
    merged = merged[merged["dists"] <= 5]
    merged = merged.rename(columns={"x_coord": "x_coord_y", "y_coord": "y_coord_y", "z_coord": "z_coord_y"})
    coords = coords[coords.copy_index.isin(merged.copy_index.unique())]
    test_sample_graph_lig = dot_pred.pd_to_data_func_edge_att_single_prot(merged)
    return test_sample_graph_lig, coords["elem"], coords["copy_index"].values


def dot_prediction(sample_graph, elems, list_of_neurals, local_device="cpu", method="expo", dot_pred_batch_size=-1):
    expo_coefs = np.array(
        [
            0.05024999,
            0.32560112,
            0.09280444,
            0.405353,
            0.89743839,
            0.35144602,
            0.21282903,
            0.49684751,
            2.44118501,
            1.64869154,
            2.36071556,
            1.48195494,
            3.61421087,
        ]
    )

    sigmoid_coef_1 = np.array(
        [
            -0.11681227,
            0.46217051,
            0.15195799,
            0.42068638,
            0.50716267,
            0.48053579,
            0.28246032,
            0.4399476,
            0.52548684,
            0.52307923,
            0.51286999,
            0.50230432,
            0.5143657,
        ]
    )

    sigmoid_coef_2 = np.array(
        [
            7.98056929,
            10.25368463,
            9.96147297,
            9.26474631,
            10.25958158,
            10.01830618,
            9.71476016,
            9.50566272,
            9.6617758,
            10.47369277,
            9.94375251,
            9.84230245,
            10.03839996,
        ]
    )

    scores = []
    probas = []
    indices = []
    for elem in elems.unique():
        current_elem_neural = list_of_neurals[GLOBAL_TARGET_NAMES.index(elem)]
        current_indices = np.where(elems.values == elem)[0]
        current_graph = [sample_graph[x] for x in current_indices.astype(np.int16)]
        batch_size = len(current_graph) if dot_pred_batch_size == -1 else dot_pred_batch_size
        sample_loader_lig = DataLoader(current_graph, batch_size=batch_size, shuffle=False)
        proba = dot_pred.dot_prediction_solo_by_elem(local_device, sample_loader_lig, current_elem_neural)
        if method == "expo":
            score = proba ** expo_coefs[GLOBAL_TARGET_NAMES.index(elem)]
        elif method == "sigmoid":
            score = (proba - sigmoid_coef_1[GLOBAL_TARGET_NAMES.index(elem)]) * sigmoid_coef_2[GLOBAL_TARGET_NAMES.index(elem)]
        del sample_loader_lig
        scores.extend(score)
        probas.extend(proba)
        indices.extend(current_indices)
    sorted_indices = np.argsort(indices)
    return np.array(scores)[sorted_indices], np.array(probas)[sorted_indices]


def dot_prediction_pharma(sample_graph, elems, list_of_neurals, local_device="cpu", dot_pred_batch_size=-1):
    probas = []
    indices = []
    for elem in elems.unique():
        current_elem_neural = list_of_neurals[GLOBAL_TARGET_NAMES.index(elem)]
        current_indices = np.where(elems.values == elem)[0]
        current_graph = [sample_graph[x] for x in current_indices.astype(np.int16)]
        batch_size = len(current_graph) if dot_pred_batch_size == -1 else dot_pred_batch_size
        sample_loader_lig = DataLoader(current_graph, batch_size=batch_size, shuffle=False)
        proba = dot_pred.dot_prediction_solo_by_elem(local_device, sample_loader_lig, current_elem_neural)
        probas.extend(proba)
        indices.extend(current_indices)
    sorted_indices = np.argsort(indices)
    return np.array(probas)[sorted_indices]


def get_arom_Car_with_no_double_bonds(frag):
    frag = Chem.Mol(frag)
    Chem.Kekulize(frag)
    probe_name = frag.GetProp("name").split()
    return [
        atom.GetIdx()
        for atom in frag.GetAtoms()
        if probe_name[atom.GetIdx()] == "Car" and len(atom.GetNeighbors()) == 2 and not any(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds())
    ]


def ionize(mol2):
    mol_name = mol2.GetProp("name").split()
    mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol2).replace("0  0  0  0  0  1  0  0", "0  0  0  0  0  0  0  0"))
    for atom in mol.GetAtoms():
        if mol_name[atom.GetIdx()] == "Nd+":
            atom.SetFormalCharge(1)
            atom.UpdatePropertyCache()
        elif mol_name[atom.GetIdx()] == ".=O" and mol.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()).GetBondType() == Chem.rdchem.BondType.SINGLE:
            atom.SetFormalCharge(-1)
            atom.UpdatePropertyCache()
    for prop_name in mol2.GetPropNames():
        mol.SetProp(prop_name, mol2.GetProp(prop_name))
    return mol


def get_uncycled_atoms(probe):
    probe_name = probe.GetProp("name").split()
    return [a.GetIdx() for a in probe.GetAtoms() if probe_name[a.GetIdx()] in ring_types and not a.IsInRing() and len(a.GetNeighbors()) == 1]

def bisect_angles_are_ok(frag, ring_atom, new_atom_index):
    conf = frag.GetConformer().GetPositions()
    p2 = conf[ring_atom.GetIdx()]
    neighbors = [a.GetIdx() for a in ring_atom.GetNeighbors()]
    pairs = [pair for pair in combinations(neighbors, 2) if new_atom_index in pair]
    angles = [calc_angle(conf[pair[0]], p2, conf[pair[1]]) for pair in pairs]
    return abs(angles[0] - angles[1]) < 15


def task_initializer(mode: "GenerationMode", probe, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring):
    probe_name = probe.GetProp("name").split()
    init_probe = probe
    probe = Chem.Mol(probe)
    positions = probe.GetConformer().GetPositions()
    Chem.Kekulize(probe)
    tasks_list = [
        "close_chiral_center",
        "build_to_Csp",
        "build_to_terminal_Csp",
        "close_aromatic_ring",
        "fix_Car",
        "close_aliphatic_ring",
        "build_orthosubstituent_to_rotated_amide",
        # "initiate_ring",
        "build_DB_to_Cs2",
        "build_SB_to_Cs2",
        "build_to_Sul",
        "build_sp2_to_Nac",
        "build_sp2_to_Nd0",
        "build_.=O_to_SO2",
        "build_to_SO2",
        "build_to_O_a",
        "build_to_Nac",
        "build_to_Nd0",
        "close_guanidine_nitrogen",
        "no_task",
    ]
    tasks = set()
    tasks.add("no_task")
    for atom in probe.GetAtoms():
        if atom.GetIdx() in forbidden_atoms:
            continue
        atom_type = probe_name[atom.GetIdx()]
        # check amide dihedral and try to prove it if it is not in the same plane as the nearest cycle
        C_amide = next(
        (a for a in atom.GetNeighbors() if probe_name[a.GetIdx()] == "Cs2"),
        None
        )
        if atom_type == 'Nd0' and C_amide and next((a for a in C_amide.GetNeighbors() if probe_name[a.GetIdx()] == ".=O"),None):
            dihedral = None
            indexes_to_build_to = None
            C_amide_coords = positions[C_amide.GetIdx()]
            N_amide_coords = positions[atom.GetIdx()]
            n_aromatic_neighbors = [a for a in atom.GetNeighbors() if a.GetIsAromatic() and a.GetIdx() != C_amide.GetIdx()]
            c_aromatic_neighbors = [a for a in C_amide.GetNeighbors() if a.GetIsAromatic() and a.GetIdx() != atom.GetIdx()]
            # if/if system here is used because i try to solve an example, where there is a cycle on each end of the amide
            # and we need to explain each of their geometries
            if n_aromatic_neighbors:
                ring_atom = n_aromatic_neighbors[0]
                ring_atom_neighbors = [a for a in ring_atom.GetNeighbors() if a.GetIsAromatic() and a.GetIdx() != atom.GetIdx() and len(a.GetNeighbors()) < 3]
                if len(ring_atom_neighbors) == 2:
                    coords1 = positions[ring_atom_neighbors[0].GetIdx()]
                    coords2 = positions[ring_atom.GetIdx()]
                    coords3 = N_amide_coords
                    coords4 = C_amide_coords
                    dihedral = abs(new_dihedral([coords1, coords2, coords3, coords4]))
                if dihedral and dihedral > 45 and dihedral < 135 and all([len(b.GetNeighbors()) == 2 for b in ring_atom_neighbors]):
                    indexes_to_build_to = [str(a.GetIdx()) for a in ring_atom_neighbors] # check for forbidden idxs and probe skipping will be performed after task assignment
                    init_probe.SetProp('build_orthosubstituent_to_rotated_amide', ' '.join(indexes_to_build_to))
                    tasks.add("build_orthosubstituent_to_rotated_amide")
            if c_aromatic_neighbors and not indexes_to_build_to:
                ring_atom = c_aromatic_neighbors[0]
                ring_atom_neighbors = [a for a in ring_atom.GetNeighbors() if a.GetIsAromatic() and a.GetIdx() != C_amide.GetIdx() and len(a.GetNeighbors()) < 3]
                if len(ring_atom_neighbors) == 2:
                    coords1 = positions[ring_atom_neighbors[0].GetIdx()]
                    coords2 = positions[ring_atom.GetIdx()]
                    coords3 = C_amide_coords
                    coords4 = N_amide_coords
                    dihedral = abs(new_dihedral([coords1, coords2, coords3, coords4]))
                if dihedral and dihedral > 45 and dihedral < 135 and all([len(b.GetNeighbors()) == 2 for b in ring_atom_neighbors]):
                    indexes_to_build_to = [str(a.GetIdx()) for a in ring_atom_neighbors] # check for forbidden idxs and probe skipping will be performed after task assignment
                    init_probe.SetProp('build_orthosubstituent_to_rotated_amide', ' '.join(indexes_to_build_to))
                    tasks.add("build_orthosubstituent_to_rotated_amide")
        # build to Csp
        if atom_type == "Csp":
            if not any(bond.GetBondType() is Chem.BondType.TRIPLE for bond in atom.GetBonds()):
                tasks.add("build_to_Csp")
            elif len(atom.GetNeighbors()) == 1:
                tasks.add("build_to_terminal_Csp")
        # build to so2
        if atom_type == "SO2":
            if len([n for n in atom.GetNeighbors() if probe_name[n.GetIdx()] == ".=O"]) <= 1:
                tasks.add("build_.=O_to_SO2")
            elif len(atom.GetNeighbors()) < 4:
                tasks.add("build_to_SO2")
        # build to cs2
        if atom_type in ["Cs2", "C2r"] and not any(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()):
            tasks.add("build_DB_to_Cs2")
        if atom_type in ["Cs2", "C2r"] and len(atom.GetNeighbors()) < 3:
            tasks.add("build_SB_to_Cs2")
        # arom rings
        if not atom.IsInRing():
            if not Nac_outside_arom_ring and atom_type in ["Nac"] and any(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()):
                tasks.add("close_aromatic_ring")
            if atom_type in ["Car"]:
                tasks.add("close_aromatic_ring")

            elif atom_type == "Nac" and len(atom.GetNeighbors()) == 1:
                if probe_name[atom.GetNeighbors()[0].GetIdx()] == "Nd0":
                    tasks.add("close_aromatic_ring")
                else:
                    neighb = atom.GetNeighbors()[0]
                    bond_type = str(probe.GetBondBetweenAtoms(atom.GetIdx(), neighb.GetIdx()).GetBondType())
                    if bond_type in ["DOUBLE"]:
                        tasks.add("close_aromatic_ring")
            # another check, that will force to close any ring with ring_type-///-ring_type angle < 110 and when two of the atoms are not in ring
            if atom_type in ring_types and len(atom.GetNeighbors()) == 1 and not atom.GetNeighbors()[0].IsInRing():
                ring_neigbors_of_atom = [a for a in atom.GetNeighbors()[0].GetNeighbors() if probe_name[a.GetIdx()] in ring_types and a.IsInRing()]
                if ring_neigbors_of_atom:
                    coord1 = positions[atom.GetIdx()]
                    coord2 = positions[atom.GetNeighbors()[0].GetIdx()]
                    coord3 = positions[ring_neigbors_of_atom[0].GetIdx()]
                    angle = calc_angle(coord1,coord2,coord3)
                    if angle < 110:
                        init_probe.SetProp('initiate_ring_110', '')
            ring_neighbours = [atom for atom in atom.GetNeighbors() if atom.IsInRing()]
            new_atom_index = atom.GetIdx()
            if ring_neighbours and atom_type in ring_types and not bisect_angles_are_ok(probe, ring_neighbours[0], new_atom_index) and len(atom.GetNeighbors()) == 1:
                init_probe.SetProp('initiate_ring_bisect', '')

        # build to Sul
        if atom_type == "Sul" and len(atom.GetNeighbors()) == 1:
            tasks.add("build_to_Sul")
        # aliph rings
        if atom_type in ["C3r", "C2r"] and not atom.IsInRing():
            tasks.add("close_aliphatic_ring")
        # build to O_a
        if atom_type == "O_a" and len(atom.GetNeighbors()) < 2:
            tasks.add("build_to_O_a")
        # build to Nac
        if atom_type == "Nac" and sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) < 3:
            tasks.add("build_to_Nac")
        # build sp2 to Nd0
        if atom_type == "Nac" and not any(probe_name[neighb.GetIdx()] in ["Car", "Cs2", "SO2", "C2r"] for neighb in atom.GetNeighbors()):
            tasks.add("build_sp2_to_Nac")
        # build sp2 to Nd0
        if atom_type == "Nd0" and not any(probe_name[neighb.GetIdx()] in ["Car", "Cs2", "SO2", "C2r"] for neighb in atom.GetNeighbors()):
            tasks.add("build_sp2_to_Nd0")
        # build to terminal Nd0
        if mode is GenerationMode.SCAFFOLD and atom_type == "Nd0" and len(atom.GetNeighbors()) < 2:
            tasks.add("build_to_Nd0")
        # fix Car with no double bonds
        if atom_type == "Car" and atom.IsInRing() and not any(b.GetBondTypeAsDouble() == 2.0 for b in atom.GetBonds()):
            tasks.add("fix_Car")
        # close guanidine nitrogen
        if atom_type == "Nd+" and len(atom.GetNeighbors()) == 1 and probe_name[atom.GetNeighbors()[0].GetIdx()] == "Car":
            tasks.add("close_guanidine_nitrogen")

    # excessive chirality
    if count_chiral_centers(probe) > max_num_chiral_centers:
        tasks.add("close_chiral_center")

    for task in tasks_list:
        if task in tasks:
            return task
    return None


def map_atoms_to_build_to(probe):
    probe_name = probe.GetProp("name").split()
    return [
        atom.GetIdx()
        for atom in probe.GetAtoms()
        if (probe_name[atom.GetIdx()] in ["Nd+", "Cs3", "C3r"] and len(atom.GetNeighbors()) < 4)
        or (probe_name[atom.GetIdx()] in ["Car", "Cs2", "C2r"] and len(atom.GetNeighbors()) < 3)
        or (probe_name[atom.GetIdx()] in ["Nd0"] and sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) < 2)
        or (probe_name[atom.GetIdx()] in ["Nac"] and sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) < 3)
        or (probe_name[atom.GetIdx()] in ["Csp", "O_a", "Sul"] and len(atom.GetNeighbors()) < 2)
        or (probe_name[atom.GetIdx()] in ["SO2"] and len(atom.GetNeighbors()) < 4)
    ]


def scen_mapping(
    current_task,
    probe,
    probe_name,
    probe_len,
    tasks,
    iter_n,
    iterations_number,
    allow_Cs3_in_arom_rings,
    allow_linker_Sul,
):
    atoms = map_atoms_to_build_to(probe)
    scen1_allowed = False
    scen2_allowed = False
    scen1_ids_allowed = []
    scen1_fragments_allowed = []
    scen2_ids_allowed = []
    scen2_fragments_allowed = []

    if current_task == "close_guanidine_nitrogen":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Nd+" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Cs2"]

    elif current_task == "build_to_Nd0":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Nd0" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Cs2", "Car", "C3r", "C2r"]

    elif current_task == "build_to_Csp":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Csp" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Csp", "Nac"]

    elif current_task == "build_to_terminal_Csp":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Csp" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Car", "Cs3", "C3r", "Cs2", "C2r"]

    elif current_task == "build_.=O_to_SO2":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = [
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if probe_name[atom.GetIdx()] == "SO2" and len([n for n in atom.GetNeighbors() if probe_name[n.GetIdx()] == ".=O"]) <= 1 and atom.GetIdx() in atoms
        ]
        scen1_fragments_allowed = [".=O"]
        scen2_ids_allowed = [
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if probe_name[atom.GetIdx()] == "SO2" and len([n for n in atom.GetNeighbors() if probe_name[n.GetIdx()] == ".=O"]) <= 1 and atom.GetIdx() in atoms
        ]
        scen2_fragments_allowed = [".=O"]

    elif current_task == "build_to_SO2":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "SO2" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Nd0", "Nac", "Cs3", "C3r", "Car", "Cs2", "C2r"]
        scen2_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "SO2" and atom.GetIdx() in atoms]
        scen2_fragments_allowed = ["Nd0", "Nac", "Cs3", "C3r", "Car", "Cs2", "C2r"]

    elif current_task == "build_DB_to_Cs2":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] in ["Cs2", "C2r"] and atom.GetIdx() in atoms]
        scen1_fragments_allowed = [".=O", "Car", "Cs2", "C2r"]
        scen2_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] in ["Cs2", "C2r"] and atom.GetIdx() in atoms]
        scen2_fragments_allowed = [".=O", "Car", "Cs2", "C2r"]

    elif current_task == "build_SB_to_Cs2":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] in ["Cs2", "C2r"] and atom.GetIdx() in atoms]
        scen1_fragments_allowed = [".=O", "Nd0", "Nac", "Cs3", "C3r", "Car", "Cs2", "C2r"]
        scen2_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] in ["Cs2", "C2r"] and atom.GetIdx() in atoms]
        scen2_fragments_allowed = [".=O", "Nd0", "Nac", "Cs3", "C3r", "Car", "Cs2", "C2r"]

    elif current_task == "build_orthosubstituent_to_rotated_amide":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if atom.GetIdx() in atoms]
        scen1_fragments_allowed = GLOBAL_TARGET_NAMES_RING
        scen2_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if atom.GetIdx() in atoms]
        scen2_fragments_allowed = GLOBAL_TARGET_NAMES_RING

    elif current_task == "close_aromatic_ring":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = [
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if probe_name[atom.GetIdx()] in ["Car", "Nac", "Nd0", "Sul", "O_a"]
            and len([n for n in atom.GetNeighbors() if probe_name[n.GetIdx()] in ["Car", "Nac", "Nd0", "Sul", "O_a"] and not n.IsInRing()]) == 1
            and not atom.IsInRing()
            and atom.GetIdx() in atoms
        ]

        scen1_fragments_allowed = ["Car", "O_a", "Nac", "Nd0", "Sul"]
        scen2_ids_allowed = [
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if probe_name[atom.GetIdx()] in ["Car", "Nac", "Nd0", "Sul", "O_a"]
            and len([n for n in atom.GetNeighbors() if probe_name[n.GetIdx()] in ["Car", "Nac", "Nd0", "Sul", "O_a"] and not n.IsInRing()]) == 1
            and not atom.IsInRing()
            and atom.GetIdx() in atoms
        ]

        scen2_fragments_allowed = ["Car", "O_a", "Nac", "Nd0", "Sul"]
        if allow_Cs3_in_arom_rings:
            scen1_fragments_allowed.append("C3r")
            scen2_fragments_allowed.append("C3r")
        if tasks.split()[-1] != "close_aromatic_ring":
            scen1_ids_allowed.append(probe_len - 1)
            scen2_ids_allowed.append(probe_len - 1)

    elif current_task == "fix_Car":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = get_arom_Car_with_no_double_bonds(probe)
        scen1_fragments_allowed = [".=O", "Car", "Cs2", "C2r", "Nac"]

        scen2_ids_allowed = scen1_ids_allowed
        scen2_fragments_allowed = [".=O", "Car", "Cs2", "C2r", "Nac"]

    elif current_task == "build_to_Sul":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Sul" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Car", "C3r", "C2r"]

    elif current_task == "close_aliphatic_ring":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if probe_name[atom.GetIdx()] in ["C3r", "Nd+", "Nd0", "Nac", "O_a", "C2r", "SO2"]
            and len([n for n in atom.GetNeighbors() if probe_name[n.GetIdx()] in ["C3r", "Nd+", "Nd0", "Nac", "O_a", "C2r", "SO2"] and not n.IsInRing()]) == 1
            and atom.GetIdx() in atoms
        ]
        scen1_fragments_allowed = ["C3r", "Nd+", "Nac", "Nd0", "O_a", "C2r", "SO2"]
        if allow_linker_Sul:
            scen1_fragments_allowed.append("Sul")

    elif current_task == "build_to_O_a":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "O_a" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Nac", "Cs3", "C3r", "Car", "Cs2", "C2r"]

    elif current_task == "build_to_Nac":
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Nac" and atom.GetIdx() in atoms]
        scen1_fragments_allowed = ["Car", "Cs3", "C3r", "Cs2", "C2r", "O_a", "Nac", "Nd0", "SO2"]
        scen2_ids_allowed = [atom.GetIdx() for atom in probe.GetAtoms() if probe_name[atom.GetIdx()] == "Nac" and atom.GetIdx() in atoms]
        scen2_fragments_allowed = ["Car", "Cs3", "C3r", "Cs2", "C2r", "O_a", "Nac", "Nd0", "SO2"]
        if allow_linker_Sul:
            scen1_fragments_allowed.append("Sul")
            scen2_fragments_allowed.append("Sul")

    elif current_task == "build_sp2_to_Nac":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if probe_name[atom.GetIdx()] == "Nac" and not any(probe_name[neighb.GetIdx()] in ["Car", "Cs2", "SO2"] for neighb in atom.GetNeighbors()) and atom.GetIdx() in atoms
        ]
        scen1_fragments_allowed = ["Car", "Cs2", "SO2", "C2r"]

    elif current_task == "build_sp2_to_Nd0":
        scen1_allowed = True
        scen2_allowed = False
        scen1_ids_allowed = [
            atom.GetIdx()
            for atom in probe.GetAtoms()
            if probe_name[atom.GetIdx()] == "Nd0" and not any(probe_name[neighb.GetIdx()] in ["Car", "Cs2", "SO2"] for neighb in atom.GetNeighbors()) and atom.GetIdx() in atoms
        ]
        scen1_fragments_allowed = ["Car", "Cs2", "SO2", "C2r"]

    else:
        scen1_allowed = True
        scen2_allowed = True
        scen1_ids_allowed = atoms
        scen1_fragments_allowed = GLOBAL_TARGET_NAMES_RING.copy()
        scen2_ids_allowed = atoms
        scen2_fragments_allowed = GLOBAL_TARGET_NAMES_RING.copy()
        scen1_fragments_allowed.remove(".=O")
        scen2_fragments_allowed.remove(".=O")
        if not allow_linker_Sul:
            scen1_fragments_allowed.remove("Sul")
            scen2_fragments_allowed.remove("Sul")
        # avoid aromatic cycles when less than 5 iterations left
        if iter_n > iterations_number - 5:
            scen1_fragments_allowed.remove("Car")
            scen2_fragments_allowed.remove("Car")
        # avoid SO2 when less than 4 iterations left
        if iter_n > iterations_number - 4:
            scen1_fragments_allowed.remove("SO2")
            scen2_fragments_allowed.remove("SO2")
        # avoid any cycles when less than 3 iterations left
        if iter_n > iterations_number - 3:
            scen1_fragments_allowed.remove("C2r")
            scen1_fragments_allowed.remove("C3r")
            scen2_fragments_allowed.remove("C2r")
            scen2_fragments_allowed.remove("C3r")

    return (
        scen1_allowed,
        scen2_allowed,
        scen1_ids_allowed,
        scen1_fragments_allowed,
        scen2_ids_allowed,
        scen2_fragments_allowed,
    )


def parse_config(path_to_working_dir):
    config = configparser.ConfigParser()
    config.read(path_to_working_dir + "/config.ini")
    args = {key: config["DEFAULT"][key] for key in config["DEFAULT"]}
    args["path_to_working_dir"] = path_to_working_dir
    args["limits_dict"] = {line.split(":")[0]: int(line.split(":")[1]) for line in args["limits_dict"].split(",")}
    args["elem_dict"] = {line.split(":")[0]: int(line.split(":")[1]) for line in args["elem_dict"].split(",")}
    args["no_limit"] = int(args["no_limit"])
    args["max_num_low_cycles"] = int(args["max_num_low_cycles"])
    args["important_amino_acid_number"] = int(args["important_amino_acid_number"])
    args["skip_minimization"] = args["skip_minimization"].lower() == "true"
    args["max_num_charged_atoms"] = int(args["max_num_charged_atoms"])
    args["max_num_chiral_centers"] = int(args["max_num_chiral_centers"])
    args["min_mol_size"] = int(args["min_mol_size"])
    allowed_atoms_str = args["allowed_atoms_idxs"][1:-1]
    args["allowed_atoms_idxs"] = [] if not allowed_atoms_str else [int(i) for i in allowed_atoms_str.split(",")]
    args["max_acyclic_chain_length"] = int(args["max_acyclic_chain_length"])
    args["max_clashes"] = int(args["max_clashes"])
    args["number_of_poses_on_iter"] = int(args["number_of_poses_on_iter"])
    args["num_heavy_atoms_to_add"] = int(args["num_heavy_atoms_to_add"])
    args["max_execution_time"] = int(args["max_execution_time"])
    # ------------------------------------------------------------
    # Time management settings
    # ------------------------------------------------------------
    args["time_management_mode"] = str(args.get("time_management_mode", "old")).lower()

    if args["time_management_mode"] not in {"old", "dynamic", "dynamic_snapshot"}:
        raise ValueError(
            "Unknown time_management_mode. "
            "Use 'old','dynamic' or 'dynamic_snapshot'."
        )

    args["branch_time_factor"] = float(args.get("branch_time_factor", 1.0))
    args["min_branch_seconds"] = int(args.get("min_branch_seconds", 300))
    args["max_branch_seconds"] = int(args.get("max_branch_seconds", 3600))
    args["dynamic_split_until_iter"] = int(args.get("dynamic_split_until_iter", 3))

    if args["dynamic_split_until_iter"] < 1:
        raise ValueError("dynamic_split_until_iter must be >= 1")
    # ------------------------------------------------------------
    # Time management settings
    # ------------------------------------------------------------
    args["ref_lib_to_pass"] = list(Chem.SDMolSupplier(f"{path_to_working_dir}/{args['ref_lib_to_pass']}")) if args["ref_lib_to_pass"] else []
    args["max_num_positive_charges"] = int(args["max_num_positive_charges"])
    args["max_num_negative_charges"] = int(args["max_num_negative_charges"])
    args["max_num_charged_atoms"] = int(args["max_num_charged_atoms"])
    return args


def strange_cycle_present(mol):
    for ring in mol.GetRingInfo().AtomRings():
        sp2_counter = sum([str(mol.GetAtomWithIdx(ID).GetHybridization()) == "SP2" for ID in ring])
        sp3_counter = sum([str(mol.GetAtomWithIdx(ID).GetHybridization()) == "SP3" for ID in ring])
        if sp2_counter == len(ring) - 1 and sp3_counter == 1:
            return True
    return False


def linker_Sul_present(mol):
    name = mol.GetProp("name").split()
    return any(name[atom.GetIdx()] == "Sul" and not atom.IsInRing() for atom in mol.GetAtoms())


def non_ring_Nac_with_DB_present(mol):
    name = mol.GetProp("name").split()
    return any(name[atom.GetIdx()] == "Nac" and any(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()) and not atom.IsInRing() for atom in mol.GetAtoms())


def lib_to_pass_checker(ref_lib_to_pass, args, mcf_df, final_mcf_df):
    if len(mcf_df):
        mcf_fragments, fragments_names, fragments_counts = check_fragments(mcf_df)

    disabled_mcf_ids = []
    final_disabled_mcf_ids = []
    changed_parameters = {}

    if len(final_mcf_df):
        final_mcf_fragments, final_fragments_names, final_fragments_counts = check_fragments(final_mcf_df)

    for mol in ref_lib_to_pass:
        name = [define_atom_type(a) for a in Chem.AddHs(mol, addCoords=True).GetAtoms() if a.GetSymbol() != "H"]
        mol.SetProp("name", " ".join(name))
        sum([len(r) in [3, 4] for r in mol.GetRingInfo().AtomRings()])

        if not args["allow_cs3_in_arom_rings"] and strange_cycle_present(mol):
            args["allow_cs3_in_arom_rings"] = True
            changed_parameters["allow_cs3_in_arom_rings"] = args["allow_cs3_in_arom_rings"]

        if args["allow_chair_only"] and not aliphatic_rings_are_ok(mol):
            args["allow_chair_only"] = False
            changed_parameters["allow_chair_only"] = args["allow_chair_only"]

        if not args["allow_axial_groups"] and axial_groups_present(mol):
            args["allow_axial_groups"] = True
            changed_parameters["allow_axial_groups"] = args["allow_axial_groups"]

        if not args["allow_condensed_alicyclics"] and condensed_aliphatic_rings_present(mol):
            args["allow_condensed_alicyclics"] = True
            changed_parameters["allow_condensed_alicyclics"] = args["allow_condensed_alicyclics"]

        if not args["allow_paired_nd0"] and paired_Nd0_present(mol):
            args["allow_paired_nd0"] = True
            changed_parameters["allow_paired_nd0"] = args["allow_paired_nd0"]

        if not args["allow_nd0_nac_pair"] and pair_Nd0_Nac_present(mol):
            args["allow_nd0_nac_pair"] = True
            changed_parameters["allow_nd0_nac_pair"] = args["allow_nd0_nac_pair"]

        if not args["allow_double_bonded_arom_rings"] and double_bonded_arom_rings(mol):
            args["allow_double_bonded_arom_rings"] = True
            changed_parameters["allow_double_bonded_arom_rings"] = args["allow_double_bonded_arom_rings"]

        if not args["allow_alkenes"] and alkene_present(mol):
            args["allow_alkenes"] = True
            changed_parameters["allow_alkenes"] = args["allow_alkenes"]

        if not args["allow_linker_sul"] and linker_Sul_present(mol):
            args["allow_linker_sul"] = True
            changed_parameters["allow_linker_sul"] = args["allow_linker_sul"]

        if not args["nac_outside_arom_ring"] and non_ring_Nac_with_DB_present(mol):
            args["nac_outside_arom_ring"] = True
            changed_parameters["nac_outside_arom_ring"] = args["nac_outside_arom_ring"]

        if len(mcf_df):
            result = passes_all_mcf(
                mol,
                fragments_names=fragments_names,
                mcf=mcf_fragments,
                mcf_counts=fragments_counts,
                tolerance=0,
                return_all_unpassed=True,
            )
            if not result["passed"]:
                for count_change in result["all_unpassed"]:
                    mcf_df.loc[mcf_df.index == count_change[0], "count"] = count_change[1]
                # mcf_df = mcf_df.drop(result["result"])
                disabled_mcf_ids.extend(result["result"])
                mcf_fragments, fragments_names, fragments_counts = check_fragments(mcf_df)

        if len(final_mcf_df):
            result = passes_all_mcf(
                Chem.AddHs(mol, addCoords=True),
                fragments_names=final_fragments_names,
                mcf=final_mcf_fragments,
                mcf_counts=final_fragments_counts,
                tolerance=0,
                return_all_unpassed=True,
            )
            if not result["passed"]:
                for count_change in result["all_unpassed"]:
                    final_mcf_df.loc[final_mcf_df.index == count_change[0], "count"] = count_change[1]
                # final_mcf_df = final_mcf_df.drop(result["result"])
                final_disabled_mcf_ids.extend(result["result"])  # было append
                final_mcf_fragments, final_fragments_names, final_fragments_counts = check_fragments(final_mcf_df)

    return args, mcf_df, final_mcf_df, list({i[0] for i in disabled_mcf_ids}), list({i[0] for i in final_disabled_mcf_ids}), changed_parameters


def out_duplicates_checker(probe, compare, max_poses_of_one_structure: int | Literal["all"]):
    duplicates_check_results = duplicates_check(probe, compare)
    duplicates_check_results[0]
    biggest_local_difference = duplicates_check_results[1]
    final_out = []

    if max_poses_of_one_structure == "all":
        final_out.extend(mol for mol in compare if mol)
        if biggest_local_difference > 1.5:
            final_out.append(probe)
        return sorted(final_out, key=rdMolDescriptors.CalcExactMolWt)

    duplicates_list, others_list = [], []
    for mol in compare:
        if duplicates_check(probe, [mol]) != (100, 100):
            duplicates_list.append(mol)
        else:
            others_list.append(mol)
    if biggest_local_difference > 1.5:
        duplicates_list.append(probe)
    duplicates_list.sort(key=lambda x: float(x.GetProp("NBS")), reverse=True)
    if len(duplicates_list) > max_poses_of_one_structure:
        duplicates_list.pop(-1)
    duplicates_list.extend(others_list)
    final_out.extend(mol for mol in duplicates_list if mol)
    return sorted(final_out, key=rdMolDescriptors.CalcExactMolWt)


def save_mols_for_super_output(path, res_mol):
    with open(str(path / "super_final_out.sdf"), "a") as f:
        writer = Chem.SDWriter(f)
        writer.write(res_mol)
        writer.flush()


# ------------------------------------------------------------
# Global selected deduplication helpers
# ------------------------------------------------------------
def mol_type_counter(mol):
    """Counter of generator atom types stored in mol prop 'name'."""
    if not mol.HasProp("name"):
        return Counter()
    return Counter(mol.GetProp("name").split())


def mol_type_counter_key(mol):
    return tuple(sorted(mol_type_counter(mol).items()))


def safe_canonical_smiles(mol):
    """Canonical SMILES for topology-level grouping before costly RMSD checks."""
    try:
        mol_copy = Chem.Mol(mol)
        return Chem.MolToSmiles(mol_copy, canonical=True, isomericSmiles=False)
    except Exception:
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            return Chem.MolToSmiles(mol_copy, canonical=True, isomericSmiles=False)
        except Exception:
            return None


def build_global_selected_index(all_selected_path):
    """
    Read previous all_selected.sdf and group molecules by:
    1) canonical SMILES;
    2) Counter(atom types from prop 'name').
    """
    index = defaultdict(list)
    if not all_selected_path.exists() or all_selected_path.stat().st_size == 0:
        return index

    for mol_idx, mol in enumerate(Chem.SDMolSupplier(str(all_selected_path), removeHs=False)):
        if mol is None or not mol.HasProp("name"):
            continue
        smiles = safe_canonical_smiles(mol)
        if smiles is None:
            continue
        mol.SetProp("global_selected_index", str(mol_idx))
        index[(smiles, mol_type_counter_key(mol))].append(mol)
    return index


def rmsd_no_alignment_for_mapping(query_mol, target_mol, query_to_target):
    """
    RMSD without alignment/re-centering for a concrete atom mapping.

    query_to_target[i] is the target atom index corresponding to query atom i.
    """
    q_pos = query_mol.GetConformer().GetPositions()
    t_pos = target_mol.GetConformer().GetPositions()
    sq = []
    for q_idx, t_idx in enumerate(query_to_target):
        diff = q_pos[q_idx] - t_pos[t_idx]
        sq.append(float(np.dot(diff, diff)))
    return float(np.sqrt(np.mean(sq)))


def get_type_matched_isomorphism_mappings(query_mol, target_mol):
    """
    Return mappings query_idx -> target_idx for isomorphic molecules,
    additionally requiring equality of generator atom types from prop 'name'.
    """
    if query_mol.GetNumAtoms() != target_mol.GetNumAtoms():
        return []
    if not query_mol.HasProp("name") or not target_mol.HasProp("name"):
        return []

    query_types = query_mol.GetProp("name").split()
    target_types = target_mol.GetProp("name").split()
    if len(query_types) != query_mol.GetNumAtoms() or len(target_types) != target_mol.GetNumAtoms():
        return []

    try:
        raw_mappings = target_mol.GetSubstructMatches(query_mol, uniquify=False)
    except Exception:
        raw_mappings = []

    matched = []
    for mapping in raw_mappings:
        if len(mapping) != query_mol.GetNumAtoms():
            continue
        if all(query_types[q_idx] == target_types[t_idx] for q_idx, t_idx in enumerate(mapping)):
            matched.append(tuple(mapping))
    return matched


def best_type_matched_rmsd_no_alignment(query_mol, target_mol):
    """
    Best no-alignment RMSD over all atom mappings with matching generator types.
    Returns (best_rmsd, best_mapping). Mapping is query_idx -> target_idx.
    """
    best_rmsd = None
    best_mapping = None
    for mapping in get_type_matched_isomorphism_mappings(query_mol, target_mol):
        rmsd = rmsd_no_alignment_for_mapping(query_mol, target_mol, mapping)
        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd
            best_mapping = mapping
    return best_rmsd, best_mapping




ROTATABLE_BOND_TORSIONS_PROP = "rotatable_bond_torsions"
PROTECTED_RING_ATTACHMENTS_PROP = "protected_ring_attachments"
ROTATABLE_TORSION_THRESHOLD = 30.0
LAST_GROWTH_TORSION_THRESHOLD = 30.0
LAST_GROWTH_BOND_ANGLE_THRESHOLD = 5.0
RING_ATTACHMENT_TORSION_THRESHOLD = 30.0
RING_ATTACHMENT_PLANE_ANGLE_THRESHOLD = 5.0


def torsion_indices_are_valid(mol, torsion):
    """Return True if torsion indices exist and describe a bonded a-b-c-d path."""
    if torsion is None or len(torsion) != 4:
        return False
    if len(set(torsion)) != 4:
        return False
    if any(i < 0 or i >= mol.GetNumAtoms() for i in torsion):
        return False
    a_idx, b_idx, c_idx, d_idx = torsion
    return (
        mol.GetBondBetweenAtoms(a_idx, b_idx) is not None
        and mol.GetBondBetweenAtoms(b_idx, c_idx) is not None
        and mol.GetBondBetweenAtoms(c_idx, d_idx) is not None
    )


def angle_indices_are_valid(mol, angle_indices):
    """Return True if b-c-d is a valid bonded angle."""
    if angle_indices is None or len(angle_indices) != 3:
        return False
    if len(set(angle_indices)) != 3:
        return False
    if any(i < 0 or i >= mol.GetNumAtoms() for i in angle_indices):
        return False
    b_idx, c_idx, d_idx = angle_indices
    return (
        mol.GetBondBetweenAtoms(b_idx, c_idx) is not None
        and mol.GetBondBetweenAtoms(c_idx, d_idx) is not None
    )


def heavy_neighbor_indices(atom, exclude_idx):
    """Heavy-atom neighbours of atom, excluding one atom index."""
    return sorted(
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetIdx() != exclude_idx and n.GetSymbol() != "H"
    )


def choose_torsion_side_neighbor(mol, center_idx, exclude_idx):
    """Pick a deterministic heavy neighbour used to define a torsion side."""
    center_atom = mol.GetAtomWithIdx(center_idx)
    candidates = heavy_neighbor_indices(center_atom, exclude_idx)
    if not candidates:
        return None

    names = mol.GetProp("name").split() if mol.HasProp("name") else []

    def candidate_key(idx):
        atom = mol.GetAtomWithIdx(idx)
        atom_type = names[idx] if idx < len(names) else define_atom_type(atom)
        non_terminal_penalty = 0 if heavy_neighbor_indices(atom, center_idx) else 1
        carbonyl_penalty = 1 if atom_type == ".=O" else 0
        return (non_terminal_penalty, carbonyl_penalty, idx)

    return sorted(candidates, key=candidate_key)[0]


def bond_is_rotatable_for_duplicate_filter(mol, bond):
    """Broad generator-level definition of a currently rotatable bond."""
    if bond.IsInRing():
        return False
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False

    b_idx = bond.GetBeginAtomIdx()
    c_idx = bond.GetEndAtomIdx()
    b_atom = mol.GetAtomWithIdx(b_idx)
    c_atom = mol.GetAtomWithIdx(c_idx)

    if b_atom.GetSymbol() == "H" or c_atom.GetSymbol() == "H":
        return False
    if not heavy_neighbor_indices(b_atom, c_idx):
        return False
    if not heavy_neighbor_indices(c_atom, b_idx):
        return False
    return True


def find_all_rotatable_bond_torsion_indices(mol):
    """Return one a-b-c-d torsion for every currently rotatable bond b-c."""
    if mol.GetNumAtoms() < 4:
        return []

    torsions = []
    seen_bonds = set()
    for bond in mol.GetBonds():
        if not bond_is_rotatable_for_duplicate_filter(mol, bond):
            continue

        b_idx = bond.GetBeginAtomIdx()
        c_idx = bond.GetEndAtomIdx()
        if b_idx > c_idx:
            b_idx, c_idx = c_idx, b_idx

        central_key = (b_idx, c_idx)
        if central_key in seen_bonds:
            continue
        seen_bonds.add(central_key)

        a_idx = choose_torsion_side_neighbor(mol, b_idx, c_idx)
        d_idx = choose_torsion_side_neighbor(mol, c_idx, b_idx)
        torsion = (a_idx, b_idx, c_idx, d_idx)
        if torsion_indices_are_valid(mol, torsion):
            torsions.append(torsion)

    torsions.sort(key=lambda t: (min(t[1], t[2]), max(t[1], t[2]), t[0], t[3]))
    return torsions


def find_last_growth_dihedral_indices(mol):
    """Return a deterministic a-b-c-d torsion where d is the last added atom."""
    if mol.GetNumAtoms() < 4:
        return None

    d_idx = mol.GetNumAtoms() - 1
    d_atom = mol.GetAtomWithIdx(d_idx)
    candidate_torsions = []
    for c_atom in d_atom.GetNeighbors():
        c_idx = c_atom.GetIdx()
        for b_atom in c_atom.GetNeighbors():
            b_idx = b_atom.GetIdx()
            if b_idx == d_idx:
                continue
            for a_atom in b_atom.GetNeighbors():
                a_idx = a_atom.GetIdx()
                torsion = (a_idx, b_idx, c_idx, d_idx)
                if torsion_indices_are_valid(mol, torsion):
                    candidate_torsions.append(torsion)

    if not candidate_torsions:
        return None
    candidate_torsions.sort(
        key=lambda t: (
            abs(t[2] - (d_idx - 1)),
            abs(t[1] - (d_idx - 2)),
            abs(t[0] - (d_idx - 3)),
        )
    )
    return candidate_torsions[0]


def find_new_ring_attachment_geometries(mol):
    """
    Find geometry created when a ring-type atom is attached to an existing ring.

    Each descriptor contains both a-b-c-d dihedral and b-c-d in-plane angle.
    The attached atom d must still be outside a ring; descriptors are persisted in
    a molecule property so the same geometry remains protected after later growth
    or after the new ring is closed.
    """
    if not mol.HasProp("name") or mol.GetNumAtoms() < 4:
        return []

    names = mol.GetProp("name").split()
    descriptors = []
    for d_atom in mol.GetAtoms():
        d_idx = d_atom.GetIdx()
        if d_idx >= len(names) or names[d_idx] not in ring_types or d_atom.IsInRing():
            continue
        for c_atom in d_atom.GetNeighbors():
            if not c_atom.IsInRing():
                continue
            c_idx = c_atom.GetIdx()
            ring_b_neighbors = sorted(
                n.GetIdx()
                for n in c_atom.GetNeighbors()
                if n.GetIdx() != d_idx and n.IsInRing()
            )
            for b_idx in ring_b_neighbors:
                b_atom = mol.GetAtomWithIdx(b_idx)
                a_candidates = sorted(
                    n.GetIdx()
                    for n in b_atom.GetNeighbors()
                    if n.GetIdx() != c_idx and n.IsInRing()
                )
                if not a_candidates:
                    a_candidates = heavy_neighbor_indices(b_atom, c_idx)
                if not a_candidates:
                    continue
                torsion = (a_candidates[0], b_idx, c_idx, d_idx)
                angle_indices = (b_idx, c_idx, d_idx)
                if torsion_indices_are_valid(mol, torsion) and angle_indices_are_valid(mol, angle_indices):
                    descriptors.append((torsion, angle_indices))

    unique = {}
    for torsion, angle_indices in descriptors:
        unique[(torsion, angle_indices)] = (torsion, angle_indices)
    return [unique[key] for key in sorted(unique)]


def serialize_torsions(torsions):
    return ";".join(",".join(map(str, torsion)) for torsion in torsions)


def parse_torsions(raw, mol):
    torsions = []
    for chunk in raw.split(";"):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 4:
            continue
        try:
            torsion = tuple(int(x) for x in parts)
        except ValueError:
            continue
        if torsion_indices_are_valid(mol, torsion):
            torsions.append(torsion)
    return torsions


def serialize_ring_attachments(descriptors):
    return ";".join(
        f"{','.join(map(str, torsion))}|{','.join(map(str, angle_indices))}"
        for torsion, angle_indices in descriptors
    )


def parse_ring_attachments_prop(mol):
    if not mol.HasProp(PROTECTED_RING_ATTACHMENTS_PROP):
        return []
    descriptors = []
    for chunk in mol.GetProp(PROTECTED_RING_ATTACHMENTS_PROP).split(";"):
        if "|" not in chunk:
            continue
        torsion_raw, angle_raw = chunk.split("|", 1)
        try:
            torsion = tuple(int(x.strip()) for x in torsion_raw.split(","))
            angle_indices = tuple(int(x.strip()) for x in angle_raw.split(","))
        except ValueError:
            continue
        if torsion_indices_are_valid(mol, torsion) and angle_indices_are_valid(mol, angle_indices):
            descriptors.append((torsion, angle_indices))
    return descriptors


def update_branch_geometry_props(mol):
    """Refresh current torsions and retain all previously discovered ring attachments."""
    torsions = find_all_rotatable_bond_torsion_indices(mol)
    if torsions:
        mol.SetProp(ROTATABLE_BOND_TORSIONS_PROP, serialize_torsions(torsions))
    elif mol.HasProp(ROTATABLE_BOND_TORSIONS_PROP):
        mol.ClearProp(ROTATABLE_BOND_TORSIONS_PROP)

    protected = parse_ring_attachments_prop(mol)
    protected.extend(find_new_ring_attachment_geometries(mol))
    unique = {(torsion, angle): (torsion, angle) for torsion, angle in protected}
    protected = [unique[key] for key in sorted(unique)]
    if protected:
        mol.SetProp(PROTECTED_RING_ATTACHMENTS_PROP, serialize_ring_attachments(protected))
    elif mol.HasProp(PROTECTED_RING_ATTACHMENTS_PROP):
        mol.ClearProp(PROTECTED_RING_ATTACHMENTS_PROP)


def parse_rotatable_bond_torsions_prop(mol):
    if not mol.HasProp(ROTATABLE_BOND_TORSIONS_PROP):
        return []
    return parse_torsions(mol.GetProp(ROTATABLE_BOND_TORSIONS_PROP), mol)


def mapped_torsion_difference(query_mol, target_mol, query_to_target, torsion, threshold):
    if not torsion_indices_are_valid(query_mol, torsion):
        return False
    try:
        mapped = tuple(query_to_target[i] for i in torsion)
    except (KeyError, IndexError, TypeError):
        return True
    if not torsion_indices_are_valid(target_mol, mapped):
        return True
    q_pos = query_mol.GetConformer().GetPositions()
    t_pos = target_mol.GetConformer().GetPositions()
    q_dihedral = new_dihedral([q_pos[i] for i in torsion])
    t_dihedral = new_dihedral([t_pos[i] for i in mapped])
    return angular_diff_deg(q_dihedral, t_dihedral) > threshold


def mapped_angle_difference(query_mol, target_mol, query_to_target, angle_indices, threshold):
    if not angle_indices_are_valid(query_mol, angle_indices):
        return False
    try:
        mapped = tuple(query_to_target[i] for i in angle_indices)
    except (KeyError, IndexError, TypeError):
        return True
    if not angle_indices_are_valid(target_mol, mapped):
        return True
    q_pos = query_mol.GetConformer().GetPositions()
    t_pos = target_mol.GetConformer().GetPositions()
    q_angle = calc_angle(*(q_pos[i] for i in angle_indices))
    t_angle = calc_angle(*(t_pos[i] for i in mapped))
    return angular_diff_deg(q_angle, t_angle) > threshold


def mapped_branch_geometry_difference_keeps_branch(query_mol, target_mol, query_to_target):
    """
    Preserve a branch if any current or growth-history geometry is distinct.

    For a sewn ring-type atom, dihedral and in-plane angle are checked together:
    such geometries are duplicates only when BOTH values are within threshold.
    """
    update_branch_geometry_props(query_mol)
    update_branch_geometry_props(target_mol)

    for torsion in parse_rotatable_bond_torsions_prop(query_mol):
        if mapped_torsion_difference(
            query_mol, target_mol, query_to_target, torsion, ROTATABLE_TORSION_THRESHOLD
        ):
            return True

    last_growth = find_last_growth_dihedral_indices(query_mol)
    if last_growth is not None:
        if mapped_torsion_difference(
            query_mol, target_mol, query_to_target, last_growth, LAST_GROWTH_TORSION_THRESHOLD
        ):
            return True
        _, b_idx, c_idx, d_idx = last_growth
        if mapped_angle_difference(
            query_mol,
            target_mol,
            query_to_target,
            (b_idx, c_idx, d_idx),
            LAST_GROWTH_BOND_ANGLE_THRESHOLD,
        ):
            return True

    for torsion, angle_indices in parse_ring_attachments_prop(query_mol):
        torsion_differs = mapped_torsion_difference(
            query_mol,
            target_mol,
            query_to_target,
            torsion,
            RING_ATTACHMENT_TORSION_THRESHOLD,
        )
        plane_angle_differs = mapped_angle_difference(
            query_mol,
            target_mol,
            query_to_target,
            angle_indices,
            RING_ATTACHMENT_PLANE_ANGLE_THRESHOLD,
        )
        if torsion_differs or plane_angle_differs:
            return True

    return False


def branch_geometry_debug_info(mol):
    update_branch_geometry_props(mol)
    rotatable = parse_rotatable_bond_torsions_prop(mol)
    protected = parse_ring_attachments_prop(mol)
    return {
        "rotatable_bond_torsions": serialize_torsions(rotatable) if rotatable else "[]",
        "last_growth_torsion": str(find_last_growth_dihedral_indices(mol)),
        "protected_ring_attachments": serialize_ring_attachments(protected) if protected else "[]",
    }


def is_global_selected_duplicate(mol, global_selected_index, *, rmsd_threshold=2.0):
    """
    Global duplicate criterion:
    1) same canonical SMILES and atom-type multiset;
    2) best no-alignment type-matched RMSD below threshold;
    3) no meaningful difference in current rotatable torsions, last-growth
       geometry, or persisted ring-attachment dihedral/plane-angle geometry.
    """
    smiles = safe_canonical_smiles(mol)
    if smiles is None:
        return False, None

    candidates = global_selected_index.get((smiles, mol_type_counter_key(mol)), [])
    for old_mol in candidates:
        rmsd, mapping = best_type_matched_rmsd_no_alignment(mol, old_mol)
        if rmsd is None or rmsd >= rmsd_threshold:
            continue

        if mapped_branch_geometry_difference_keeps_branch(mol, old_mol, mapping):
            continue

        new_geom_info = branch_geometry_debug_info(mol)
        old_geom_info = branch_geometry_debug_info(old_mol)
        duplicate_info = {
            "rmsd": rmsd,
            "rotatable_bond_torsions": new_geom_info["rotatable_bond_torsions"],
            "last_growth_torsion": new_geom_info["last_growth_torsion"],
            "protected_ring_attachments": new_geom_info["protected_ring_attachments"],
            "old_rotatable_bond_torsions": old_geom_info["rotatable_bond_torsions"],
            "old_last_growth_torsion": old_geom_info["last_growth_torsion"],
            "old_protected_ring_attachments": old_geom_info["protected_ring_attachments"],
            "old_source_iter": old_mol.GetProp("source_iter") if old_mol.HasProp("source_iter") else "NA",
            "old_source_parent_mol": old_mol.GetProp("source_parent_mol") if old_mol.HasProp("source_parent_mol") else "NA",
            "old_global_selected_index": old_mol.GetProp("global_selected_index") if old_mol.HasProp("global_selected_index") else "NA",
        }
        return True, duplicate_info

    return False, None

def add_mol_to_global_selected_index(mol, global_selected_index):
    """Add accepted molecule to the in-memory index to avoid duplicates inside the same final beam."""
    smiles = safe_canonical_smiles(mol)
    if smiles is None or not mol.HasProp("name"):
        return
    global_selected_index[(smiles, mol_type_counter_key(mol))].append(mol)


def filter_selected_against_global_history(selected_candidates, all_selected_path, max_selected, log=None, verbose=False):
    """
    Keep candidates in their ranking order, skipping molecules already present in
    all_selected.sdf according to the global duplicate criterion.

    Each candidate is first passed through update_branch_geometry_props(),
    so all current rotatable-bond torsions are stored in the SDF props before the
    duplicate check and before writing selected/all_selected outputs.
    """
    global_selected_index = build_global_selected_index(all_selected_path)
    kept = []
    skipped = []

    for mol in selected_candidates:
        update_branch_geometry_props(mol)
        is_dup, info = is_global_selected_duplicate(mol, global_selected_index)
        if is_dup:
            mol.SetProp("reason_to_skip", "global selected duplicate")
            if info:
                mol.SetProp("global_duplicate_rmsd", f"{info['rmsd']:.4f}")
                mol.SetProp("global_duplicate_old_source_iter", str(info["old_source_iter"]))
                mol.SetProp("global_duplicate_old_source_parent_mol", str(info["old_source_parent_mol"]))
                mol.SetProp("global_duplicate_old_index", str(info["old_global_selected_index"]))
                mol.SetProp("global_duplicate_rotatable_bond_torsions", str(info.get("rotatable_bond_torsions", "NA")))
                mol.SetProp("global_duplicate_old_rotatable_bond_torsions", str(info.get("old_rotatable_bond_torsions", "NA")))
                mol.SetProp("global_duplicate_last_growth_torsion", str(info.get("last_growth_torsion", "NA")))
                mol.SetProp("global_duplicate_protected_ring_attachments", str(info.get("protected_ring_attachments", "NA")))
                mol.SetProp("global_duplicate_old_last_growth_torsion", str(info.get("old_last_growth_torsion", "NA")))
                mol.SetProp("global_duplicate_old_protected_ring_attachments", str(info.get("old_protected_ring_attachments", "NA")))
            skipped.append(mol)
            continue

        kept.append(mol)
        add_mol_to_global_selected_index(mol, global_selected_index)
        if len(kept) >= max_selected:
            break

    if verbose:
        print(f"Global selected duplicate filter: {len(kept)} kept, {len(skipped)} skipped")
    if log is not None:
        log.write(f"Global selected duplicate filter: {len(kept)} kept, {len(skipped)} skipped\n")
        for i, mol in enumerate(skipped[:20]):
            msg = (
                f"\tSkipped global duplicate {i}: "
                f"NBS={mol.GetProp('NBS') if mol.HasProp('NBS') else 'NA'}, "
                f"rmsd={mol.GetProp('global_duplicate_rmsd') if mol.HasProp('global_duplicate_rmsd') else 'NA'}, "
                f"old_iter={mol.GetProp('global_duplicate_old_source_iter') if mol.HasProp('global_duplicate_old_source_iter') else 'NA'}, "
                f"old_parent={mol.GetProp('global_duplicate_old_source_parent_mol') if mol.HasProp('global_duplicate_old_source_parent_mol') else 'NA'}, "
                f"rotatable_torsions={mol.GetProp('global_duplicate_rotatable_bond_torsions') if mol.HasProp('global_duplicate_rotatable_bond_torsions') else 'NA'}, "
                f"last_growth={mol.GetProp('global_duplicate_last_growth_torsion') if mol.HasProp('global_duplicate_last_growth_torsion') else 'NA'}, "
                f"ring_attachments={mol.GetProp('global_duplicate_protected_ring_attachments') if mol.HasProp('global_duplicate_protected_ring_attachments') else 'NA'}\n"
            )
            log.write(msg)
        if len(skipped) > 20:
            log.write(f"\t... {len(skipped) - 20} more global duplicates skipped\n")

    return kept, skipped


class GenerationMode(Enum):
    SCAFFOLD = "scaffold"
    PERIPHERY = "periphery"


# поменял логику довешивания кислородов на серу, не нужно теперь менять логику алгоритма
def launch_nbg(
    args,
    log,
    loaded_neurals,
    loaded_pharma_neurals,
    cat,
    device,
    reference=None,
    dot_pred_batch_size=-1,
    *,
    verbose=True,
    precise=False,
    debug=False,
    super_output=False,
    ref_match_debug=False,
    ref_match_debug_path="ref.sdf",
    no_task_preference=False,
):
    # fix initial time for optimization purposes
    start_time = time.time()

    dict_of_thresholds = {
        ".=O": 0.875,
        "Car": 0.75,
        "Cs2": 0.75,
        "Cs3": 0.75,
        "Csp": 0.75,
        "Hal": 0.875,
        "Nac": 0.875,
        "Nd0": 0.875,
        "Nd+": 0.875,
        "O_a": 0.875,
        "O_d": 0.875,
        "SO2": 0.75,
        "Sul": 0.75,
    }

    dict_of_coefs = {
        ".=O": 1.0,
        "Car": 0.2,
        "Cs2": 0.2,
        "Cs3": 0.2,
        "Csp": 0.2,
        "Hal": 0.3,
        "Nac": 1.0,
        "Nd0": 1.0,
        "Nd+": 1.0,
        "O_a": 1.0,
        "O_d": 0.3,
        "SO2": 0.2,
        "Sul": 0.2,
    }

    symbols = {
        "Car": "C",
        "O_a": "O",
        "Cs3": "C",
        "C3r": "C",
        "Nac": "N",
        "Nd+": "N",
        "Nd0": "N",
        "Cs2": "C",
        "C2r": "C",
        ".=O": "O",
        "Hal": "Cl",
        "O_d": "O",
        "Csp": "C",
        "Sul": "S",
        "SO2": "S",
    }

    bond_types = {
        "SINGLE": Chem.rdchem.BondType.SINGLE,
        "DOUBLE": Chem.rdchem.BondType.DOUBLE,
        "TRIPLE": Chem.rdchem.BondType.TRIPLE,
    }

    bonds_df = pd.read_csv(f"{args['path_to_working_dir']}/bonds_df.csv", index_col=0)["0"]
    dihedrals_df = pd.read_csv(f"{args['path_to_working_dir']}/dihedrals_df.csv", index_col=0)

    mcf_df, final_mcf_df = [], []
    filtration_mode = args["final_mcf_mode"]
    output_mcf = filtration_mode != "none"
    if output_mcf:
        mcf_df = pd.concat(
            [pd.read_csv(f"{args['path_to_working_dir']}/building_mcf.csv", index_col=0), pd.read_csv(f"{args['path_to_working_dir']}/additional_mcf.csv", index_col=0)],
            ignore_index=True,
        )
        final_mcf_df = pd.concat(
            [pd.read_csv(f"{args['path_to_working_dir']}/final_mcf.csv", index_col=0), pd.read_csv(f"{args['path_to_working_dir']}/additional_mcf.csv", index_col=0)],
            ignore_index=True,
        )

        if filtration_mode == "simplified":
            not_needed = ["basic", "strict"]

        elif filtration_mode == "basic":
            not_needed = ["strict"]

        else:
            not_needed = []

        mcf_df = mcf_df[~mcf_df.group.isin(not_needed)]
        final_mcf_df = final_mcf_df[~final_mcf_df.group.isin(not_needed)]

    path_to_working_dir = Path(args["path_to_working_dir"])

    all_file = path_to_working_dir / "all_brokens.sdf" # для тестирования
    all_out = path_to_working_dir / "all_selected.sdf" # для тестирования
    all_scored_out = path_to_working_dir / "all_scored.sdf"  # для тестирования

    mode = GenerationMode(args["mode"])

    args["allow_cs3_in_arom_rings"] = False
    args["allow_chair_only"] = True
    args["allow_axial_groups"] = False
    args["allow_condensed_alicyclics"] = False
    args["allow_paired_nd0"] = False
    args["allow_nd0_nac_pair"] = False
    args["allow_double_bonded_arom_rings"] = False
    args["allow_alkenes"] = False
    args["allow_linker_sul"] = False
    args["nac_outside_arom_ring"] = False
    if verbose:
        print("Generator launched\nInput parameters:")
    log.write("Generator launched\nInput parameters:\n")
    for key, arg in args.items():
        if key != "ref_lib_to_pass":
            if verbose:
                print(f"\t{key}: {arg}")
            log.write(f"\t{key}: {arg}\n")

    # probe checking module (work in progress)
    probe = path_to_working_dir / str(args["probe_name"])

    ref_debug_reference = None
    ref_debug_dir = None
    ref_debug_counters = None

    if ref_match_debug:
        ref_debug_reference, ref_debug_dir, ref_debug_counters = init_ref_match_debug(
            path_to_working_dir,
            ref_path=ref_match_debug_path,
        )

        log.write(f"[REF DEBUG] enabled, reference = {path_to_working_dir / ref_match_debug_path}\n")

    ref_lib_to_pass = args["ref_lib_to_pass"]

    if output_mcf:
        if reference:
            ref_lib_to_pass = list(Chem.SDMolSupplier(str(path_to_working_dir / "ref.sdf")))
        if mode is GenerationMode.PERIPHERY:
            ref_lib_to_pass.extend(list(Chem.SDMolSupplier(str(probe)))+[Chem.SDMolSupplier("FLT3_invitro_check_paper_periphery_onestep/temp_iter_6_broken.sdf")[0]]) # исключили мсф для серы кратности 2 в кольце 6, чтобы достроить периферию
        if ref_lib_to_pass:  # make exception if no mcfs
            if verbose:
                print("Reference library to pass filters initiated")
            log.write("Reference library to pass filters initiated\n")
            args, mcf_df, final_mcf_df, disabled_mcf_ids, final_disabled_mcf_ids, changed_parameters = lib_to_pass_checker(ref_lib_to_pass, args, mcf_df, final_mcf_df)
            for i in [disabled_mcf_ids, final_disabled_mcf_ids]:
                if i:
                    if verbose:
                        print(f"Building mcf with ids {i} disabled")
                    log.write(f"Building mcf with ids {i} disabled\n")
            if changed_parameters:
                if verbose:
                    print("The following rules changed:")
                log.write("The following rules changed:\n")
                for key, value in changed_parameters.items():
                    if verbose:
                        print(f"\t{key}: {value}")
                    log.write(f"\t{key}: {value}\n")
            if verbose:
                print("Filters tunung successful")
            log.write("Filters tunung successful\n")

        else:
            if verbose:
                print("No reference library to pass filters present")
            log.write("No reference library to pass filters present\n")

        mcf_fragments, fragments_names, fragments_counts = check_fragments(mcf_df)
        final_mcf_fragments, final_fragments_names, final_fragments_counts = check_fragments(final_mcf_df)

    dct = args["limits_dict"]
    if mode is GenerationMode.SCAFFOLD:
        dct["O_d"] = 0
        dct["Hal"] = 0
    elem_dct = args["elem_dict"]
    NO_limit = args["no_limit"]
    max_num_low_cycles = args["max_num_low_cycles"]
    protein_name = str(args["protein_name"])
    grid_name = args["grid_name"]
    phar_model = args["phar_model"]
    allow_axial_groups = args["allow_axial_groups"]
    allow_chair_only = args["allow_chair_only"]
    allow_Cs3_in_arom_rings = args["allow_cs3_in_arom_rings"]
    allow_condensed_alicyclics = args["allow_condensed_alicyclics"]
    allow_paired_Nd0 = args["allow_paired_nd0"]
    Nac_outside_arom_ring = args["nac_outside_arom_ring"]
    skip_minimization = args["skip_minimization"]
    allow_Nd0_Nac_pair = args["allow_nd0_nac_pair"]
    allow_double_bonded_arom_rings = args["allow_double_bonded_arom_rings"]
    allow_linker_Sul = args["allow_linker_sul"]
    allow_alkenes = args["allow_alkenes"]
    max_num_positive_charges = int(args["max_num_positive_charges"])
    max_num_negative_charges = int(args["max_num_negative_charges"])
    max_num_chiral_centers = args["max_num_chiral_centers"]
    allowed_atoms_idxs = args["allowed_atoms_idxs"]
    max_len = args["max_acyclic_chain_length"]
    min_mol_size = args["min_mol_size"]
    args["max_poses_of_one_structure"]
    max_clashes = args["max_clashes"]
    number_of_poses = args["number_of_poses_on_iter"]

    max_execution_time = args["max_execution_time"] * 60 * 60
    time_management_mode = args["time_management_mode"]
    branch_time_factor = args["branch_time_factor"]
    min_branch_seconds = args["min_branch_seconds"]
    max_branch_seconds = args["max_branch_seconds"]
    dynamic_split_until_iter = args.get("dynamic_split_until_iter", 3)
    max_num_charged_atoms = args["max_num_charged_atoms"]

    # protein cropping
    path_protein = str(path_to_working_dir / protein_name)
    ppdf = PandasPdb().read_pdb(path_protein)

    protein_df = pd.concat([ppdf.df["ATOM"], ppdf.df["HETATM"]])

    protein_df = protein_df.reset_index(drop=True)

    if protein_df["charge"].isnull().all():
        with open(path_protein, "r") as f:
            txt = f.readlines()

        txt_filtered = []

        for line in txt:
            if line.split()[0] in ["ATOM", "HETATM"]:
                txt_filtered.append(line.split())

        charge_list = []

        for line in txt_filtered:
            if line[-1][-1] in ["+", "-"]:
                charge_list.append(np.float64(line[-1][-2:][::-1]))
            else:
                charge_list.append(np.float64(np.nan))

        protein_df["charge"] = charge_list

    grid_df = PandasPdb().read_pdb(f"{path_to_working_dir}/{grid_name}").df["HETATM"]
    coords = protein_df[["x_coord", "y_coord", "z_coord"]]
    grid_coords = grid_df[["x_coord", "y_coord", "z_coord"]]
    dists = euclidean_distances(coords, grid_coords)
    dists_df = pd.DataFrame(dists)
    dists_df["min_dist"] = dists_df.min(axis=1)
    dists_df = dists_df[dists_df.min_dist < 7]
    protein_df["unique_aa_marker"] = protein_df["chain_id"] + protein_df["residue_number"].apply(str)
    selected_aas = protein_df.loc[dists_df.index, :]["unique_aa_marker"].unique()
    protein_df = protein_df[protein_df["unique_aa_marker"].isin(selected_aas)]
    protein_df = protein_df.reset_index(drop=True)

    ppdf.df["ATOM"] = protein_df[protein_df["record_name"] == "ATOM"]
    ppdf.df["HETATM"] = protein_df[protein_df["record_name"] == "HETATM"].reset_index(drop=True)
    records_list = ["ATOM", "HETATM", "OTHERS"] if "HETATM" in protein_df["record_name"].unique() else ["ATOM"]
    ppdf.to_pdb(path=str(path_to_working_dir / "cropped_protein.pdb"), records=records_list, gz=False, append_newline=True)

    # ppdf = PandasPdb().read_pdb(str(path_to_working_dir / protein_name))
    # protein_df = ppdf.df["ATOM"]
    # grid_df = PandasPdb().read_pdb(f"{path_to_working_dir}/{grid_name}").df["HETATM"]
    # coords = protein_df[["x_coord", "y_coord", "z_coord"]]
    # grid_coords = grid_df[["x_coord", "y_coord", "z_coord"]]
    # dists = euclidean_distances(coords, grid_coords)
    # dists_df = pd.DataFrame(dists)
    # dists_df["min_dist"] = dists_df.min(axis=1)
    # dists_df = dists_df[dists_df.min_dist < 7]
    # protein_df["unique_aa_marker"] = protein_df["chain_id"] + protein_df["residue_number"].apply(str)
    # selected_aas = protein_df.loc[dists_df.index, :]["unique_aa_marker"].unique()
    # protein_df = protein_df[protein_df["unique_aa_marker"].isin(selected_aas)]
    # protein_df = protein_df.reset_index(drop=True)
    # ppdf.df["ATOM"] = protein_df
    # ppdf.to_pdb(path=str(path_to_working_dir / "cropped_protein.pdb"), records=["ATOM"], gz=False, append_newline=True)
    os.system(
        f"obabel {path_to_working_dir / 'cropped_protein.pdb'} -O "  # noqa: S605
        f"{path_to_working_dir / 'cropped_protein.pdbqt'} -xr 1>/dev/null 2>&1"
    )

    # grod center for docking
    gridbox_center_x, gridbox_center_y, gridbox_center_z = grid_coords.mean(axis=0)
    # protein for graph maker
    prot = PandasPdb().read_pdb(f"{path_to_working_dir}/cropped_protein.pdb").df["ATOM"]
    prot = prot[~prot["element_symbol"].isin(["H", ""])]
    prot = prot.drop(
        columns=[
            "record_name",
            "atom_number",
            "blank_1",
            "element_symbol",
            "blank_3",
            "charge",
            "alt_loc",
            "blank_2",
            "residue_number",
            "chain_id",
            "occupancy",
            "b_factor",
            "blank_4",
            "segment_id",
            "line_idx",
            "insertion",
        ]
    )

    # coords for crutial interaction
    check_crucial_interaction = args["important_amino_acid_number"] and args["important_atom_name"] and args["ligand_role"]
    if check_crucial_interaction:
        important_amino_acid_number = args["important_amino_acid_number"]
        important_atom_name = args["important_atom_name"]
        ligand_role = args["ligand_role"]
        atom_df = protein_df[(protein_df.atom_name == important_atom_name) & (protein_df.residue_number == important_amino_acid_number)]
        atom_df = atom_df.reset_index(drop=True)
        crucial_x = atom_df.x_coord[0]
        crucial_y = atom_df.y_coord[0]
        crucial_z = atom_df.z_coord[0]
        crucial_coords = np.array([crucial_x, crucial_y, crucial_z])
    fragments = Chem.SDMolSupplier(str(path_to_working_dir / args["frag_lib"]))
    fragments_scen1 = [f for f in fragments if f.GetProp("scen") == "1"]
    fragments_scen2 = [f for f in fragments if f.GetProp("scen") == "2"]
    method = args["scoring_method"]
    iter_n = 1  # iter to start with (probe placing is considered as iter 0)
    iterations_number = args["num_heavy_atoms_to_add"] + iter_n + 1  # 1 extra iter to write output only
    shutil.copyfile(probe, str(path_to_working_dir / "iter0_selected.sdf"))

    init_forbidden_atoms = (
        [a.GetIdx() for a in Chem.SDMolSupplier(str(path_to_working_dir / "iter0_selected.sdf"))[0].GetAtoms() if a.GetIdx() not in allowed_atoms_idxs]
        if allowed_atoms_idxs
        else []
    )
    a = {i: 0 for i in range(1, 40)}


    # Old time manager state
    time_restrictions = {}
    time_restrictions[0] = max_execution_time

    # Dynamic time manager state
    deadline_by_iter = {}
    deadline_by_iter[0] = start_time + max_execution_time

    timeout_manager = None
    if time_management_mode == "dynamic_snapshot":
        timeout_manager = TimeoutSnapshotManager(
            path_to_working_dir=path_to_working_dir,
            max_execution_time=max_execution_time,
            start_time=start_time,
            max_branch_seconds=max_branch_seconds,
            log=log,
            verbose=verbose,
        )

        # Use one source of truth for global deadline.
        deadline_by_iter[0] = timeout_manager.global_deadline

    if verbose:
        print("Generation cycle started")
    log.write("Generation cycle started\n")
    with Pool(processes=4) as multiprocpool:
        # main cycle
        while iter_n != 0:
            if verbose:
                print(f"!!!! {iter_n} !!!!")
            log.write(f"!!!! {iter_n} !!!!\n")

            if iter_n == iterations_number:  # if iterations limit reached
                if verbose:
                    print("Go one step back due to limit reached")
                log.write("Go one step back due to limit reached\n")
                iter_n -= 1
                a[iter_n] += 1
                continue

            elif a[iter_n] == len(Chem.SDMolSupplier(str(path_to_working_dir / f"iter{iter_n - 1}_selected.sdf"))) and iter_n != 1:
                if time_management_mode == "dynamic_snapshot":
                    timeout_manager.mark_parent_completed(a=a, iter_level=iter_n)

                if verbose:
                    print("Go one step back due to all molecules studied")
                log.write("Go one step back due to all molecules studied\n")

                a[iter_n] = 0
                iter_n -= 1
                a[iter_n] += 1

                if time_management_mode in ["dynamic", "dynamic_snapshot"]:
                    for key in list(deadline_by_iter.keys()):
                        if key > iter_n:
                            del deadline_by_iter[key]

                continue

            elif a[iter_n] == len(Chem.SDMolSupplier(str(path_to_working_dir / f"iter{iter_n - 1}_selected.sdf"))) and iter_n == 1:
                if time_management_mode == "dynamic_snapshot":
                    restored_iter = timeout_manager.restore_next_if_available(
                        a=a,
                        deadline_by_iter=deadline_by_iter,
                    )

                    if restored_iter is not None:
                        iter_n = restored_iter
                        continue

                if verbose:
                    print("All possible combinations were generated")
                log.write("All possible combinations were generated\n")
                break

            # ------------------------------------------------------------
            # dynamic_snapshot:
            # If restored DFS reaches a branch that was already fully completed
            # by normal DFS, skip it. Timeout branches are not skipped.
            # ------------------------------------------------------------
            if time_management_mode == "dynamic_snapshot":
                if timeout_manager.should_skip_completed_branch(a=a, iter_level=iter_n):
                    a[iter_n] += 1

                    for key in a:
                        if key > iter_n:
                            a[key] = 0

                    for key in list(deadline_by_iter.keys()):
                        if key >= iter_n:
                            del deadline_by_iter[key]

                    continue

            # common generation path
            if verbose:
                print(f"Generation started for iter {iter_n} from mol {a[iter_n]}")
            log.write(f"Generation started for iter {iter_n} from mol {a[iter_n]}\n")
            gen_start = time.time()
            probes = Chem.SDMolSupplier(str(path_to_working_dir / f"iter{iter_n - 1}_selected.sdf"))
            out = Chem.SDWriter(str(path_to_working_dir / f"iter{iter_n}_generated.sdf"))
            bad_out = Chem.SDWriter(str(path_to_working_dir / f"iter{iter_n}_broken.sdf"))
            probe = probes[a[iter_n]]
            compare = []
            probe_pos = probe.GetConformer().GetPositions()
            # case when probe has 2 uncycled ring types - duplicate probe and make one ring type forbidden
            if iter_n == 1:
                uncycled = get_uncycled_atoms(probe)
                if len(uncycled) > 1 and len(probes) == 1:
                    if verbose:
                        print("Two or more uncycled atoms detected, probe duplicated")
                    log.write("Two or more uncycled atoms detected, probe duplicated\n")
                    with Chem.SDWriter(str(path_to_working_dir / f"iter{iter_n - 1}_selected.sdf")) as w:
                        probe.SetProp("dummy", " ".join(list(map(str, [uncycled[0]]))))
                        w.write(probe)
                        probe.ClearProp("dummy")
                        probe.SetProp("dummy", " ".join(list(map(str, [uncycled[1]]))))
                        w.write(probe)
                        probe.ClearProp("dummy")
                        continue

            # ------------------------------------------------------------
            # Execution time restriction block
            # Two modes:
            #   old     - original behavior: time is split only on iters 1, 2, 3
            #   dynamic - time is split on every iter.
            #             If parent budget is exhausted, return to parent.
            #             If current child budget is exhausted, skip only current child mol.
            # ------------------------------------------------------------
            if time_management_mode == "old":
                if iter_n in [1, 2, 3]:
                    mount_time = time.time() - start_time
                    time_limit = time_restrictions[iter_n - 1]
                    probes_left = len(probes) - a[iter_n]

                    if probes_left <= 0:
                        probes_left = 1

                    time_left = time_limit - mount_time
                    time_restrictions[iter_n] = time_left / probes_left
                    time_to_return = mount_time + time_restrictions[iter_n]

                    if verbose:
                        print(
                            f"""Timestamp: {probes_left} probes left,
                                {round(time_restrictions[1] - mount_time)} seconds left in total,
                                {round(time_left)} seconds left from prev iter,
                                {round(time_restrictions[iter_n])} seconds left for the current branch"""
                        )

                    log.write(
                        f"""Timestamp: {probes_left} probes left,
                                {round(time_restrictions[1] - mount_time)} seconds left in total,
                                {round(time_left)} seconds left from prev iter,
                                {round(time_restrictions[iter_n])} seconds left for the current branch\n"""
                    )
                else:
                    # In old mode, deeper iters reuse previously defined time_to_return.
                    # This preserves the original behavior.
                    pass

            elif time_management_mode in ["dynamic", "dynamic_snapshot"]:
                parent_iter = iter_n - 1

                if parent_iter not in deadline_by_iter:
                    deadline_by_iter[parent_iter] = start_time + max_execution_time

                parent_deadline = deadline_by_iter[parent_iter]

                probes_left = len(probes) - a[iter_n]
                if probes_left <= 0:
                    probes_left = 1

                time_left = parent_deadline - time.time()

                # Parent branch budget is exhausted.
                # Save a snapshot everywhere, but after dynamic_split_until_iter
                # return old-like to the split boundary instead of cascading
                # through every intermediate depth.
                if time_left <= 0:
                    if iter_n > 1:
                        if time_management_mode == "dynamic_snapshot":
                            timeout_manager.save_timeout_if_new(
                                a=a,
                                iter_level=iter_n,
                                mol_idx=a[iter_n],
                                reason="parent_branch_budget_exceeded",
                            )

                        old_iter = iter_n
                        return_iter = parent_timeout_return_iter(
                            iter_n,
                            dynamic_split_until_iter,
                        )

                        if verbose:
                            print(f"Time for parent branch exceeded, return to iter {return_iter}")
                        log.write(f"Time for parent branch exceeded, return to iter {return_iter}\n")

                        a[old_iter] = 0
                        iter_n = return_iter
                        a[iter_n] += 1

                        for key in a:
                            if key > iter_n:
                                a[key] = 0

                        for key in list(deadline_by_iter.keys()):
                            if key > iter_n:
                                del deadline_by_iter[key]

                        continue

                    else:
                        if verbose:
                            print("Global time limit exceeded")
                        log.write("Global time limit exceeded\n")
                        break

                # Hybrid time policy:
                # 1) restored snapshot: one clean retry budget;
                # 2) iter <= dynamic_split_until_iter: dynamic split;
                # 3) iter > dynamic_split_until_iter: old-like inherited deadline.
                budget_mode = "dynamic_split"
                retry_budget = None

                if time_management_mode == "dynamic_snapshot":
                    retry_budget = timeout_manager.active_snapshot_budget_override(
                        iter_n=iter_n,
                        time_left=time_left,
                    )

                if retry_budget is not None:
                    branch_budget = retry_budget
                    time_to_return = time.time() + branch_budget
                    deadline_by_iter[iter_n] = time_to_return
                    budget_mode = "restored_snapshot"

                    if verbose:
                        print(
                            f"Restored snapshot budget override: "
                            f"iter {iter_n}, mol {a[iter_n]}, "
                            f"{round(branch_budget)} seconds"
                        )
                    log.write(
                        f"Restored snapshot budget override: "
                        f"iter {iter_n}, mol {a[iter_n]}, "
                        f"{round(branch_budget)} seconds\n"
                    )

                elif iter_n <= dynamic_split_until_iter:
                    fair_budget = (time_left / probes_left) * branch_time_factor

                    # Apply min_branch_seconds only if there is enough parent time
                    # to give this minimum to every remaining probe.
                    if time_left >= probes_left * min_branch_seconds:
                        branch_budget = max(min_branch_seconds, fair_budget)
                    else:
                        branch_budget = fair_budget

                    branch_budget = min(branch_budget, max_branch_seconds, time_left)
                    time_to_return = time.time() + branch_budget
                    deadline_by_iter[iter_n] = time_to_return
                    budget_mode = "dynamic_split"

                else:
                    # Old-like behavior after dynamic_split_until_iter:
                    # do not split time further at every deep atom-add iteration.
                    # The current deep branch inherits the parent's deadline.
                    time_to_return = parent_deadline
                    deadline_by_iter[iter_n] = parent_deadline
                    branch_budget = max(0, time_to_return - time.time())
                    budget_mode = "old_like_inherited"

                current_branch_time_left = max(0, time_to_return - time.time())

                if verbose:
                    print(
                        f"""Timestamp: {probes_left} probes left on iter {iter_n},
                                {round(max(0, deadline_by_iter[0] - time.time()))} seconds left in total,
                                {round(max(0, time_left))} seconds left from parent iter,
                                {round(current_branch_time_left)} seconds left for the current branch,
                                time policy: {budget_mode}, dynamic_split_until_iter={dynamic_split_until_iter}"""
                    )

                log.write(
                    f"""Timestamp: {probes_left} probes left on iter {iter_n},
                                {round(max(0, deadline_by_iter[0] - time.time()))} seconds left in total,
                                {round(max(0, time_left))} seconds left from parent iter,
                                {round(current_branch_time_left)} seconds left for the current branch,
                                time policy: {budget_mode}, dynamic_split_until_iter={dynamic_split_until_iter}\n"""
                )
            forbidden_atoms = init_forbidden_atoms.copy()
            if probe.HasProp("dummy"):
                forbidden_atoms += list(map(int, probe.GetProp("dummy").split()))

            probe_name = probe.GetProp("name").split()
            probe_len = len(probe.GetAtoms())

            # list of mols to start with on each iter to reproduce the mol
            scenario = f"{probe.GetProp('scenario')} {a[iter_n]}" if probe.HasProp("scenario") else str(a[iter_n])

            # list of previous tasks
            if probe.HasProp("tasks"):
                tasks = probe.GetProp("tasks")
            else:
                tasks = "probe_placing"
                probe.SetProp("tasks", tasks)

            Chem.Kekulize(probe)
            current_task = (
                probe.GetProp("current_task") if probe.HasProp("current_task") else task_initializer(mode, probe, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring)
            )
            if verbose:
                print(current_task)
            log.write(current_task + "\n")
            # change forbidden atoms if the task is 'build_orthosubstituent_to_rotated_amide', skip probe, if can't build to needed ortho
            if current_task == 'build_orthosubstituent_to_rotated_amide':
                orthosubstituent_ids = probe.GetProp('build_orthosubstituent_to_rotated_amide').split()
                crucial_atoms = [int(i) for i in orthosubstituent_ids if int(i) not in forbidden_atoms]
                if not crucial_atoms:
                    iter_n -= 1
                    a[iter_n] += 1
                    for key in a:
                        if key > iter_n:
                            a[key] = 0
                    if verbose:
                        print(f"No avaliable ortho position to build for amide correction, skipping probe, return to iter {iter_n}")
                    log.write(f"No avaliable ortho position to build for amide correction, skipping probe, return to iter {iter_n}\n")
                    continue
                else:
                    forbidden_atoms = [a.GetIdx() for a in probe.GetAtoms() if a.GetIdx() not in crucial_atoms]
            # if more than 6 attempts to close any ring - bad way
            tasks_list = tasks.split()
            count_corrector = 0
            if probe.HasProp('initiate_ring_bisect'):
                count_corrector = 1
            if probe.HasProp('initiate_ring_110'):
                count_corrector = 2
            ring_task_len = 6 - count_corrector
            if len(tasks_list) > 6 and current_task.endswith("ring") and all(t.endswith("ring") for t in tasks_list[-ring_task_len:]):
                if verbose:
                    print("Too many ring closing attempts, skip to the next molecule")
                log.write("Too many ring closing attempts, skip to the next molecule\n")
                a[iter_n] += 1
                continue

            (
                scen1_allowed,
                scen2_allowed,
                scen1_ids_allowed,
                scen1_fragments_allowed,
                scen2_ids_allowed,
                scen2_fragments_allowed,
            ) = scen_mapping(
                current_task,
                probe,
                probe_name,
                probe_len,
                tasks,
                iter_n,
                iterations_number,
                allow_Cs3_in_arom_rings,
                allow_linker_Sul,
            )

            # if reference ligand to reproduce present (for test launches only)
            if reference:
                Chem.SanitizeMol(probe)
                if probe.HasSubstructMatch(reference) and reference.HasSubstructMatch(probe):
                    rms = AllChem.GetBestRMS(probe, reference)
                    if not precise or rms < 0.5:
                        log.write("Match found\n")
                        yield probe
                        if super_output:
                            save_mols_for_super_output(path_to_working_dir, probe)
                        #return
                Chem.Kekulize(probe)


            # ------------------------------------------------------------
            # Timeout handling
            # ------------------------------------------------------------
            if time_management_mode == "old":
                # Original behavior.
                if time.time() - start_time > time_to_return:
                    if iter_n > 3:
                        iter_n = 4
                    iter_n -= 1
                    a[iter_n] += 1

                    for key in a:
                        if key > iter_n:
                            a[key] = 0

                    if verbose:
                        print(f"Time for probe exceeded, return to iter {iter_n}")
                    log.write(f"Time for probe exceeded, return to iter {iter_n}\n")

                    continue

            elif time_management_mode in ["dynamic", "dynamic_snapshot"]:
                if time.time() > time_to_return:
                    if verbose:
                        print(
                            f"Time for current branch exceeded, "
                            f"skip mol {a[iter_n]} on iter {iter_n}"
                        )
                    log.write(
                        f"Time for current branch exceeded, "
                        f"skip mol {a[iter_n]} on iter {iter_n}\n"
                    )

                    if time_management_mode == "dynamic_snapshot":
                        timeout_manager.safe_close_writer(out)
                        timeout_manager.safe_close_writer(bad_out)

                        timeout_manager.save_timeout_if_new(
                            a=a,
                            iter_level=iter_n,
                            mol_idx=a[iter_n],
                            reason="current_branch_timeout",
                        )

                    old_iter = iter_n
                    return_iter = current_timeout_return_iter(
                        iter_n,
                        dynamic_split_until_iter,
                    )

                    if return_iter == old_iter:
                        # Dynamic behavior before / on split boundary:
                        # skip only the current molecule from iter{iter_n - 1}_selected.sdf.
                        a[iter_n] += 1
                    else:
                        # Old-like behavior after split boundary:
                        # return to dynamic_split_until_iter and move that branch forward.
                        if verbose:
                            print(
                                f"Old-like deep timeout return: "
                                f"from iter {old_iter} to iter {return_iter}"
                            )
                        log.write(
                            f"Old-like deep timeout return: "
                            f"from iter {old_iter} to iter {return_iter}\n"
                        )

                        a[old_iter] = 0
                        iter_n = return_iter
                        a[iter_n] += 1

                    for key in a:
                        if key > iter_n:
                            a[key] = 0

                    for key in list(deadline_by_iter.keys()):
                        if key >= iter_n:
                            del deadline_by_iter[key]

                    continue

            # last 1 iter is used only to write output
            if iter_n == iterations_number - 1:
                a[iter_n] += 1
                continue

            pool = []
            written = 0
            synthetic = False
            for atom in probe.GetAtoms():
                idx1 = atom.GetIdx()
                if idx1 in forbidden_atoms:
                    continue

                # define generation scenes
                scen1 = scen1_allowed and idx1 in scen1_ids_allowed
                scen2 = scen2_allowed and idx1 in scen2_ids_allowed
                if not (scen1 or scen2):
                    continue
                atom_type = probe_name[idx1]

                # synthetic part
                if atom_type == "Car" and atom.IsInRing():
                    synthetic = True
                    for new_atom_type in set(scen1_fragments_allowed + scen2_fragments_allowed + ["Sul"]):
                        neighbs = [a.GetIdx() for a in atom.GetNeighbors()]
                        p1, p2, p3 = probe_pos[neighbs[0]], probe_pos[idx1], probe_pos[neighbs[1]]
                        v1 = p1 - p2
                        v2 = p3 - p2
                        v1 /= np.linalg.norm(v1)
                        v2 /= np.linalg.norm(v2)
                        bisector = (v1 + v2) / np.linalg.norm(v1 + v2)
                        sections = [bisector]
                        if new_atom_type in ["Car", "C3r", "C2r", "Sul", "O_a", "Nac", "Nd0"]:
                            deviation1 = (v1 * 1.25 + v2) / np.linalg.norm(v1 * 1.25 + v2)
                            sections.append(deviation1)
                            deviation2 = (v1 + v2 * 1.25) / np.linalg.norm(v1 + v2 * 1.25)
                            sections.append(deviation2)
                        for bis in sections:
                            bond_type = Chem.rdchem.BondType.DOUBLE if current_task == "fix_Car" else Chem.rdchem.BondType.SINGLE
                            bond_len = bonds_df[atom_type + new_atom_type + str(bond_type)]
                            point_on_bisector = p2 - bis * bond_len
                            edmol = Chem.EditableMol(probe)
                            idx11 = edmol.AddAtom(Chem.Atom(symbols[new_atom_type]))
                            edmol.AddBond(idx1, idx11, bond_type)
                            frag = edmol.GetMol()
                            frag.GetConformer().SetAtomPosition(idx11, point_on_bisector)
                            name = f"{' '.join(probe_name)} {new_atom_type}"
                            frag.SetProp("name", name)
                            try:
                                Chem.SanitizeMol(frag)
                                frag = ionize(frag)
                            except Exception:
                                frag.SetProp("reason_to_skip", "can't sanitize/ionize")
                                bad_out.write(frag)
                                continue
                            if not atom_counters_checker(frag, name, NO_limit, elem_dct, dct):
                                frag.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                                bad_out.write(frag)
                                continue
                            if probe.HasProp("dummy"):
                                frag.SetProp("dummy", probe.GetProp("dummy"))
                            if (
                                output_mcf
                                and not passes_all_mcf(
                                    frag,
                                    fragments_names=fragments_names,
                                    mcf=mcf_fragments,
                                    mcf_counts=fragments_counts,
                                    tolerance=0,
                                    return_all_unpassed=True,
                                )["passed"]
                            ):
                                frag.SetProp("reason_to_skip", "mcf")
                                bad_out.write(frag)
                                continue
                            if frag.HasProp('build_orthosubstituent_to_rotated_amide'):
                                frag.SetProp('flag ortho', '')
                                frag.ClearProp('build_orthosubstituent_to_rotated_amide')
                            next_task = task_initializer(mode, frag, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring)
                            frag.SetProp("current_task", next_task)
                            pool.append(frag)
                    continue
                if current_task == "build_to_Nac" and len(list(atom.GetNeighbors())) == 1:
                    synthetic = True
                    rules_dct = {  # atom_type: angles, bond_orders, theta to rotate, dihedral angs to exclude
                        "C3r": [[120, 125, 135], ["SINGLE"], 15, []],
                        "Car": [[120, 125, 135], ["SINGLE", "DOUBLE"], 30, []],
                        "C2r": [[120], ["SINGLE", "DOUBLE"], 30, []],
                        "Nac": [[120, 125], ["SINGLE", "DOUBLE"], 30, []],
                        "Cs3": [[120], ["SINGLE"], 15, []],
                        "Cs2": [[120], ["SINGLE"], 30, [90]],
                        "SO2": [[120], ["SINGLE"], 15, []],
                    }
                    for new_atom_type, rules in rules_dct.items():
                        if new_atom_type not in ring_types and (probe.HasProp('initiate_ring_bisect') or probe.HasProp('initiate_ring_110')):
                            continue
                        neighb = atom.GetNeighbors()[0]
                        v = probe_pos[neighb.GetIdx()] - probe_pos[atom.GetIdx()]
                        v /= np.linalg.norm(v)
                        angles = rules[0]
                        bond_orders = rules[1]
                        theta = rules[2]
                        excluded_angs = rules[3]
                        for ang in angles:
                            angle = np.radians(ang)
                            for bond_order in bond_orders:
                                try:
                                    bond_len = bonds_df[atom_type + new_atom_type + bond_order]
                                except Exception:
                                    continue
                                bond_type = bond_types[bond_order]
                                point = probe_pos[atom.GetIdx()] + v * bond_len
                                edmol = Chem.EditableMol(probe)
                                idx11 = edmol.AddAtom(Chem.Atom(symbols[new_atom_type]))
                                edmol.AddBond(idx1, idx11, bond_type)
                                frag = edmol.GetMol()
                                name = f"{' '.join(probe_name)} {new_atom_type}"
                                frag.SetProp("name", name)
                                if not atom_counters_checker(frag, name, NO_limit, elem_dct, dct):
                                    frag.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                                    bad_out.write(frag)
                                    continue
                                if (
                                    output_mcf
                                    and not passes_all_mcf(
                                        frag,
                                        fragments_names=fragments_names,
                                        mcf=mcf_fragments,
                                        mcf_counts=fragments_counts,
                                        tolerance=0,
                                        return_all_unpassed=True,
                                    )["passed"]
                                ):
                                    frag.SetProp("reason_to_skip", "mcf")
                                    bad_out.write(frag)
                                    continue
                                next_task = task_initializer(mode, frag, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring)
                                frag.SetProp("current_task", next_task)
                                if probe.HasProp("dummy"):
                                    frag.SetProp("dummy", probe.GetProp("dummy"))
                                frag.GetConformer().SetAtomPosition(idx11, point + 0.0001)
                                neighbs = [a.GetIdx() for a in frag.GetAtomWithIdx(idx1).GetNeighbors()]
                                Chem.rdMolTransforms.SetAngleRad(frag.GetConformer(), neighbs[0], idx1, neighbs[-1], angle)
                                # here
                                try:
                                    Chem.SanitizeMol(frag)
                                    frag = ionize(frag)
                                except Exception:
                                    frag.SetProp("reason_to_skip", "can't sanitize/ionize")
                                    bad_out.write(frag)
                                    continue
                                atoms = [atom.GetIdx(), neighb.GetIdx()]
                                neighbor_2 = next(a.GetIdx() for a in neighb.GetNeighbors() if a.GetIdx() not in atoms)
                                for i in range(0, 361, theta):
                                    if i in excluded_angs or i - 180 in excluded_angs:
                                        continue
                                    Chem.rdMolTransforms.SetDihedralDeg(frag.GetConformer(), neighbor_2, atoms[1], atoms[0], idx11, i)
                                    copy_mol = Chem.Mol(frag)
                                    Chem.SanitizeMol(copy_mol)
                                    pool.append(copy_mol)
                    continue
                if current_task == "build_.=O_to_SO2":
                    synthetic = True
                    if atom.IsInRing():
                        # если сера в алифатическом цикле, предполагается, что у нее никак не могло быть кислорода с двойной связью, если есть старый - сносим, вешаем сразу 2 без проверки
                        # Кольцевая сера: строим O относительно плоскости двух кольцевых соседей.
                        # Важно: чистый прирост тяжелых атомов всегда +1.
                        #
                        # Случай 1: старого .=O нет -> добавляем 1 кислород.
                        # Случай 2: старый .=O есть -> удаляем его и ставим 2 кислорода в SO2-геометрии.
                        #            Чистый прирост: -1 + 2 = +1.

                        s_idx_old = atom.GetIdx()

                        ring_neighbs_old = [
                            a.GetIdx()
                            for a in atom.GetNeighbors()
                            if a.IsInRing()
                        ]

                        if len(ring_neighbs_old) != 2:
                            probe.SetProp(
                                "reason_to_skip",
                                f"ring sulfur has {len(ring_neighbs_old)} ring neighbors instead of 2"
                            )
                            bad_out.write(probe)
                            continue

                        n1_idx_old, n2_idx_old = ring_neighbs_old

                        old_o_idxs = [
                            a.GetIdx()
                            for a in atom.GetNeighbors()
                            if probe_name[a.GetIdx()] == ".=O"
                        ]

                        if len(old_o_idxs) > 1:
                            probe.SetProp(
                                "reason_to_skip",
                                f"ring sulfur already has {len(old_o_idxs)} .=O atoms"
                            )
                            bad_out.write(probe)
                            continue

                        # Если старый кислород есть, удаляем его.
                        # После удаления индексы сдвигаются, поэтому строим old_idx -> new_idx.
                        remove_set = set(old_o_idxs)

                        old_to_new = {}
                        new_probe_name = []

                        new_idx = 0
                        for old_idx, old_type in enumerate(probe_name):
                            if old_idx in remove_set:
                                continue
                            old_to_new[old_idx] = new_idx
                            new_probe_name.append(old_type)
                            new_idx += 1

                        s_idx = old_to_new[s_idx_old]
                        n1_idx = old_to_new[n1_idx_old]
                        n2_idx = old_to_new[n2_idx_old]

                        rw = Chem.RWMol(probe)
                        for old_o_idx in sorted(old_o_idxs, reverse=True):
                            rw.RemoveAtom(old_o_idx)

                        probe_base = rw.GetMol()

                        bond_len = bonds_df["SO2" + ".=O" + "DOUBLE"]
                        bond_type = bond_types["DOUBLE"]

                        # Считаем две идеальные позиции кислородов относительно плоскости n1-S-n2.
                        conf_base = probe_base.GetConformer()

                        s_p = conf_base.GetAtomPosition(s_idx)
                        n1_p = conf_base.GetAtomPosition(n1_idx)
                        n2_p = conf_base.GetAtomPosition(n2_idx)

                        s_pos = np.array([s_p.x, s_p.y, s_p.z], dtype=float)
                        n1_pos = np.array([n1_p.x, n1_p.y, n1_p.z], dtype=float)
                        n2_pos = np.array([n2_p.x, n2_p.y, n2_p.z], dtype=float)

                        v1 = n1_pos - s_pos
                        v2 = n2_pos - s_pos

                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)

                        if v1_norm < 1e-8 or v2_norm < 1e-8:
                            probe_base.SetProp("reason_to_skip", "bad ring sulfur neighbor coordinates")
                            bad_out.write(probe_base)
                            continue

                        u1 = v1 / v1_norm
                        u2 = v2 / v2_norm

                        normal = np.cross(u1, u2)
                        normal_norm = np.linalg.norm(normal)

                        if normal_norm < 1e-8:
                            probe_base.SetProp("reason_to_skip", "bad ring sulfur plane normal")
                            bad_out.write(probe_base)
                            continue

                        normal /= normal_norm

                        bisector = u1 + u2
                        bisector_norm = np.linalg.norm(bisector)

                        if bisector_norm < 1e-8:
                            probe_base.SetProp("reason_to_skip", "bad ring sulfur angle bisector")
                            bad_out.write(probe_base)
                            continue

                        bisector /= bisector_norm

                        # u1 + u2 направлен внутрь угла N1-S-N2.
                        # Для кислородов берем направление наружу от кольца.
                        outward = -bisector

                        nso_angle = np.radians(109.0)

                        cos_nsn = np.clip(np.dot(u1, u2), -1.0, 1.0)
                        nsn_angle = np.arccos(cos_nsn)

                        denom = np.cos(nsn_angle / 2.0)

                        if abs(denom) < 1e-8:
                            probe_base.SetProp("reason_to_skip", "bad denominator for SO2 geometry")
                            bad_out.write(probe_base)
                            continue

                        a_v = -np.cos(nso_angle) / denom
                        a_v = np.clip(a_v, 0.05, 0.95)

                        b = np.sqrt(max(0.0, 1.0 - a_v * a_v))

                        o1_vec = a_v * outward + b * normal
                        o2_vec = a_v * outward - b * normal

                        o1_vec /= np.linalg.norm(o1_vec)
                        o2_vec /= np.linalg.norm(o2_vec)

                        o1_pos = s_pos + bond_len * o1_vec
                        o2_pos = s_pos + bond_len * o2_vec

                        # ------------------------------------------------------------
                        # Случай 1: старого кислорода не было.
                        # Добавляем только 1 кислород.
                        # Чтобы не потерять возможную сторону относительно плоскости,
                        # создаем две альтернативы: O сверху и O снизу.
                        # Каждая молекула все равно содержит только один новый атом.
                        # ------------------------------------------------------------
                        if len(old_o_idxs) == 0:
                            for new_o_pos in [o1_pos, o2_pos]:
                                edmol = Chem.EditableMol(probe_base)

                                o_idx = edmol.AddAtom(Chem.Atom(symbols[".=O"]))
                                edmol.AddBond(s_idx, o_idx, bond_type)

                                frag = edmol.GetMol()

                                frag_name = new_probe_name + [".=O"]
                                name = " ".join(frag_name)
                                frag.SetProp("name", name)

                                frag.GetConformer().SetAtomPosition(o_idx, new_o_pos)

                                if not atom_counters_checker(frag, name, NO_limit, elem_dct, dct):
                                    frag.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                                    bad_out.write(frag)
                                    continue

                                if (
                                    output_mcf
                                    and not passes_all_mcf(
                                        frag,
                                        fragments_names=fragments_names,
                                        mcf=mcf_fragments,
                                        mcf_counts=fragments_counts,
                                        tolerance=0,
                                        return_all_unpassed=True,
                                    )["passed"]
                                ):
                                    frag.SetProp("reason_to_skip", "mcf")
                                    bad_out.write(frag)
                                    continue

                                try:
                                    Chem.SanitizeMol(frag)
                                    frag = ionize(frag)
                                except Exception:
                                    frag.SetProp("reason_to_skip", "can't sanitize/ionize")
                                    bad_out.write(frag)
                                    continue

                                next_task = task_initializer(
                                    mode,
                                    frag,
                                    forbidden_atoms,
                                    max_num_chiral_centers,
                                    Nac_outside_arom_ring,
                                )
                                frag.SetProp("current_task", next_task)

                                if probe.HasProp("dummy"):
                                    frag.SetProp("dummy", probe.GetProp("dummy"))

                                copy_mol = Chem.Mol(frag)
                                Chem.SanitizeMol(copy_mol)
                                pool.append(copy_mol)

                            continue

                        # ------------------------------------------------------------
                        # Случай 2: старый кислород был.
                        # Сносим старый O и ставим сразу два O в правильной SO2-геометрии.
                        # Но чистый прирост тяжелых атомов все равно +1.
                        # Поэтому итерационная логика не ломается.
                        # ------------------------------------------------------------
                        edmol = Chem.EditableMol(probe_base)

                        o1_idx = edmol.AddAtom(Chem.Atom(symbols[".=O"]))
                        edmol.AddBond(s_idx, o1_idx, bond_type)

                        o2_idx = edmol.AddAtom(Chem.Atom(symbols[".=O"]))
                        edmol.AddBond(s_idx, o2_idx, bond_type)

                        frag = edmol.GetMol()

                        frag_name = new_probe_name + [".=O", ".=O"]
                        name = " ".join(frag_name)
                        frag.SetProp("name", name)

                        conf = frag.GetConformer()
                        conf.SetAtomPosition(o1_idx, o1_pos)
                        conf.SetAtomPosition(o2_idx, o2_pos)

                        if not atom_counters_checker(frag, name, NO_limit, elem_dct, dct):
                            frag.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                            bad_out.write(frag)
                            continue

                        if (
                            output_mcf
                            and not passes_all_mcf(
                                frag,
                                fragments_names=fragments_names,
                                mcf=mcf_fragments,
                                mcf_counts=fragments_counts,
                                tolerance=0,
                                return_all_unpassed=True,
                            )["passed"]
                        ):
                            frag.SetProp("reason_to_skip", "mcf")
                            bad_out.write(frag)
                            continue

                        try:
                            Chem.SanitizeMol(frag)
                            frag = ionize(frag)
                        except Exception:
                            frag.SetProp("reason_to_skip", "can't sanitize/ionize")
                            bad_out.write(frag)
                            continue

                        next_task = task_initializer(
                            mode,
                            frag,
                            forbidden_atoms,
                            max_num_chiral_centers,
                            Nac_outside_arom_ring,
                        )
                        frag.SetProp("current_task", next_task)

                        if probe.HasProp("dummy"):
                            frag.SetProp("dummy", probe.GetProp("dummy"))

                        copy_mol = Chem.Mol(frag)
                        Chem.SanitizeMol(copy_mol)
                        pool.append(copy_mol)

                        continue
                    rules_dct = {  # atom_type: angles, bond_orders, theta to rotate, dihedral angs to exclude
                        ".=O": [[109], "DOUBLE", 15, []]
                    }
                    neighb = atom.GetNeighbors()[0]
                    angle = np.radians(109)
                    bond_len = bonds_df["SO2" + ".=O" + "DOUBLE"]
                    bond_type = bond_types["DOUBLE"]
                    v = probe_pos[neighb.GetIdx()] - probe_pos[atom.GetIdx()]
                    v /= np.linalg.norm(v)
                    v *= bond_len
                    point = probe_pos[atom.GetIdx()] + v
                    edmol = Chem.EditableMol(probe)
                    idx11 = edmol.AddAtom(Chem.Atom(symbols[".=O"]))
                    edmol.AddBond(idx1, idx11, bond_type)
                    frag = edmol.GetMol()
                    name = f"{' '.join(probe_name)} {'.=O'}"
                    frag.SetProp("name", name)
                    if not atom_counters_checker(frag, name, NO_limit, elem_dct, dct):
                        frag.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                        bad_out.write(frag)
                        continue
                    if (
                        output_mcf
                        and not passes_all_mcf(
                            frag,
                            fragments_names=fragments_names,
                            mcf=mcf_fragments,
                            mcf_counts=fragments_counts,
                            tolerance=0,
                            return_all_unpassed=True,
                        )["passed"]
                    ):
                        frag.SetProp("reason_to_skip", "mcf")
                        bad_out.write(frag)
                        continue
                    next_task = task_initializer(mode, frag, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring)
                    frag.SetProp("current_task", next_task)
                    if probe.HasProp("dummy"):
                        frag.SetProp("dummy", probe.GetProp("dummy"))
                    frag.GetConformer().SetAtomPosition(idx11, point + 0.0001)
                    neighbs = [a.GetIdx() for a in frag.GetAtomWithIdx(idx1).GetNeighbors()]
                    Chem.rdMolTransforms.SetAngleRad(frag.GetConformer(), neighbs[0], atom.GetIdx(), neighbs[-1], angle)
                    try:
                        Chem.SanitizeMol(frag)
                        frag = ionize(frag)
                    except Exception:
                        frag.SetProp("reason_to_skip", "can't sanitize/ionize")
                        bad_out.write(frag)
                        continue

                    if len(atom.GetNeighbors()) > 1:  # if one .=O present already in PROBE! Does NOT check frag. убрал заглушку "and (not atom.IsInRing())", так как кислороды на SO2 в цикле вешаются отдельным ифом
                        if len(atom.GetNeighbors()) != 2 :
                            msg = "len(atom.GetNeighbors()) != 2"
                            raise ValueError(msg)
                        atom1 = next(a.GetIdx() for a in atom.GetNeighbors() if name.split()[a.GetIdx()] == ".=O")
                        atom2 = atom.GetIdx()
                        atom3 = next(a.GetIdx() for a in atom.GetNeighbors() if name.split()[a.GetIdx()] != ".=O")
                        atom4 = next(a.GetIdx() for a in frag.GetAtomWithIdx(atom3).GetNeighbors() if a.GetIdx() != atom2)
                        dihedral = Chem.rdMolTransforms.GetDihedralDeg(probe.GetConformer(), atom4, atom3, atom2, atom1)
                        for i in [-120, 120]:
                            copy_mol = Chem.Mol(frag)
                            Chem.rdMolTransforms.SetDihedralDeg(copy_mol.GetConformer(), atom4, atom3, atom2, idx11, dihedral + i)
                            coords = copy_mol.GetConformer().GetPositions()[-1]
                            frag.GetConformer().SetAtomPosition(idx11, coords)
                            Chem.SanitizeMol(frag)
                            pool.append(Chem.Mol(frag))
                    else:
                        atoms = [atom.GetIdx(), neighb.GetIdx()]
                        neighbor_2 = next(a.GetIdx() for a in neighb.GetNeighbors() if a.GetIdx() not in atoms)
                        for i in range(0, 361, 15):
                            Chem.rdMolTransforms.SetDihedralDeg(frag.GetConformer(), neighbor_2, atoms[1], atoms[0], idx11, i)
                            copy_mol = Chem.Mol(frag)
                            Chem.SanitizeMol(copy_mol)
                            pool.append(copy_mol)
                    continue

                if current_task == "build_to_terminal_Csp":
                    synthetic = True
                    rules_dct = {  # atom_type: angles, bond_orders, theta to rotate, dihedral angs to exclude
                        "Car": [180, "SINGLE", 0, []],
                        "Cs3": [180, "SINGLE", 0, []],
                        "C3r": [180, "SINGLE", 0, []],
                        "Cs2": [180, "SINGLE", 0, []],
                        "C2r": [180, "SINGLE", 0, []],
                    }
                    for new_atom_type, rules in rules_dct.items():
                        neighb = atom.GetNeighbors()[0]
                        v = probe_pos[neighb.GetIdx()] - probe_pos[atom.GetIdx()]
                        v /= np.linalg.norm(v)
                        angle = np.radians(rules[0])
                        bond_order = rules[1]
                        try:
                            bond_len = bonds_df[atom_type + new_atom_type + bond_order]
                        except Exception:
                            continue
                        bond_type = bond_types[bond_order]
                        point = probe_pos[atom.GetIdx()] + v * bond_len
                        edmol = Chem.EditableMol(probe)
                        idx11 = edmol.AddAtom(Chem.Atom(symbols[new_atom_type]))
                        edmol.AddBond(idx1, idx11, bond_type)
                        frag = edmol.GetMol()
                        name = f"{' '.join(probe_name)} {new_atom_type}"
                        frag.SetProp("name", name)
                        if not atom_counters_checker(frag, name, NO_limit, elem_dct, dct):
                            frag.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                            bad_out.write(frag)
                            continue
                        if (
                            output_mcf
                            and not passes_all_mcf(
                                frag,
                                fragments_names=fragments_names,
                                mcf=mcf_fragments,
                                mcf_counts=fragments_counts,
                                tolerance=0,
                                return_all_unpassed=True,
                            )["passed"]
                        ):
                            frag.SetProp("reason_to_skip", "mcf")
                            bad_out.write(frag)
                            continue
                        next_task = task_initializer(mode, frag, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring)
                        frag.SetProp("current_task", next_task)
                        if probe.HasProp("dummy"):
                            frag.SetProp("dummy", probe.GetProp("dummy"))
                        frag.GetConformer().SetAtomPosition(idx11, point + 0.0001)
                        neighbs = [a.GetIdx() for a in frag.GetAtomWithIdx(idx1).GetNeighbors()]
                        Chem.rdMolTransforms.SetAngleRad(frag.GetConformer(), neighbs[0], idx1, neighbs[-1], angle)
                        # here
                        try:
                            Chem.SanitizeMol(frag)
                            frag = ionize(frag)
                        except Exception:
                            frag.SetProp("reason_to_skip", "can't sanitize/ionize")
                            bad_out.write(frag)
                            continue
                        atoms = [atom.GetIdx(), neighb.GetIdx()]
                        neighbor_2 = next(a.GetIdx() for a in neighb.GetNeighbors() if a.GetIdx() not in atoms)
                        copy_mol = Chem.Mol(frag)
                        Chem.SanitizeMol(copy_mol)
                        pool.append(copy_mol)
                    continue

                if current_task == "build_to_Csp":
                    synthetic = True
                    rules_dct = {  # atom_type: angles, bond_orders, theta to rotate, dihedral angs to exclude
                        "Csp": [180, "TRIPLE", 0, []],
                        "Nac": [180, "TRIPLE", 0, []],
                    }
                    for new_atom_type, rules in rules_dct.items():
                        neighb = atom.GetNeighbors()[0]
                        v = probe_pos[neighb.GetIdx()] - probe_pos[atom.GetIdx()]
                        v /= np.linalg.norm(v)
                        angle = np.radians(rules[0])
                        bond_order = rules[1]
                        try:
                            bond_len = bonds_df[atom_type + new_atom_type + bond_order]
                        except Exception:
                            continue
                        bond_type = bond_types[bond_order]
                        point = probe_pos[atom.GetIdx()] + v * bond_len
                        edmol = Chem.EditableMol(probe)
                        idx11 = edmol.AddAtom(Chem.Atom(symbols[new_atom_type]))
                        edmol.AddBond(idx1, idx11, bond_type)
                        frag = edmol.GetMol()
                        name = f"{' '.join(probe_name)} {new_atom_type}"
                        frag.SetProp("name", name)
                        if not atom_counters_checker(frag, name, NO_limit, elem_dct, dct):
                            frag.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                            bad_out.write(frag)
                            continue
                        if (
                            output_mcf
                            and not passes_all_mcf(
                                frag,
                                fragments_names=fragments_names,
                                mcf=mcf_fragments,
                                mcf_counts=fragments_counts,
                                tolerance=0,
                                return_all_unpassed=True,
                            )["passed"]
                        ):
                            frag.SetProp("reason_to_skip", "mcf")
                            bad_out.write(frag)
                            continue
                        next_task = task_initializer(mode, frag, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring)
                        frag.SetProp("current_task", next_task)
                        if probe.HasProp("dummy"):
                            frag.SetProp("dummy", probe.GetProp("dummy"))
                        frag.GetConformer().SetAtomPosition(idx11, point + 0.0001)
                        neighbs = [a.GetIdx() for a in frag.GetAtomWithIdx(idx1).GetNeighbors()]
                        Chem.rdMolTransforms.SetAngleRad(frag.GetConformer(), neighbs[0], idx1, neighbs[-1], angle)
                        # here
                        try:
                            Chem.SanitizeMol(frag)
                            frag = ionize(frag)
                        except Exception:
                            frag.SetProp("reason_to_skip", "can't sanitize/ionize")
                            bad_out.write(frag)
                            continue
                        atoms = [atom.GetIdx(), neighb.GetIdx()]
                        neighbor_2 = next(a.GetIdx() for a in neighb.GetNeighbors() if a.GetIdx() not in atoms)
                        copy_mol = Chem.Mol(frag)
                        Chem.SanitizeMol(copy_mol)
                        pool.append(copy_mol)
                    continue

                # natural part
                for neighb in atom.GetNeighbors():
                    idx2 = neighb.GetIdx()

                    # two atoms mapped - scen definition required to map 3rd
                    requests = {}
                    if scen1:
                        requests["1"] = {
                            "frags": scen1_fragments_allowed,
                            "atoms3": neighb.GetNeighbors(),
                            "4-atomic": fragments_scen1,
                            "request": f"{probe_name[idx2]} {probe_name[idx1]} ",
                            "fixed": [idx2, idx1],
                        }
                    if scen2:
                        requests["2"] = {
                            "frags": scen2_fragments_allowed,
                            "atoms3": atom.GetNeighbors(),
                            "4-atomic": fragments_scen2,
                            "request": f"{probe_name[idx1]} {probe_name[idx2]} ",
                            "fixed": [idx1, idx2],
                        }
                    for scen in requests:
                        for neighb2 in requests[scen]["atoms3"]:
                            idx3 = neighb2.GetIdx()
                            if idx3 == idx1:
                                continue
                            search_request = probe_name[idx3] + " " + requests[scen]["request"]
                            fixed_atoms1 = [idx3] + requests[scen]["fixed"]
                            for frag in requests[scen]["4-atomic"]:
                                try:
                                    frag_name = frag.GetProp("name")
                                    if search_request not in frag_name:
                                        continue
                                    atom_name = frag_name.split()[3]
                                    if atom_name not in requests[scen]["frags"]:
                                        continue
                                    if atom_name not in ring_types and (probe.HasProp('initiate_ring_bisect') or probe.HasProp('initiate_ring_110')):
                                        continue
                                    frag_indices = list(map(int, frag.GetProp("indices")[1:][:-1].split(", ")))
                                    new_atom_index = frag_indices[3]
                                    new_bond_type = frag.GetBondBetweenAtoms(new_atom_index, frag.GetAtoms()[new_atom_index].GetNeighbors()[0].GetIdx()).GetBondType()
                                    if atom.GetSymbol() != "S" and any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds()) and str(new_bond_type) == "DOUBLE":
                                        continue  # rule for 5-valent C
                                    if current_task in ["fix_Car", "build_DB_to_Cs2"] and str(new_bond_type) != "DOUBLE":
                                        continue  # only double bonds allowed to fix Car
                                    if atom_name == "Nd0" and str(new_bond_type) == "DOUBLE":
                                        continue  # double-bonded Nd0 are not allowed
                                    if atom_name == "Sul" and str(new_bond_type) == "DOUBLE":
                                        continue
                                    fixed_atoms2 = frag_indices[0:3]
                                    rms = AllChem.GetBestRMS(frag, probe, map=[list(zip(fixed_atoms2, fixed_atoms1))])
                                    if rms > 0.3:
                                        continue
                                    frag_copy = Chem.Mol(frag)
                                    edmol = Chem.EditableMol(probe)
                                    idx11 = edmol.AddAtom(Chem.Atom(frag.GetAtoms()[new_atom_index].GetAtomicNum()))
                                    edmol.AddBond(idx1, probe_len, new_bond_type)
                                    frag_edited = edmol.GetMol()
                                    name = f"{' '.join(probe_name)} {atom_name}"
                                    frag_edited.SetProp("name", name)
                                    if not atom_counters_checker(frag_edited, name, NO_limit, elem_dct, dct):
                                        frag_edited.SetProp("reason_to_skip", "some_atom_counter_exceeded")
                                        bad_out.write(frag_edited)
                                        continue
                                    rms_coord3 = frag_copy.GetConformer().GetPositions()[list(map(int, frag_copy.GetProp("indices")[1:][:-1].split(", ")))[3]]
                                    frag_edited.GetConformer().SetAtomPosition(idx11, rms_coord3)
                                    if not angles_are_ok(frag_edited, frag_edited.GetAtomWithIdx(idx11).GetNeighbors()[0]):
                                        frag_edited.SetProp("reason_to_skip", "bad angles")
                                        bad_out.write(frag_edited)
                                        continue
                                    if (
                                        atom_type == "Nac" and len(frag_edited.GetAtomWithIdx(idx1).GetNeighbors()) == 3
                                    ):  # when build 3rd neighbor to Nac it's important to remove H
                                        frag_edited.GetAtomWithIdx(idx1).SetNoImplicit(what=True)
                                        frag_edited.GetAtomWithIdx(idx1).SetNumExplicitHs(0)
                                    for s_atom in frag_edited.GetAtoms():
                                        if s_atom.GetSymbol() == "S":
                                            s_atom.SetNoImplicit(what=True)
                                            s_atom.SetNumExplicitHs(0)
                                    if probe.HasProp("dummy"):
                                        frag_edited.SetProp("dummy", probe.GetProp("dummy"))
                                    if not all_valencies_are_ok(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "valencies")
                                        bad_out.write(frag_edited)
                                        continue
                                    frag_edited.UpdatePropertyCache()
                                    frag_edited = make_cycle_where_possible(frag_edited)
                                    try:
                                        Chem.SanitizeMol(frag_edited)
                                        frag_edited = ionize(frag_edited)
                                    except Exception:
                                        frag_edited.SetProp("reason_to_skip", "can't sanitize/ionize")
                                        bad_out.write(frag_edited)
                                        continue
                                    for atom_frag1 in frag_edited.GetAtoms():
                                        atom_frag1.SetNumRadicalElectrons(0)
                                    if not valencies_after_cyclization_are_ok(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "valencies (cyclization)")
                                        bad_out.write(frag_edited)
                                        continue
                                    charges = [a.GetFormalCharge() for a in frag_edited.GetAtoms()]
                                    if (
                                        sum([c == 1 for c in charges]) > max_num_positive_charges
                                        or sum([c == -1 for c in charges]) > max_num_negative_charges
                                        or sum([c != 0 for c in charges]) > max_num_charged_atoms
                                    ):
                                        frag_edited.SetProp("reason_to_skip", "max num charges exceeded")
                                        bad_out.write(frag_edited)
                                        continue
                                    if sum(len(r) in [3, 4] for r in frag_edited.GetRingInfo().AtomRings()) > max_num_low_cycles:
                                        frag_edited.SetProp("reason_to_skip", "max num low cycles exceeded")
                                        bad_out.write(frag_edited)
                                        continue
                                    if current_task == "close_chiral_center" and count_chiral_centers(probe) > max_num_chiral_centers:
                                        frag_edited.SetProp("reason_to_skip", "cant close chiral center")
                                        bad_out.write(frag_edited)
                                        continue
                                    if current_task == "close_aromatic_ring" and incorrect_arom_ring_closing(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "incorrect arom ring closing")
                                        bad_out.write(frag_edited)
                                        continue
                                    if current_task == "close_aliphatic_ring" and incorrect_aliph_ring_closing(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "incorrect aliph ring closing")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not Nac_are_conjugated(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "Nac not conjugated")
                                        bad_out.write(frag_edited)
                                        continue
                                    if non_ring_atom_in_ring(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "non-ring atom in ring")
                                        bad_out.write(frag_edited)
                                        continue
                                    if Nd0_with_3_neighbors_present(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "Nd0 with 3 neighbors")
                                        bad_out.write(frag_edited)
                                        continue
                                    if mol_has_Cs2_with_3_neighbors_and_valence_3(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "3-valent Cs2")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not mol_has_optimal_acyclic_chains_length(frag_edited, max_len):
                                        frag_edited.SetProp("reason_to_skip", "excessive acyclic chain len")
                                        bad_out.write(frag_edited)
                                        continue
                                    if charged_N_with_arom_neighb(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "Nd+ with sp2 neighb")
                                        bad_out.write(frag_edited)
                                        continue
                                    if len(get_arom_Car_with_no_double_bonds(frag_edited)) > 2:
                                        frag_edited.SetProp("reason_to_skip", "too many Car with no DBs")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not Car_Nac_DB_dihedrals_are_ok(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "Car-Nac-DB dihedral")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not angles_for_SP2_atoms_with_3_neighbors_are_ok(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "bad angles for SP2 atoms")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not Car_DB_Car_dihedrals_are_ok(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "Car-DB-Car-Car dihedral")
                                        bad_out.write(frag_edited)
                                        continue
                                    if intramol_repulsion(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "intramol repulsion")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not all_needed_N_are_planar(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "needed N not planar")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not all_Cs2_are_planar(frag_edited, forbidden_atoms):
                                        frag_edited.SetProp("reason_to_skip", "Cs2 not planar")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not all_aromatic_rings_are_planar(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "arom rings not planar")
                                        bad_out.write(frag_edited)
                                        continue
                                    if mol_has_CNdSO2_with_bad_geometry(frag_edited, forbidden_atoms):
                                        frag_edited.SetProp("reason_to_skip", "bad SO2")
                                        bad_out.write(frag_edited)
                                        continue
                                    if mol_has_acyclic_cis_amide(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "cis-amide")
                                        bad_out.write(frag_edited)
                                        continue
                                    if mol_contains_macrocycles(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "macrocycles")
                                        bad_out.write(frag_edited)
                                        continue
                                    if gem_O_d_present(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "geminal O_d")
                                        bad_out.write(frag_edited)
                                        continue
                                    if two_O_a_in_one_ring(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "two O_a in ring")
                                        bad_out.write(frag_edited)
                                        continue
                                    if parallel_arom_DB(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "parallel arom DB")
                                        bad_out.write(frag_edited)
                                        continue
                                    if some_ring_substituted_by_more_than_2_Csp3_or_more_than_2_Nd0(frag_edited, forbidden_atoms):
                                        frag_edited.SetProp("reason_to_skip", "excessive ring substituents")
                                        bad_out.write(frag_edited)
                                        continue
                                    if (
                                        output_mcf
                                        and not passes_all_mcf(
                                            frag_edited,
                                            fragments_names=fragments_names,
                                            mcf=mcf_fragments,
                                            mcf_counts=fragments_counts,
                                            tolerance=0,
                                            return_all_unpassed=True,
                                        )["passed"]
                                    ):
                                        frag_edited.SetProp("reason_to_skip", "mcf")
                                        bad_out.write(frag_edited)
                                        continue
                                    if allow_chair_only and not aliphatic_rings_are_ok(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "non-chair aliph ring")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not allow_axial_groups and axial_groups_present(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "axial ring substituent")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not allow_paired_Nd0 and paired_Nd0_present(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "paired Nd0")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not allow_Nd0_Nac_pair and pair_Nd0_Nac_present(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "pair Nd0 Nac")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not allow_condensed_alicyclics and condensed_aliphatic_rings_present(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "condensed alicyclics")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not allow_double_bonded_arom_rings and double_bonded_arom_rings(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "DB between aromatics")
                                        bad_out.write(frag_edited)
                                        continue
                                    if double_bonded_Cs2_and_Car(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "DB between Cs2 and Car")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not allow_alkenes and alkene_present(frag_edited):
                                        frag_edited.SetProp("reason_to_skip", "alkene present")
                                        bad_out.write(frag_edited)
                                        continue
                                    if not all_dihedrals_of_new_atom_are_ok(frag_edited, dihedrals_df):
                                        frag_edited.SetProp("reason_to_skip", "bad dihedral")
                                        bad_out.write(frag_edited)
                                        continue
                                    if frag_edited.HasProp('build_orthosubstituent_to_rotated_amide'):
                                        frag_edited.SetProp('flag ortho', '')
                                        frag_edited.ClearProp('build_orthosubstituent_to_rotated_amide')
                                    next_task = task_initializer(mode, frag_edited, forbidden_atoms, max_num_chiral_centers, Nac_outside_arom_ring)
                                    frag_edited.SetProp("current_task", next_task)
                                    compare.append(frag_edited)
                                    pool.append(frag_edited)
                                except Exception as e:
                                    log.write(str(e) + "\n")

            if ref_match_debug:
                save_ref_matches(
                    pool,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="generated_pool_before_intramol_clashes",
                    scenario=scenario,
                    current_task=current_task,
                )
            filtered = []
            for frag in pool:
                if intramolecular_clashes_building(frag, frag.GetProp("current_task")):
                    frag.SetProp("reason_to_skip", "intramol clashes")
                    bad_out.write(frag)
                    continue
                filtered.append(frag)

            if ref_match_debug:
                save_ref_matches(
                    filtered,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="after_intramol_clashes",
                    scenario=scenario,
                    current_task=current_task,
                )

            if not synthetic:
                pool = duplicates_check_local(filtered)
                # fix problem, when duplicates_check_local ломает геометрию колец из-за усреднения позиции последнего атома
                # pool_checked = []

                # for mol_pool in pool:
                #     if not all_aromatic_rings_are_planar(mol_pool):
                #         mol_pool.SetProp("reason_to_skip", "arom rings not planar after duplicates")
                #         bad_out.write(mol_pool)
                #         continue

                #     if not all_needed_N_are_planar(mol_pool):
                #         mol_pool.SetProp("reason_to_skip", "needed N not planar after duplicates")
                #         bad_out.write(mol_pool)
                #         continue

                #     if not all_Cs2_are_planar(mol_pool, forbidden_atoms):
                #         mol_pool.SetProp("reason_to_skip", "Cs2 not planar after duplicates")
                #         bad_out.write(mol_pool)
                #         continue

                #     pool_checked.append(mol_pool)

                # pool = pool_checked

            else:
                pool = filtered # fix problem, when intramol clashes after synthetic mode are saved to scored and selected

            if ref_match_debug:
                save_ref_matches(
                    pool,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="after_duplicates",
                    scenario=scenario,
                    current_task=current_task,
                )

            for mol in pool:
                if reference and not mol_subgraph_match(reference, mol):
                    mol.SetProp("reason_to_skip", "subgraph mismatch")
                    bad_out.write(mol)
                    continue

                out.write(mol)
                written += 1

            if not written:
                if debug and iter_n != 1:
                    break
                if a[iter_n] < len(Chem.SDMolSupplier(str(path_to_working_dir / f"iter{iter_n - 1}_selected.sdf"))):
                    a[iter_n] += 1
                #для тестирования
                bad_out.close()
                with open(path_to_working_dir / f"iter{iter_n}_broken.sdf", "r") as f_in, open(all_file, "a") as f_out:
                    f_out.write(f_in.read())
                #для тестирования
                continue
            out.close()
            if verbose:
                print(f"{written} mols were written")
                print(f"Generation finished in {round((time.time() - gen_start) * 1000)} ms")

                # minimization
                print("Minimization started")
            if ref_match_debug:
                generated_written = [
                    m for m in Chem.SDMolSupplier(
                        str(path_to_working_dir / f"iter{iter_n}_generated.sdf"),
                        removeHs=False,
                    )
                    if m is not None
                ]

                save_ref_matches(
                    generated_written,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="generated_written",
                    scenario=scenario,
                    current_task=current_task,
                )
            min_start = time.time()
            log.write(f"{written} mols were written\n")
            log.write(f"Generation finished in {round((time.time() - gen_start) * 1000)} ms\n")
            log.write("Minimization started\n")
            written = 0
            with Chem.SDWriter(str(path_to_working_dir / f"iter{iter_n}_minimized.sdf")) as out:
                list_for_multiprocessing = []
                probes = list(Chem.SDMolSupplier(str(path_to_working_dir / f"iter{iter_n}_generated.sdf")))
                for probe in probes:
                    prop_dict = probe.GetPropsAsDict()

                    list_for_multiprocessing.append(
                        (
                            probe,
                            prop_dict,
                            (path_to_working_dir / "cropped_protein.pdbqt").read_text(),
                            gridbox_center_x,
                            gridbox_center_y,
                            gridbox_center_z,
                            skip_minimization,
                        )
                    )
                min_probes_list = multiprocpool.starmap(probe_minimizer, list_for_multiprocessing)

                min_probes_list_filtered = []
                for m, initial in zip(min_probes_list, probes):
                    if m and m[0]:
                        if isinstance(m[0], list):
                            min_probes_list_filtered.append(tuple([m[0][0]] + list(m[1:])))
                            min_probes_list_filtered.append(tuple([m[0][1]] + list(m[1:])))
                        else:
                            min_probes_list_filtered.append(m)
                    else:
                        initial.SetProp("reason_to_skip", "minimization failed")
                        bad_out.write(initial)
                for min_probe, prop_dict in min_probes_list_filtered:
                    for i in prop_dict.keys():
                        min_probe.SetProp(i, str(prop_dict[i]))

                    out.write(min_probe)
                    written += 1

            if not written:
                if debug and iter_n != 1:
                    break
                if a[iter_n] < len(Chem.SDMolSupplier(f"{path_to_working_dir}/iter{iter_n - 1}_selected.sdf")):
                    a[iter_n] += 1
                continue
            if verbose:
                print(f"{written} mols were written")
                print(f"Minimization finished in {round((time.time() - min_start) * 1000)} ms")

                # filtration
                print("Filtration started")

            if ref_match_debug:
                minimized_mols = [
                    m for m in Chem.SDMolSupplier(
                        str(path_to_working_dir / f"iter{iter_n}_minimized.sdf"),
                        removeHs=False,
                    )
                    if m is not None
                ]

                save_ref_matches(
                    minimized_mols,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="minimized",
                    scenario=scenario,
                    current_task=current_task,
                )
            filt_start = time.time()
            log.write(f"{written} mols were written\n")
            log.write(f"Minimization finished in {round((time.time() - min_start) * 1000)} ms\n")
            log.write("Filtration started\n")
            written = 0
            with Chem.SDWriter(f"{path_to_working_dir}/iter{iter_n}_filtered.sdf") as out:
                for probe in Chem.SDMolSupplier(f"{path_to_working_dir}/iter{iter_n}_minimized.sdf"):
                    if lig_prot_repulsion(probe, protein_df, forbidden_atoms):
                        probe.SetProp("reason_to_skip", "repulsion after minimization")
                        bad_out.write(probe)
                        continue
                    if critical_clashes(probe, protein_df, probe.GetProp("current_task")):
                        probe.SetProp("reason_to_skip", "critical clashes")
                        bad_out.write(probe)
                        continue
                    clashes = general_utils.calc_probe_clashes(Chem.AddHs(probe, addCoords=True), protein_df)
                    if clashes > max_clashes:
                        probe.SetProp("reason_to_skip", f"too many clashes ({clashes} with max {max_clashes})")
                        bad_out.write(probe)
                        continue
                    if not ligand_is_within_grid(probe.GetConformer().GetPositions(), grid_coords):
                        probe.SetProp("reason_to_skip", "lig outside grid")
                        bad_out.write(probe)
                        continue
                    if check_crucial_interaction and not probe_has_crucial_interaction(probe, crucial_coords, ligand_role):
                        probe.SetProp("reason_to_skip", "missing crucial interaction")
                        bad_out.write(probe)
                        continue
                    probe.SetProp("prot_clashes", str(clashes))
                    out.write(probe)
                    written += 1
            bad_out.close()

            #для тестирования
            with open(path_to_working_dir / f"iter{iter_n}_broken.sdf", "r") as f_in, open(all_file, "a") as f_out:
                f_out.write(f_in.read())
            if not written:
                if debug and iter_n != 1:
                    break
                if a[iter_n] < len(Chem.SDMolSupplier(f"{path_to_working_dir}/iter{iter_n - 1}_selected.sdf")):
                    a[iter_n] += 1
                continue
            if verbose:
                print(f"{written} mols were written")
                print(f"Filtration finished in {round((time.time() - filt_start) * 1000)} ms")


                # scoring
                print("Scoring started")

            if ref_match_debug:
                filtered_mols = [
                    m for m in Chem.SDMolSupplier(
                        str(path_to_working_dir / f"iter{iter_n}_filtered.sdf"),
                        removeHs=False,
                    )
                    if m is not None
                ]

                save_ref_matches(
                    filtered_mols,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="protein_filtered",
                    scenario=scenario,
                    current_task=current_task,
                )
            scor_start = time.time()
            log.write(f"{written} mols were written\n")
            log.write(f"Filtration finished in {round((time.time() - filt_start) * 1000)} ms\n")
            log.write("Scoring started\n")
            written = 0
            probes = list(Chem.SDMolSupplier(f"{path_to_working_dir}/iter{iter_n}_filtered.sdf"))
            atom_types = []
            coords_list = []
            for probe in probes:
                atom_types.extend(probe.GetProp("name").replace("C3r", "Cs3").replace("C2r", "Cs2").split())
                coords_list.extend(list(probe.GetConformer().GetPositions()))
            probe_len = len(atom_types) // len(probes)
            related_probes = pd.Series(np.array([[i] * probe_len for i in range(len(probes))]).flatten())
            [i * probe_len for i in range(len(probes))]
            sample_graph, elem_data, indices = data_preparation_for_dot_prediction(prot, coords_list, atom_types)
            final_scores, intermediate_scores = dot_prediction(
                sample_graph,
                elem_data,
                list_of_neurals=loaded_neurals,
                local_device=device,
                method=method,
                dot_pred_batch_size=dot_pred_batch_size,
            )
            phar_scores = dot_prediction_pharma(
                sample_graph,
                elem_data,
                list_of_neurals=loaded_pharma_neurals,
                local_device=device,
                dot_pred_batch_size=dot_pred_batch_size,
            )
            scores_df = pd.DataFrame(
                index=indices,
                data={
                    "final_score": final_scores,
                    "intermediate_score": intermediate_scores,
                    "phar_score": phar_scores,
                    "atom_type": elem_data,
                    "related_probe": related_probes,
                },
            )
            for i, ppdf in scores_df.groupby("related_probe"):
                mol_score = (ppdf.intermediate_score).sum() / probe_len  # change the divisor!!!!!!!!!!!!!!!!
                mol_phar_score = 0
                if phar_model == "MedChem":
                    for _, line in ppdf.iterrows():
                        atom_type = line.atom_type
                        phar_score = line.phar_score
                        if phar_score >= dict_of_thresholds[atom_type]:
                            mol_phar_score += dict_of_coefs[atom_type]
                    mol_phar_score /= 10
                elif phar_model == "CatBoost":
                    phar_df = pd.DataFrame(
                        data=np.zeros((1, 26)),
                        columns=[
                            "Cs3",
                            "Nd0",
                            "Car",
                            "Nac",
                            "Csp",
                            "O_d",
                            "Sul",
                            "O_a",
                            ".=O",
                            "SO2",
                            "Cs2",
                            "Nd+",
                            "Hal",
                            "Cs3prob",
                            "Nd0prob",
                            "Carprob",
                            "Nacprob",
                            "Cspprob",
                            "O_dprob",
                            "Sulprob",
                            "O_aprob",
                            ".=Oprob",
                            "SO2prob",
                            "Cs2prob",
                            "Nd+prob",
                            "Halprob",
                        ],
                    )
                    for _, line in ppdf.iterrows():
                        atom_type = line.atom_type
                        phar_score = line.phar_score
                        phar_df[atom_type + "prob"] += phar_score
                        if phar_score >= dict_of_thresholds[atom_type]:
                            phar_df[atom_type] += 1
                    mol_phar_score = (cat.predict(phar_df)[0] - 6) / 6
                probes[i].SetProp("NBS phar", str(mol_phar_score))
                probes[i].SetProp("NBS_non-phar", str(mol_score))
                probes[i].SetProp("NBS", str(mol_phar_score + mol_score))
            probes = sorted(probes, key=lambda x: float(x.GetProp("NBS")), reverse=True)
            with Chem.SDWriter(f"{path_to_working_dir}/iter{iter_n}_scored.sdf") as out:
                for scored_rank, probe in enumerate(probes):
                    # метаданные для последующего поиска в all_scored.sdf
                    probe.SetProp("source_stage", "scored")
                    probe.SetProp("source_iter", str(iter_n))
                    probe.SetProp("source_parent_mol", str(a[iter_n]))
                    probe.SetProp("source_scenario", scenario)
                    probe.SetProp("source_rank_in_scored_call", str(scored_rank))
                    probe.SetProp("source_current_task", current_task)

                    out.write(probe)
                    written += 1

                    # branch for yielding output
                    if probe.GetProp("current_task") == "no_task" and probe.HasProp("NBS") and rdMolDescriptors.CalcNumRings(probe) > 0 and probe_len >= min_mol_size:
                        if verbose:
                            print("Trying to yield output")
                        log.write("Trying to yield output\n")
                        if not intramolecular_clashes(Chem.AddHs(probe, addCoords=True)):
                            probe_copy = Chem.AddHs(probe, addCoords=True)
                            Chem.SanitizeMol(probe_copy)
                            if output_mcf:
                                result = passes_all_mcf(
                                    probe_copy,
                                    fragments_names=final_fragments_names,
                                    mcf=final_mcf_fragments,
                                    mcf_counts=final_fragments_counts,
                                    tolerance=0,
                                    return_all_unpassed=True,
                                )
                            if not output_mcf or result["passed"]:
                                #оставил структуру функции как в probe_minimizer(), так как может потребоваться параллелизация
                                prop_dict = probe.GetPropsAsDict()
                                docked_de_novo_probe, prop_dict = probe_docker_denovo(probe_copy,
                                                                                      prop_dict,
                                                                                      (path_to_working_dir / "cropped_protein.pdbqt").read_text(),
                                                                                      gridbox_center_x,
                                                                                      gridbox_center_y,
                                                                                      gridbox_center_z,
                                                                                      )
                                if docked_de_novo_probe:
                                    for i in prop_dict.keys():
                                        docked_de_novo_probe.SetProp(i, str(prop_dict[i]))
                                    if verbose:
                                        print("Yielding output successful")
                                    log.write("Yielding output successful\n")

                                    yield docked_de_novo_probe
                                    if super_output:
                                        save_mols_for_super_output(path_to_working_dir, docked_de_novo_probe)
                                else:
                                    if verbose:
                                        print("Yielding canceled by bad redocking")
                                    log.write("Yielding canceled by bad redocking\n")
                            else:
                                if verbose:
                                    print("Yielding canceled by output mcf")
                                log.write("Yielding canceled by output mcf\n")
                        else:
                            if verbose:
                                print("Yielding canceled due to intramolecular clashes")
                            log.write("Yielding canceled due to intramolecular clashes\n")

            # для тестирования: сохраняем все scored-молекулы по всем веткам
            with open(path_to_working_dir / f"iter{iter_n}_scored.sdf", "r") as f_in, open(all_scored_out, "a") as f_out:
                f_out.write(f_in.read())

            torch.cuda.empty_cache()
            if not written:
                if debug and iter_n != 1:
                    break
                if a[iter_n] < len(Chem.SDMolSupplier(f"{path_to_working_dir}/iter{iter_n - 1}_selected.sdf")):
                    a[iter_n] += 1
                continue
            if verbose:
                print(f"{written} mols were written")
                print(f"Scoring finished in {round((time.time() - scor_start) * 1000)} ms")

                # selection
                print("Selection started")

            if ref_match_debug:
                scored_mols = [
                    m for m in Chem.SDMolSupplier(
                        str(path_to_working_dir / f"iter{iter_n}_scored.sdf"),
                        removeHs=False,
                    )
                    if m is not None
                ]

                save_ref_matches(
                    scored_mols,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="scored",
                    scenario=scenario,
                    current_task=current_task,
                )

            sel_start = time.time()
            log.write(f"{written} mols were written\n")
            log.write(f"Scoring finished in {round((time.time() - scor_start) * 1000)} ms\n")
            log.write("Selection started\n")
            written = 0
            mols = list(Chem.SDMolSupplier(f"{path_to_working_dir}/iter{iter_n}_scored.sdf"))
            # select poses with with NBS at least 50% of max
            best_score = max(float(m.GetProp("NBS")) for m in mols)
            selected_1 = [m for m in mols if float(m.GetProp("NBS")) > 0.5 * best_score]
            # select only 5 poses of each last added atom type + bond type
            if ref_match_debug:
                save_ref_matches(
                    selected_1,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="selected_1_score_threshold",
                    scenario=scenario,
                    current_task=current_task,
                )
            # with Chem.SDWriter(f"{path_to_working_dir}/iter{iter_n}_selected_1.sdf") as out_1:
            #     for mol in selected_1:
            #         mol.SetProp("scenario", scenario)
            #         if current_task == "close_aromatic_ring" and mol.GetProp("name").endswith(".=O"):
            #             mol.SetProp("tasks", tasks)
            #         else:
            #             curr = tasks + " " + current_task
            #             mol.SetProp("tasks", curr)
            #         out_1.write(mol)

            selected_2 = []
            types_counter, subgroups_counter = 0, 0
            for atom_type in GLOBAL_TARGET_NAMES_RING:
                group = sorted(
                    [m for m in selected_1 if m.GetProp("name").split()[-1] == atom_type],
                    key=lambda x: float(x.GetProp("NBS")),
                    reverse=True,
                )
                if not group:
                    continue
                types_counter += 1
                group = filter_similar_dihedrals(group, atom_type)
                groups_dct = {}
                for i in range(len(group)):
                    if i not in sum(groups_dct.values(), []):
                        for j in range(len(group)):
                            if group[j].HasSubstructMatch(group[i]):
                                if i in groups_dct:
                                    groups_dct[i].append(j)
                                else:
                                    groups_dct[i] = [j]
                for val in groups_dct.values():
                    subgroups_counter += 1
                    selected_2.extend([group[i] for i in val][:5])
            # select only certain number of poses left
            if ref_match_debug:
                save_ref_matches(
                    selected_2,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="selected_2_grouped",
                    scenario=scenario,
                    current_task=current_task,
                )
            # with Chem.SDWriter(f"{path_to_working_dir}/iter{iter_n}_selected_2.sdf") as out_2:
            #     for mol in selected_2:
            #         mol.SetProp("scenario", scenario)
            #         if current_task == "close_aromatic_ring" and mol.GetProp("name").endswith(".=O"):
            #             mol.SetProp("tasks", tasks)
            #         else:
            #             curr = tasks + " " + current_task
            #             mol.SetProp("tasks", curr)
            #         out_2.write(mol)

            if no_task_preference:
                selected_no_task  =  sorted([mol for mol in selected_2 if mol.GetProp('current_task') == 'no_task'], key=lambda x: float(x.GetProp("NBS")), reverse=True)
                selected_task = sorted([mol for mol in selected_2 if mol.GetProp('current_task') != 'no_task'], key=lambda x: float(x.GetProp("NBS")), reverse=True)

                selected_no_task_fin = selected_no_task[:min(len(selected_no_task), number_of_poses // 2+1)]
                selected_candidates = selected_no_task_fin + selected_task
            else:
                selected_candidates = sorted(selected_2, key=lambda x: float(x.GetProp("NBS")), reverse=True)

            # Global duplicate protection before saving the final selected beam.
            # The check compares current candidates with all molecules previously
            # written to all_selected.sdf: SMILES -> Counter(types) -> best
            # type-matched RMSD without alignment -> persistent branch geometry history.
            selected_3_raw = selected_candidates[:number_of_poses]
            selected_3, selected_3_global_duplicates = filter_selected_against_global_history(
                selected_candidates,
                all_out,
                number_of_poses,
                log=log,
                verbose=verbose,
            )
            if verbose:
                print(f"Selection stage 1: {len(selected_1)} mols selected")
                print(f"Selection stage 2: {len(selected_2)} mols selected")
                print(f"\t{subgroups_counter} groups of {types_counter} atom types distinguished")
                print(f"Selection stage 3 raw: {len(selected_3_raw)} mols selected")
                print(f"Selection stage 3 after global duplicate filter: {len(selected_3)} mols selected")
            log.write(f"Selection stage 1: {len(selected_1)} mols selected\n")
            log.write(f"Selection stage 2: {len(selected_2)} mols selected\n")
            log.write(f"\t{subgroups_counter} groups of {types_counter} atom types distinguished\n")
            log.write(f"Selection stage 3 raw: {len(selected_3_raw)} mols selected\n")
            log.write(f"Selection stage 3 after global duplicate filter: {len(selected_3)} mols selected\n")

            with Chem.SDWriter(f"{path_to_working_dir}/iter{iter_n}_selected.sdf") as out:
                for mol in selected_3:
                    mol.SetProp("scenario", scenario)
                    if current_task == "close_aromatic_ring" and mol.GetProp("name").endswith(".=O"):
                        mol.SetProp("tasks", tasks)
                    else:
                        curr = tasks + " " + current_task
                        if mol.HasProp("dummy") and mol.GetRingInfo().AtomRings():
                            if verbose:
                                print("Ring closed, next locked ring atom unlocked")
                            log.write("Ring closed, next locked ring atom unlocked\n")
                            mol.ClearProp("dummy")
                            mol.ClearProp("current_task")
                            curr += " ring_built"
                        mol.SetProp("tasks", curr)
                    out.write(mol)
                    written += 1

            if ref_match_debug:
                save_ref_matches(
                    selected_3,
                    ref=ref_debug_reference,
                    debug_dir=ref_debug_dir,
                    counters=ref_debug_counters,
                    iter_n=iter_n,
                    stage="selected_3_final_beam",
                    scenario=scenario,
                    current_task=current_task,
                )
            #для тестирования
            with open(path_to_working_dir / f"iter{iter_n}_selected.sdf", "r") as f_in, open(all_out, "a") as f_out:
                f_out.write(f_in.read())

            if not written:
                if debug and iter_n != 1:
                    break
                if a[iter_n] < len(Chem.SDMolSupplier(f"{path_to_working_dir}/iter{iter_n - 1}_selected.sdf")):
                    a[iter_n] += 1
                continue
            iter_n += 1
            if verbose:
                print(f"{written} mols were written")
                print(f"Selection finished in {round((time.time() - sel_start) * 1000)} ms")
            log.write(f"{written} mols were written\n")
            log.write(f"Selection finished in {round((time.time() - sel_start) * 1000)} ms\n")
            if debug and iter_n == max(a.keys()) + 1:
                break
        if verbose:
            print("Calculations were finished")
        log.write("Calculations were finished\n")
