import os
import json
import logging
import numpy as np
from typing import Tuple
from scipy.ndimage import binary_erosion, distance_transform_edt
from ..Utils.DataStructures.StatisticsStructure import CompartmentStatistics, CompositionStatistics


def get_compositions_folder() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Compositions')


def get_available_compositions() -> list:
    names = []
    for _, _, files in os.walk(get_compositions_folder()):
        for f in files:
            if f.endswith('.json'):
                names.append(f.split('.')[0])
        break
    return sorted(names)


def load_composition(composition) -> dict:
    """
    Composition schema telling which structures are the compartments of a lesion and which relations to compute
    between them, either given as a dict or as the name of a schema in raidionicsrads/Compositions (e.g.,
    enhancing_tumor). The optional keys are filled with empty values, and the roles used in the derived compartments,
    ratios, rim and disjoint pairs must exist in the compartments or references.
    """
    schema = None
    if isinstance(composition, dict):
        schema = composition
    else:
        composition_filename = os.path.join(get_compositions_folder(), str(composition) + '.json')
        if not os.path.exists(composition_filename):
            raise ValueError(f"[Composition] Unknown composition {composition}, available ones: {get_available_compositions()}.")
        with open(composition_filename, 'r') as infile:
            schema = json.load(infile)

    for key in ["name", "version", "base_sequence", "compartments"]:
        if key not in schema.keys():
            raise ValueError(f"[Composition] Missing key {key} in the composition schema.")
    for key in ["references", "derived", "ratios", "rim"]:
        if key not in schema.keys():
            schema[key] = {}
    if "disjoint" not in schema.keys():
        schema["disjoint"] = []

    roles = list(schema["compartments"].keys()) + list(schema["references"].keys())
    used_roles = []
    for role in schema["derived"].keys():
        used_roles.append(role)
        for part in schema["derived"][role]:
            used_roles.append(part)
    for name in schema["ratios"].keys():
        if len(schema["ratios"][name]) != 2:
            raise ValueError(f"[Composition] The ratio {name} must be given as [numerator role, denominator role].")
        for role in schema["ratios"][name]:
            used_roles.append(role)
    if len(schema["rim"]) != 0:
        for key in ["outer", "inner"]:
            if key not in schema["rim"].keys():
                raise ValueError(f"[Composition] The rim must give both an outer and an inner role.")
            used_roles.append(schema["rim"][key])
    for pair in schema["disjoint"]:
        for role in pair:
            used_roles.append(role)
    for role in used_roles:
        if role not in roles:
            raise ValueError(f"[Composition] Role {role} used in the composition {schema['name']} but not declared in "
                             f"its compartments or references.")
    return schema


def compute_rim_thickness(outer: np.ndarray, inner: np.ndarray, spacing: Tuple) -> dict:
    """
    Thickness (mm) of the outer structure around the inner one (e.g., the enhancing rim around the necrosis), as the
    distance from each voxel of the outer boundary to the inner structure. A minimum of 0 means that the inner
    structure reaches the outer boundary (open rim).
    Returns {"mean_mm", "min_mm", "max_mm", "std_mm"}, empty if the inner structure is empty.
    """
    rim = {}
    if np.count_nonzero(inner) == 0 or np.count_nonzero(outer) == 0:
        return rim
    try:
        outer_surface = np.logical_and(outer, np.logical_not(binary_erosion(outer)))
        distances = distance_transform_edt(np.logical_not(inner), sampling=np.asarray(spacing[0:3], dtype=float))
        surface_distances = distances[outer_surface]
        rim["mean_mm"] = float(np.mean(surface_distances))
        rim["min_mm"] = float(np.min(surface_distances))
        rim["max_mm"] = float(np.max(surface_distances))
        rim["std_mm"] = float(np.std(surface_distances))
    except Exception as e:
        raise ValueError(f"Rim thickness computation failed with: {e}")
    return rim


def compute_composition(masks: dict, spacing: Tuple, composition: dict, mask_filepaths: dict = None,
                        statistics_filenames: dict = None) -> CompositionStatistics:
    """
    Relations between the compartments of a lesion at one timestamp, following the composition schema (see
    load_composition).
    masks: {role: array}, for each compartment or reference having a mask, all on the same voxel grid.
    mask_filepaths and statistics_filenames: {role: filename}, kept in the results for traceability.
    The volumes are computed from the masks as given, so that all the ratios come from the same grid (the features
    computation cleans the masks a bit before measuring, its volumes can slightly differ).
    """
    if mask_filepaths is None:
        mask_filepaths = {}
    if statistics_filenames is None:
        statistics_filenames = {}
    statistics = CompositionStatistics(composition_name=composition["name"], composition_version=composition["version"])
    try:
        voxel_volume_ml = float(np.prod(np.asarray(spacing[0:3], dtype=float))) * 1e-3
        role_masks = {}
        for role in masks.keys():
            role_masks[role] = masks[role] > 0

        """1. Compartments and references having their own mask"""
        for role in composition["compartments"].keys():
            if role in role_masks.keys():
                statistics.compartments[role] = CompartmentStatistics(role=role, structure=composition["compartments"][role],
                                                                      volume=np.count_nonzero(role_masks[role]) * voxel_volume_ml,
                                                                      mask_filepath=mask_filepaths.get(role),
                                                                      statistics_filename=statistics_filenames.get(role))
        for role in composition["references"].keys():
            if role in role_masks.keys():
                statistics.references[role] = CompartmentStatistics(role=role, structure=composition["references"][role],
                                                                    volume=np.count_nonzero(role_masks[role]) * voxel_volume_ml,
                                                                    mask_filepath=mask_filepaths.get(role),
                                                                    statistics_filename=statistics_filenames.get(role))

        """2. Compartments derived from others when they have no mask of their own"""
        for role in composition["derived"].keys():
            if role in role_masks.keys():
                continue
            available_parts = []
            for part in composition["derived"][role]:
                if part in role_masks.keys():
                    available_parts.append(part)
            if len(available_parts) == 0:
                statistics.notes.append(f"{role} not derived, none of {composition['derived'][role]} available")
                continue
            union = np.zeros(role_masks[available_parts[0]].shape, dtype=bool)
            for part in available_parts:
                union = np.logical_or(union, role_masks[part])
            role_masks[role] = union
            if len(available_parts) < len(composition["derived"][role]):
                statistics.notes.append(f"{role} derived from {available_parts} only, out of {composition['derived'][role]}")
            statistics.compartments[role] = CompartmentStatistics(role=role, structure=composition["compartments"][role],
                                                                  volume=np.count_nonzero(union) * voxel_volume_ml,
                                                                  derived_from=available_parts)

        """3. Ratios between volumes"""
        volumes = {}
        for role in statistics.compartments.keys():
            volumes[role] = statistics.compartments[role].volume
        for role in statistics.references.keys():
            volumes[role] = statistics.references[role].volume
        for name in composition["ratios"].keys():
            numerator, denominator = composition["ratios"][name]
            if numerator not in volumes.keys() or denominator not in volumes.keys():
                statistics.notes.append(f"{name} not computed, {numerator} or {denominator} not available")
                continue
            if volumes[denominator] <= 0.:
                statistics.notes.append(f"{name} not computed, {denominator} is empty")
                continue
            statistics.ratios[name] = volumes[numerator] / volumes[denominator]

        """4. Rim thickness of the outer compartment around the inner one"""
        if len(composition["rim"]) != 0:
            outer = composition["rim"]["outer"]
            inner = composition["rim"]["inner"]
            if outer not in role_masks.keys() or inner not in role_masks.keys():
                statistics.notes.append(f"rim not computed, {outer} or {inner} not available")
            elif np.count_nonzero(role_masks[inner]) == 0:
                statistics.notes.append(f"rim not computed, {inner} is empty (solid lesion)")
            else:
                statistics.rim = compute_rim_thickness(outer=role_masks[outer], inner=role_masks[inner], spacing=spacing)

        """5. Overlaps between compartments meant to be disjoint, reported as inconsistencies"""
        for pair in composition["disjoint"]:
            if pair[0] in role_masks.keys() and pair[1] in role_masks.keys():
                overlap = np.count_nonzero(np.logical_and(role_masks[pair[0]], role_masks[pair[1]])) * voxel_volume_ml
                if overlap > 0.:
                    statistics.consistency.append([pair[0], pair[1], overlap])
                    logging.warning(f"[Composition] {pair[0]} and {pair[1]} overlap ({round(overlap, 2)} ml) while meant "
                                    f"to be disjoint.")
    except Exception as e:
        raise ValueError(f"Composition computation failed with: {e}")
    return statistics
