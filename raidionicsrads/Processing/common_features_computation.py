import logging
import traceback
import numpy as np
from typing import Tuple, List, Union
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import measurements, label
from skimage.measure import regionprops, marching_cubes, find_contours
from scipy.ndimage import binary_closing, _ni_support, binary_fill_holes
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from ..Utils.utilities import round


def compute_volume(volume: np.ndarray, spacing: Tuple, base_mask: np.ndarray = None) -> Tuple[float, float]:
    """

    Parameters
    ----------
    volume : np.ndarray
        Tumor annotation mask.
    spacing : tuple
        Spacing values for the provided volume array in order to relate pixels to the metric space.
    Returns
    -------
    float
        Object volume in milliliters, rounded at two decimals.
    """
    res_volume = 0.
    base_perc = -1.
    try:
        logging.debug("Computing tumor volume.")
        voxel_size = np.prod(spacing[0:3])
        volume_pixels = np.count_nonzero(volume)
        volume_mmcube = voxel_size * volume_pixels
        volume_ml = volume_mmcube * 1e-3
        res_volume = float(round(volume_ml, 3))
        if base_mask is not None:
            base_volume = np.count_nonzero(base_mask) * voxel_size * 1e-3
            base_perc = volume_ml / base_volume
            base_perc = round(base_perc, 4)
    except Exception as e:
        raise ValueError('Volume computation failed with: {}'.format(e))

    return res_volume, base_perc

def compute_depth(structure_mask: np.ndarray, target_mask: np.ndarray, spacing: Tuple = (1., 1., 1.)) -> Tuple[float, float, float]:
    """
    Depth (mm) of the structure below the outer surface of the reference structure (e.g., tumor below the brain
    surface), as the distance from each structure voxel to the reference surface. The holes inside the reference
    (e.g., ventricles excluded from the brain mask) are filled so that only the outer surface counts, and the parts of
    the structure outside the reference are at depth 0.

    Parameters
    ----------
    structure_mask : np.ndarray
        Annotation mask for the structure of interest.
    target_mask: np.ndarray
        Annotation mask for the structure of reference.
    spacing : tuple
        Spacing values for the provided volume arrays in order to relate pixels to the metric space.
    Returns
    -------
    float
        Minimum depth (mm), 0 when the structure reaches the reference surface.
    float
        Maximum depth (mm).
    float
        Mean depth (mm) over the structure voxels.
    Values are -1 for an empty structure.
    """
    min_depth = -1.
    max_depth = -1.
    mean_depth = -1.

    try:
        structure = structure_mask > 0
        if np.count_nonzero(structure) == 0:
            return min_depth, max_depth, mean_depth
        target = binary_fill_holes(target_mask > 0)
        target_surface = np.logical_and(target, np.logical_not(binary_erosion(target)))
        distance_to_surface = distance_transform_edt(np.logical_not(target_surface),
                                                     sampling=np.asarray(spacing[0:3], dtype=float))
        distance_to_surface[np.logical_not(target)] = 0.
        structure_depths = distance_to_surface[structure]
        min_depth = float(structure_depths.min())
        max_depth = float(structure_depths.max())
        mean_depth = float(structure_depths.mean())
    except Exception as e:
        raise ValueError(f'Depth computation failed with {e}.')

    return min_depth, max_depth, mean_depth

def compute_depth_old(structure_mask: np.ndarray, target_mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Parameters
    ----------
    structure_mask : np.ndarray
        Annotation mask for the structure of interest.
    target_mask: np.ndarray
        Annotation mask for the structure of reference.
    Returns
    -------
    float
    float
    float
    """
    min_depth = -1.
    max_depth = -1.
    mean_depth = -1.

    try:
        target_inner = binary_erosion(target_mask)
        target_surface = target_mask.astype('bool') ^ target_inner
        distance_to_surface = distance_transform_edt(~target_surface)
        structure_depths = distance_to_surface[structure_mask == 1]
        max_depth = structure_depths.max()
        mean_depth = structure_depths.mean()
        min_depth = structure_depths.min()
    except Exception as e:
        raise ValueError(f'Depth computation failed with {e}.')

    return min_depth, max_depth, mean_depth

def compute_diameters(volume: np.ndarray, spacing: Tuple, affine: np.ndarray = None) -> Tuple[float, float, float, float]:
    """
    Diameters of the largest focus (connected component) of the object, all in millimeters and computed from the
    physical extent of the voxels, so that anisotropic spacings are properly handled.
    The long and short axes follow the clinical (RANO/RECIST) definition: the longest in-plane diameter over all the
    axial slices, and the longest diameter perpendicular to it on the same slice.

    Parameters
    ----------
    volume : np.ndarray
        Object annotation mask.
    spacing : tuple
        Spacing values for the provided volume array in order to relate pixels to the metric space.
    affine : np.ndarray
        Affine matrix of the volume, to identify the axial (through-plane) axis. When not provided, the third array
        axis is assumed to be the through-plane axis.
    Returns
    -------
    float
        Long-axis diameter, longest in-plane diameter in an axial slice.
    float
        Short-axis diameter, longest diameter perpendicular to the long axis on the same slice.
    float
        Maximum 3D Feret diameter, largest distance between two points of the object.
    float
        Equivalent diameter, diameter of the sphere with the same volume as the object.
    Values are -1 for an empty mask.
    """
    long_axis = -1.
    short_axis = -1.
    feret = -1.
    equivalent = -1.
    try:
        logging.debug("Computing diameter characteristics.")
        spacing = np.asarray(spacing[0:3], dtype=float)
        if np.count_nonzero(volume) == 0:
            return float(long_axis), float(short_axis), float(feret), float(equivalent)

        equivalent = (6. * np.count_nonzero(volume) * np.prod(spacing) / np.pi) ** (1. / 3.)
        feret, _, _ = __compute_max_distance(points=__surface_points(mask=volume, spacing=spacing))

        through_plane_axis = __identify_through_plane_axis(affine=affine)
        in_plane_axes = [x for x in range(3) if x != through_plane_axis]
        in_plane_spacing = spacing[in_plane_axes]
        best_slice = None
        best_long = -1.
        best_extremities = None
        for s in np.unique(np.argwhere(volume)[:, through_plane_axis]):
            focus_slice = np.take(volume, indices=s, axis=through_plane_axis)
            slice_long, p1, p2 = __compute_max_distance(points=__surface_points(mask=focus_slice,
                                                                               spacing=in_plane_spacing))
            if slice_long > best_long:
                best_long = slice_long
                best_slice = focus_slice
                best_extremities = [p1, p2]
        long_axis = best_long
        short_axis = __compute_perpendicular_diameter(mask=best_slice, spacing=in_plane_spacing,
                                                      p1=best_extremities[0], p2=best_extremities[1])
    except Exception as e:
        raise ValueError('Diameters computation failed with: {}'.format(e))

    return round(long_axis, 3), round(short_axis, 3), round(feret, 3), round(equivalent, 3)


def __identify_through_plane_axis(affine: np.ndarray = None) -> int:
    """
    Array axis whose direction is the most aligned with the superior-inferior axis, i.e. the axis between axial slices.
    """
    if affine is None:
        return 2
    directions = np.abs(np.asarray(affine)[0:3, 0:3])
    return int(np.argmax(directions[2, :] / np.linalg.norm(directions, axis=0)))


def __surface_points(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    Physical positions (in mm) of the object boundary, taken half-way between the inside and outside voxels (marching
    cubes in 3D, contours in 2D), in the frame where the voxel of index i is centered at i * spacing.
    Measuring between voxel corners instead would overestimate the diagonal distances by up to one voxel.
    """
    padded = np.pad(mask.astype(float), pad_width=1)
    if mask.ndim == 3:
        points = marching_cubes(padded, level=0.5)[0]
    else:
        points = np.concatenate(find_contours(padded, level=0.5), axis=0)
    # Removing the padding offset before converting to millimeters
    return (points - 1.) * spacing


def __voxel_corners(coords: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """
    Physical positions (in mm) of the corners of the given voxels/pixels, giving the full extent of the object (used to
    bound the sampling of the perpendicular diameter).
    """
    dims = coords.shape[1]
    offsets = []
    for i in range(2 ** dims):
        offset = []
        for d in range(dims):
            if (i >> d) & 1:
                offset.append(0.5)
            else:
                offset.append(-0.5)
        offsets.append(offset)
    offsets = np.asarray(offsets)
    corners = (coords[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, dims) * spacing
    return np.unique(corners, axis=0)


def __compute_max_distance(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Largest distance between two of the points, with the two points. The candidates are reduced to the vertices of the
    convex hull, on which the largest distance always lies.
    """
    candidates = points
    try:
        candidates = points[ConvexHull(points).vertices]
    except Exception:
        # Degenerated point sets (e.g., all aligned) where the hull cannot be computed
        pass
    distances = cdist(candidates, candidates)
    i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return float(distances[i, j]), candidates[i], candidates[j]


def __compute_perpendicular_diameter(mask: np.ndarray, spacing: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Longest segment inside the 2D mask perpendicular to the [p1, p2] axis, found by sampling lines perpendicular to
    the axis along its whole length. Only contiguous inside segments are considered, a line crossing the object twice
    is not measured across the gap.
    """
    step = float(np.min(spacing)) / 4.
    u = (p2 - p1) / np.linalg.norm(p2 - p1)
    v = np.array([-u[1], u[0]])
    corners = __voxel_corners(coords=np.argwhere(mask), spacing=spacing)
    u_positions = np.arange(np.min(corners @ u), np.max(corners @ u) + step, step)
    v_positions = np.arange(np.min(corners @ v), np.max(corners @ v) + step, step)

    # Samples of each perpendicular line (one row per line), converted back to pixel indices
    samples = u_positions[:, np.newaxis, np.newaxis] * u + v_positions[np.newaxis, :, np.newaxis] * v
    indices = np.round(samples / spacing).astype(int)
    valid = np.logical_and(np.all(indices >= 0, axis=2), np.all(indices < np.array(mask.shape), axis=2))
    inside = np.zeros(valid.shape, dtype=bool)
    inside[valid] = mask[indices[valid][:, 0], indices[valid][:, 1]]

    longest_run = 0
    for line in inside:
        padded = np.concatenate([[0], line.astype(int), [0]])
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) != 0:
            longest_run = max(longest_run, int(np.max(ends - starts)))
    return float(longest_run * step)


# def compute_foci(volume: np.ndarray, spacing: Tuple, affine: np.ndarray = None, min_volume_ml: float = 0.1,
#                  connectivity: int = 1) -> list:
#     """
#     Every focus (connected component) of the object of at least min_volume_ml, ranked by decreasing volume, as
#     FocusStatistics. The connectivity is the one of scipy.ndimage.generate_binary_structure (1 = faces only, 3 = faces,
#     edges and corners).
#     """
#     from ..Utils.DataStructures.StatisticsStructure import FocusStatistics, DiameterStatistics
#     foci = []
#     try:
#         spacing = np.asarray(spacing[0:3], dtype=float)
#         clusters, nb_clusters = label(volume > 0, structure=generate_binary_structure(3, connectivity))
#         candidates = []
#         for c in range(1, nb_clusters + 1):
#             focus = clusters == c
#             focus_volume = np.count_nonzero(focus) * np.prod(spacing) * 1e-3
#             if focus_volume >= min_volume_ml:
#                 candidates.append([focus_volume, c])
#         candidates = sorted(candidates, reverse=True)
#         for i in range(len(candidates)):
#             focus = clusters == candidates[i][1]
#             long_axis, short_axis, feret, equivalent, slice_index = compute_focus_diameters(focus=focus, spacing=spacing,
#                                                                                              affine=affine)
#             centroid = np.asarray(center_of_mass(focus))
#             if affine is not None:
#                 centroid = np.asarray(affine)[0:3, 0:3] @ centroid + np.asarray(affine)[0:3, 3]
#             else:
#                 centroid = centroid * spacing
#             foci.append(FocusStatistics(rank=i + 1, volume=candidates[i][0], centroid=list(centroid),
#                                         diameters=DiameterStatistics(long_axis_diameter=long_axis,
#                                                                      short_axis_diameter=short_axis,
#                                                                      feret_diameter=feret,
#                                                                      equivalent_diameter_area=equivalent),
#                                         measurement_slice=slice_index))
#     except Exception as e:
#         raise ValueError('Foci computation failed with: {}'.format(e))
#     return foci