import logging
import operator
import traceback
import numpy as np
from typing import Tuple, List, Union
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import measurements, label, find_objects
from skimage.measure import regionprops
from scipy.ndimage import binary_closing, _ni_support
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from ..Utils.utilities import round


def compute_volume(volume: np.ndarray, spacing: Tuple, brain_mask: np.ndarray = None) -> Tuple[float, float]:
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
    brain_perc = -1.
    try:
        logging.debug("Computing tumor volume.")
        voxel_size = np.prod(spacing[0:3])
        volume_pixels = np.count_nonzero(volume)
        volume_mmcube = voxel_size * volume_pixels
        volume_ml = volume_mmcube * 1e-3
        res_volume = float(round(volume_ml, 2))
        if brain_mask is not None:
            brain_volume = np.count_nonzero(brain_mask) * voxel_size * 1e-3
            brain_perc = volume_ml / brain_volume
            brain_perc = round(brain_perc, 4)
    except Exception as e:
        raise ValueError('Volume computation failed with: {}'.format(e))

    return res_volume, brain_perc


def compute_diameters(volume: np.ndarray, spacing: Tuple) -> Tuple[float, float, float, float]:
    """

    """
    long_axis = -1.
    short_axis = -1.
    feret = -1.
    equivalent = -1.
    try:
        logging.debug("Computing diameter characteristics.")
        clusters = measurements.label(volume)[0]
        clusters_labels = regionprops(clusters)

        max_area = 0.
        for c in clusters_labels:
            if c.area > max_area:
                max_area = c.area
                long_axis = round(c.axis_major_length * np.prod(spacing[0:3]), 3)
                short_axis = round(c.axis_minor_length * np.prod(spacing[0:3]), 3)
                feret = round(c.feret_diameter_max * np.prod(spacing[0:3]), 3)
                equivalent = c.equivalent_diameter_area * np.prod(spacing[0:3])
    except Exception as e:
        raise ValueError(f'Diameter computation failed with: {e}')
    
    return float(long_axis), float(short_axis), float(feret), float(equivalent)

def group_foci(volume: np.ndarray, spacing: Tuple, min_volume_ml: float = 0., distance_threshold_mm: float = 0.,
               connectivity: int = 1) -> list:
    """
    Foci of the object, as [volume in ml, focus mask, number of connected components], ranked by decreasing volume.
    Connected components whose minimum distance (between voxel centers, in mm) is below distance_threshold_mm are
    grouped as a single focus, transitively (A close to B and B close to C makes a single focus). The foci smaller than
    min_volume_ml are then discarded, so that a fragmented lesion made of small close parts is kept as one focus.
    The connectivity is the one of scipy.ndimage.generate_binary_structure (1 = faces only, 3 = faces, edges and corners).
    """
    foci = []
    try:
        spacing = np.asarray(spacing[0:3], dtype=float)
        clusters, nb_clusters = label(volume > 0, structure=generate_binary_structure(3, connectivity))
        # Group index of each component, merged groups take the smallest index
        group_ids = np.arange(nb_clusters + 1)
        if distance_threshold_mm > 0. and nb_clusters > 1:
            margin = np.ceil(distance_threshold_mm / spacing).astype(int) + 1
            bounding_boxes = find_objects(clusters)
            for c in range(1, nb_clusters + 1):
                # Only the components within the threshold of the bounding box can be close enough
                crop = []
                for d in range(3):
                    crop.append(slice(max(bounding_boxes[c - 1][d].start - margin[d], 0),
                                      min(bounding_boxes[c - 1][d].stop + margin[d], clusters.shape[d])))
                crop = tuple(crop)
                crop_clusters = clusters[crop]
                neighbours = np.unique(crop_clusters)
                neighbours = neighbours[neighbours > c]
                if len(neighbours) == 0:
                    continue
                distances = distance_transform_edt(crop_clusters != c, sampling=spacing)
                for n in neighbours:
                    if np.min(distances[crop_clusters == n]) < distance_threshold_mm:
                        old_id = max(group_ids[c], group_ids[n])
                        new_id = min(group_ids[c], group_ids[n])
                        group_ids[group_ids == old_id] = new_id

        for g in np.unique(group_ids[1:]):
            members = np.where(group_ids == g)[0]
            members = members[members > 0]
            focus = np.isin(clusters, members)
            focus_volume = np.count_nonzero(focus) * np.prod(spacing) * 1e-3
            if focus_volume >= min_volume_ml:
                foci.append([focus_volume, focus, len(members)])
        foci = sorted(foci, key=operator.itemgetter(0), reverse=True)
    except Exception as e:
        raise ValueError('Foci grouping failed with: {}'.format(e))
    return foci

def compute_foci(volume: np.ndarray, spacing: Tuple, affine: np.ndarray = None, min_volume_ml: float = 0.1,
                 distance_threshold_mm: float = 5., connectivity: int = 1) -> list:
    """
    Every focus of the object, as FocusStatistics ranked by decreasing volume, with the same thresholds as
    compute_multifocality (pathology-dependent): the parts closer than distance_threshold_mm are grouped as a single
    focus and measured together, and the foci smaller than min_volume_ml are discarded (see group_foci).
    Each satellite focus gets its distance to the main focus, both the HD95 reported by compute_multifocality and the
    minimum distance (always at least distance_threshold_mm, the foci being grouped otherwise).
    """
    from ..Utils.DataStructures.StatisticsStructure import FocusStatistics, DiameterStatistics
    from .common_features_computation import compute_diameters
    
    foci = []
    nb_focus = None
    multifocality_status = None
    max_distance = -1.
    try:
        grouped_volume = np.zeros(volume.shape)
        spacing = np.asarray(spacing[0:3], dtype=float)
        grouped_foci = group_foci(volume=volume, spacing=spacing, min_volume_ml=min_volume_ml,
                                  distance_threshold_mm=distance_threshold_mm, connectivity=connectivity)
        for i, foc in enumerate(grouped_foci):
            grouped_volume[foc[1]] = i + 1
        nb_focus = len(grouped_foci)
        multifocality_status = nb_focus > 1
        main_focus = None
        main_focus_distances = None
        for i in range(len(grouped_foci)):
            focus_volume, focus, components = grouped_foci[i]
            distance_to_main = 0.
            min_distance_to_main = 0.
            if i == 0:
                main_focus = focus
                main_focus_distances = distance_transform_edt(np.logical_not(main_focus), sampling=spacing)
            else:
                distance_to_main = float(compute_hd95(focus, main_focus, voxelspacing=spacing, connectivity=1))
                min_distance_to_main = float(np.min(main_focus_distances[focus]))
                if min_distance_to_main > max_distance:
                    max_distance = min_distance_to_main
            long_axis, short_axis, feret, equivalent = compute_diameters(volume=focus, spacing=spacing,
                                                                                             affine=affine)
            centroid = np.asarray(center_of_mass(focus))
            if affine is not None:
                centroid = np.asarray(affine)[0:3, 0:3] @ centroid + np.asarray(affine)[0:3, 3]
            else:
                centroid = centroid * spacing
            foci.append(FocusStatistics(rank=i + 1, volume=focus_volume, centroid=list(centroid),
                                        diameters=DiameterStatistics(long_axis_diameter=long_axis,
                                                                     short_axis_diameter=short_axis,
                                                                     feret_diameter=feret,
                                                                     equivalent_diameter_area=equivalent),
                                        measurement_slice=None, distance_to_main=distance_to_main,
                                        min_distance_to_main=min_distance_to_main, components=components))
    except Exception as e:
        raise ValueError('Foci computation failed with: {}'.format(e))
    return multifocality_status, nb_focus, max_distance, foci, grouped_volume

def compute_multifocality(volume: np.ndarray, spacing: Tuple,
                          volume_threshold: float = 0.,
                          distance_threshold: float = 0.) -> Tuple[bool, int, float]:
    """

    Parameters
    ----------
    volume : np.ndarray
        Tumor annotation mask.
    spacing : tuple
        Spacing values for the provided volume array in order to relate pixels to the metric space.
    volume_threshold: float
        Minimum volume, in milliliters, for a segmented object to be considered as a focus.
    distance_threshold: float
        Minimum distance, in millimeters, between two foci to consider actual multifocality.
    Returns
    -------
    bool
        Multifocality state, True if multiple focus and False otherwise.
    int
        Number of focus(i).
    float
        Maximum minimum distance between two foci.
    """
    multifocal_status = False
    multifocal_elements = 1
    multifocal_largest_minimum_distance = -1.

    try:
        logging.debug("Computing multifocality characteristics.")
        tumor_clusters = measurements.label(volume)[0]
        tumor_clusters_labels = regionprops(tumor_clusters)

        if len(tumor_clusters_labels) > 1:
            multifocal_elements = 0

            # Computing the radius of the largest component.
            radiuses = []
            parts_labels = []
            for l in range(len(tumor_clusters_labels)):
                volume_ml = np.count_nonzero(tumor_clusters == (l + 1)) * np.prod(spacing[0:3]) * 1e-3
                if volume_ml >= volume_threshold:  # Discarding any tumor focus smaller than volume_threshold (in ml)
                    multifocal_elements = multifocal_elements + 1
                    radiuses.append(tumor_clusters_labels[l].equivalent_diameter / 2.)
                    parts_labels.append(l)
            max_radius = np.max(radiuses)
            max_radius_index = parts_labels[radiuses.index(max_radius)]

            # Computing the minimum distances between each focus
            main_tumor_label = np.zeros(volume.shape)
            main_tumor_label[tumor_clusters == (max_radius_index + 1)] = 1
            for l, lab in enumerate(parts_labels):
                if lab != max_radius_index:
                    satellite_label = np.zeros(volume.shape)
                    satellite_label[tumor_clusters == (lab + 1)] = 1
                    dist = compute_hd95(satellite_label, main_tumor_label, voxelspacing=spacing, connectivity=1)
                    if multifocal_largest_minimum_distance is None:
                        multifocal_largest_minimum_distance = dist
                    elif dist > multifocal_largest_minimum_distance:
                        multifocal_largest_minimum_distance = dist

            if multifocal_largest_minimum_distance >= distance_threshold:
                multifocal_status = True
    except Exception as e:
        raise ValueError('Multifocality computation failed with: {}'.format(e))

    return multifocal_status, multifocal_elements, float(multifocal_largest_minimum_distance)

def compute_depth(structure_mask: np.ndarray, target_mask: np.ndarray) -> Tuple[float, float, float]:
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

def compute_lateralisation(volume: np.ndarray, brain_mask: np.ndarray,
                           target: str = 'full') -> Tuple[float, float, bool]:
    """

    Parameters
    ----------
    volume : np.ndarray
        Tumor annotation mask.
    brain_mask: np.ndarray
        Brain mask with lateralization information, the right hemisphere has label 1 and the left has label 2.
    target: str
        ['com', 'full'] In 'com', the lateralisation is computed over the tumor center of mass. In 'full', the whole
        tumor extent is used to compute the lateralisation.
    Returns
    -------
    float
        Percentage of tumor overlap with the brain right hemisphere
    float
        Percentage of tumor overlap with the brain left hemisphere
    bool
        True if the tumor overlaps with both hemispheres, and False otherwise.
    """
    left_hemisphere_percentage = 0.
    right_hemisphere_percentage = 0.
    midline_crossing = False

    try:
        logging.debug("Computing lateralization characteristics.")
        # Computing the lateralisation for the center of mass
        if target == 'com':
            com_lateralisation = None
            com = center_of_mass(volume == 1)
            com_lateralization = brain_mask[int(np.round(com[0])) - 3:int(np.round(com[0])) + 3,
                                 int(np.round(com[1]))-3:int(np.round(com[1]))+3,
                                 int(np.round(com[2]))-3:int(np.round(com[2]))+3]
            com_sides_touched = list(np.unique(com_lateralization))
            if 0 in com_sides_touched:
                com_sides_touched.remove(0)
            percentage_each_com_side = [np.count_nonzero(com_lateralization == x) / np.count_nonzero(com_lateralization) for x in com_sides_touched]
            max_per = np.max(percentage_each_com_side)
            if com_sides_touched[percentage_each_com_side.index(max_per)] == 1:
                com_lateralisation = 'Right'
            else:
                com_lateralisation = 'Left'

            left_hemisphere_percentage = 100. if com_lateralisation == 'Right' else 0.
            right_hemisphere_percentage = 100. if com_lateralisation == 'Left' else 0.
            midline_crossing = True if len(com_sides_touched) == 2 else False
        elif target == 'full':
            # Computing the lateralisation for the overall tumor extent
            right_side_percentage = np.count_nonzero((brain_mask == 1) & (volume != 0)) / np.count_nonzero((volume != 0))
            left_side_percentage = np.count_nonzero((brain_mask == 2) & (volume != 0)) / np.count_nonzero((volume != 0))

            left_hemisphere_percentage = round(left_side_percentage * 100., 2)
            right_hemisphere_percentage = round(right_side_percentage * 100., 2)
            midline_crossing = True if max(left_hemisphere_percentage, right_hemisphere_percentage) < 100. else False
        else:
            pass
    except Exception as e:
        raise ValueError('Lateralisation computation failed with {}'.format(e))

    return float(left_hemisphere_percentage), float(right_hemisphere_percentage), midline_crossing


def compute_resectability_index(volume: np.ndarray, resectability_map: np.ndarray) -> Tuple[float, float, float]:
    """

    Parameters
    ----------
    volume : np.ndarray
        Tumor annotation mask.
    resectability_map: np.ndarray
        Probability map with computed resectability for each voxel, as introduced in Amsterdam's article.
        Should add the article DOI here.
    Returns
    -------
    float
        Residual tumor volume in milliliters.
    float
        Resectable volume in milliliters.
    float
        Average resectability.
    """

    residual_tumor_volume = 0.
    resectable_volume = 0.
    avg_resectability = 0.

    try:
        logging.debug("Computing tumor resectability index.")
        resectability_map = np.nan_to_num(resectability_map)
        tumor_voxels_count = np.count_nonzero(volume)
        total_resectability = np.sum(resectability_map[volume != 0])
        resectable_volume = total_resectability * 1e-3
        residual_tumor_volume = (tumor_voxels_count * 1e-3) - resectable_volume
        avg_resectability = total_resectability / tumor_voxels_count
    except Exception as e:
        raise ValueError('Resectability index computation failed with {}'.format(e))
    return float(residual_tumor_volume), float(resectable_volume), float(avg_resectability)


def compute_hd95(reference, result, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds
