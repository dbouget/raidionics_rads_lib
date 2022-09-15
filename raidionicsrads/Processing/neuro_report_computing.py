import pandas as pd
from copy import deepcopy
import operator
import collections
import nibabel as nib
from skimage.morphology import ball
from scipy.ndimage import binary_closing
from ..Processing.tumor_features_computation import *
from ..Utils.io import load_nifti_volume
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.ReportingStructures.NeuroReportingStructure import NeuroReportingStructure


def compute_neuro_report(input_filename: str, report: NeuroReportingStructure) -> NeuroReportingStructure:
    """

    """
    registered_tumor_ni = load_nifti_volume(input_filename)
    registered_tumor = registered_tumor_ni.get_data()[:]

    tumor_type = report._tumor_type
    if np.count_nonzero(registered_tumor) == 0:
        report.setup(tumor_type=tumor_type, tumor_elements=0)
        return report

    # Cleaning the segmentation mask just in case, removing potential small and noisy areas
    cluster_size_cutoff_in_pixels = 100
    kernel = ball(radius=2)
    img_ero = binary_closing(registered_tumor, structure=kernel, iterations=1)
    tumor_clusters = measurements.label(img_ero)[0]
    refined_image = deepcopy(tumor_clusters)
    for c in range(1, np.max(tumor_clusters) + 1):
        if np.count_nonzero(tumor_clusters == c) < cluster_size_cutoff_in_pixels:
            refined_image[refined_image == c] = 0
    refined_image[refined_image != 0] = 1

    if np.count_nonzero(refined_image) == 0:
        report.setup(tumor_type=tumor_type, tumor_elements=0)
        return report

    # Computing the tumor volume in original patient space
    # segmentation_ni = nib.load(ResourcesConfiguration.getInstance().runtime_tumor_mask_filepath)
    # segmentation_mask = segmentation_ni.get_data()[:]
    # volume = compute_volume(volume=segmentation_mask, spacing=segmentation_ni.header.get_zooms())
    # self.diagnosis_parameters.statistics['Main']['Overall'].original_space_tumor_volume = volume

    # Assessing if the tumor is multifocal or monofocal
    tumor_clusters = measurements.label(refined_image)[0]
    tumor_clusters_labels = regionprops(tumor_clusters)
    report.setup(tumor_type=tumor_type, tumor_elements=len(tumor_clusters_labels))
    volume_reg_space = compute_volume(volume=refined_image, spacing=registered_tumor_ni.header.get_zooms())
    report._statistics['Main']['Overall'].mni_space_tumor_volume = volume_reg_space

    status, nb, dist = compute_multifocality(volume=refined_image, spacing=registered_tumor_ni.header.get_zooms(),
                                             volume_threshold=0.1, distance_threshold=5.0)
    report._tumor_multifocal = status
    report._tumor_parts = nb
    report._tumor_multifocal_distance = dist

    # Computing localisation and lateralisation for the whole tumor extent
    brain_lateralisation_mask_ni = load_nifti_volume(
        ResourcesConfiguration.getInstance().mni_atlas_lateralisation_mask_filepath)
    brain_lateralisation_mask = brain_lateralisation_mask_ni.get_data()[:]
    left, right, mid = compute_lateralisation(volume=refined_image, brain_mask=brain_lateralisation_mask)
    report._statistics['Main']['Overall'].left_laterality_percentage = left
    report._statistics['Main']['Overall'].right_laterality_percentage = right
    report._statistics['Main']['Overall'].laterality_midline_crossing = mid

    if report._tumor_type == 'Glioblastoma':
        if report._statistics['Main']['Overall'].left_laterality_percentage >= 0.5:
            map_filepath = ResourcesConfiguration.getInstance().mni_resection_maps['Probability']['Left']
        else:
            map_filepath = ResourcesConfiguration.getInstance().mni_resection_maps['Probability']['Right']

        resection_probability_map_ni = nib.load(map_filepath)
        resection_probability_map = resection_probability_map_ni.get_data()[:]
        residual, resectable, average = compute_resectability_index(volume=refined_image,
                                                                    resectability_map=resection_probability_map)
        report._statistics['Main']['Overall'].mni_space_expected_residual_tumor_volume = residual
        report._statistics['Main']['Overall'].mni_space_expected_resectable_tumor_volume = resectable
        report._statistics['Main']['Overall'].mni_space_resectability_index = average

    for s in ResourcesConfiguration.getInstance().neuro_features_cortical_structures:
        overlap = compute_cortical_structures_location(volume=refined_image, reference=s)
        report._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[s] = overlap
        # if self.from_slicer:
        #     ordered_l = collections.OrderedDict(sorted(report._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[s].items(), key=operator.itemgetter(1), reverse=True))
        #     report._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[s] = ordered_l
    for s in ResourcesConfiguration.getInstance().neuro_features_subcortical_structures:
        overlaps, distances = compute_subcortical_structures_location(volume=refined_image, category='Main', reference=s)
        if False: #self.from_slicer:
            sorted_d = collections.OrderedDict(sorted(distances.items(), key=operator.itemgetter(1), reverse=False))
            sorted_o = collections.OrderedDict(sorted(overlaps.items(), key=operator.itemgetter(1), reverse=True))
            report._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[s] = sorted_o
            report._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[s] = sorted_d
        else:
            report._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[s] = overlaps
            report._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[s] = distances

    return report


def compute_cortical_structures_location(volume, reference='MNI'):
    logging.debug("Computing cortical structures location with {}.".format(reference))
    regions_data = ResourcesConfiguration.getInstance().cortical_structures['MNI'][reference]
    region_mask_ni = nib.load(regions_data['Mask'])
    region_mask = region_mask_ni.get_data()
    lobes_description = pd.read_csv(regions_data['Description'])

    # Computing the lobe location for the center of mass
    # @TODO. to check
    # com = center_of_mass(volume == 1)
    # com_label = region_mask[int(np.round(com[0])) - 3:int(np.round(com[0])) + 3,
    #             int(np.round(com[1]))-3:int(np.round(com[1]))+3,
    #             int(np.round(com[2]))-3:int(np.round(com[2]))+3]
    # com_lobes_touched = list(np.unique(com_label))
    # if 0 in com_lobes_touched:
    #     com_lobes_touched.remove(0)
    # percentage_each_com_lobe = [np.count_nonzero(com_label == x) / np.count_nonzero(com_label) for x in com_lobes_touched]
    # max_per = np.max(percentage_each_com_lobe)
    # com_lobe = lobes_description.loc[lobes_description['Label'] == com_lobes_touched[percentage_each_com_lobe.index(max_per)]]
    # center_of_mass_lobe = com_lobe['Region'].values[0]
    # self.diagnosis_parameters.statistics[category]['CoM'].mni_space_cortical_structures_overlap[reference][center_of_mass_lobe] = np.round(max_per * 100, 2)

    total_lobes_labels = np.unique(region_mask)[1:]  # Removing the background label with value 0.
    overlap_per_lobe = {}
    for li in total_lobes_labels:
        overlap = volume[region_mask == li]
        ratio_in_lobe = np.count_nonzero(overlap) / np.count_nonzero(volume)
        overlap = np.round(ratio_in_lobe * 100., 2)
        region_name = ''
        if reference == 'MNI':
            region_name = '-'.join(lobes_description.loc[lobes_description['Label'] == li]['Region'].values[0].strip().split(' ')) + '_' + (lobes_description.loc[lobes_description['Label'] == li]['Laterality'].values[0].strip() if lobes_description.loc[lobes_description['Label'] == li]['Laterality'].values[0].strip() is not 'None' else '')
        elif reference == 'Harvard-Oxford':
            region_name = '-'.join(lobes_description.loc[lobes_description['Label'] == li]['Region'].values[0].strip().split(' '))
        else:
            region_name = '_'.join(lobes_description.loc[lobes_description['Label'] == li]['Region'].values[0].strip().split(' '))
        overlap_per_lobe[region_name] = overlap

    return overlap_per_lobe


def compute_subcortical_structures_location(volume, category=None, reference='BCB'):
    logging.debug("Computing subcortical structures location with {}.".format(reference))
    distances = {}
    overlaps = {}
    distances_columns = []
    overlaps_columns = []
    tract_cutoff = 0.5
    if reference == 'BrainLab':
        tract_cutoff = 0.25

    tracts_dict = ResourcesConfiguration.getInstance().subcortical_structures['MNI'][reference]['Singular']
    for i, tfn in enumerate(tracts_dict.keys()):
        reg_tract_ni = nib.load(tracts_dict[tfn])
        reg_tract = reg_tract_ni.get_data()[:]
        reg_tract[reg_tract < tract_cutoff] = 0
        reg_tract[reg_tract >= tract_cutoff] = 1
        overlap_volume = np.logical_and(reg_tract, volume).astype('uint8')
        distances_columns.append('distance_' + tfn.split('.')[0][:-4] + '_' + category)
        overlaps_columns.append('overlap_' + tfn.split('.')[0][:-4] + '_' + category)
        if np.count_nonzero(overlap_volume) != 0:
            distances[tfn] = -1.
            overlaps[tfn] = (np.count_nonzero(overlap_volume) / np.count_nonzero(volume)) * 100.
        else:
            dist = -1.
            if np.count_nonzero(reg_tract) > 0:
                dist = hd95(volume, reg_tract, voxelspacing=reg_tract_ni.header.get_zooms(), connectivity=1)
            distances[tfn] = dist
            overlaps[tfn] = 0.

    return overlaps, distances
