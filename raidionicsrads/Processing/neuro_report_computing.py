import traceback

import pandas as pd
from copy import deepcopy
import operator
import collections
import nibabel as nib
from skimage.morphology import ball
from scipy.ndimage import binary_closing
from ..Processing.tumor_features_computation import *
from ..Utils.DataStructures.RadiologicalVolumeStructure import MRISequenceType
from ..Utils.io import load_nifti_volume
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.ReportingStructures.NeuroReportingStructure import *
from ..Utils.ReportingStructures.NeuroSurgicalReportingStructure import *


def compute_neuro_report(input_filename: str, report: NeuroReportingStructure) -> NeuroReportingStructure:
    """
    Main method computing all elements of the clinical report for a brain use-case.

    Parameters
    ---------
    input_filename: str

    report: NeuroReportingStructure
        Prefilled version of the report which will be further completed inside the method
    Return
    -------
    Full and final version of the report, filled in with all requested parameters.
    """
    try:
        registered_tumor_ni = load_nifti_volume(input_filename)
        registered_tumor = registered_tumor_ni.get_fdata()[:]

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
        # segmentation_mask = segmentation_ni.get_fdata()[:]
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
        brain_lateralisation_mask = brain_lateralisation_mask_ni.get_fdata()[:]
        left, right, mid = compute_lateralisation(volume=refined_image, brain_mask=brain_lateralisation_mask)
        report._statistics['Main']['Overall'].left_laterality_percentage = left
        report._statistics['Main']['Overall'].right_laterality_percentage = right
        report._statistics['Main']['Overall'].laterality_midline_crossing = mid

        if report._tumor_type == 'Glioblastoma':
            if report._statistics['Main']['Overall'].left_laterality_percentage >= 50.0:
                map_filepath = ResourcesConfiguration.getInstance().mni_resection_maps['Probability']['Left']
            else:
                map_filepath = ResourcesConfiguration.getInstance().mni_resection_maps['Probability']['Right']

            resection_probability_map_ni = nib.load(map_filepath)
            resection_probability_map = resection_probability_map_ni.get_fdata()[:]
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
        for s in ResourcesConfiguration.getInstance().neuro_features_braingrid:
            overlap_per_voxel, infiltrated_voxels = compute_braingrid_voxels_infiltration(volume=refined_image,
                                                                                           category='Main',
                                                                                           reference=s)
            report._statistics['Main']['Overall'].mni_space_braingrid_infiltration_overlap[s] = overlap_per_voxel
            report._statistics['Main']['Overall'].mni_space_braingrid_infiltration_count = infiltrated_voxels
        return report
    except Exception as e:
        raise ValueError("{}".format(e))


def compute_structure_statistics(input_mask: nib.Nifti1Image,
                                 brain_mask: nib.Nifti1Image = None) -> NeuroStructureStatistics:
    """

    Return
    -------

    """
    try:
        result = NeuroStructureStatistics()
        input_array = input_mask.get_fdata()[:]

        # Cleaning the segmentation mask just in case, removing potential small and noisy areas
        cluster_size_cutoff_in_pixels = 100
        kernel = ball(radius=2)
        img_ero = binary_closing(input_array, structure=kernel, iterations=1)
        tumor_clusters = measurements.label(img_ero)[0]
        refined_image = deepcopy(tumor_clusters)
        for c in range(1, np.max(tumor_clusters) + 1):
            if np.count_nonzero(tumor_clusters == c) < cluster_size_cutoff_in_pixels:
                refined_image[refined_image == c] = 0
        refined_image[refined_image != 0] = 1

        brain_array = None
        if brain_mask:
            brain_array = brain_mask.get_fdata()[:]
        volume, brain_perc = compute_volume(volume=refined_image, spacing=input_mask.header.get_zooms(),
                                            brain_mask=brain_array)
        result.volume = NeuroVolumeStatistics(volume=volume, brain_percentage=brain_perc)

        longa, shorta, feret, equi = compute_diameters(volume=refined_image, spacing=input_mask.header.get_zooms())
        result.diameters = NeuroDiameterStatistics(long_axis_diameter=longa, short_axis_diameter=shorta, feret_diameter=feret,
                                               equivalent_diameter_area=equi)

        status, nb, dist = compute_multifocality(volume=refined_image, spacing=input_mask.header.get_zooms(),
                                                 volume_threshold=0.1, distance_threshold=5.0)
        result.multifocality = NeuroMultifocalityStatistics(status=status, parts=nb, distance=dist)

        # Computing localisation features
        brain_lateralisation_mask_ni = load_nifti_volume(
            ResourcesConfiguration.getInstance().mni_atlas_lateralisation_mask_filepath)
        brain_lateralisation_mask = brain_lateralisation_mask_ni.get_fdata()[:]
        left, right, crossing = compute_lateralisation(volume=refined_image, brain_mask=brain_lateralisation_mask)
        result.location = NeuroLocationStatistics(left=left, right=right, crossing=crossing)

        # Compute resectability parameters -- @TODO. Should add a check on tumor type (should be only available for GBM)
        if left >= 50.0:
                map_filepath = ResourcesConfiguration.getInstance().mni_resection_maps['Probability']['Left']
        else:
            map_filepath = ResourcesConfiguration.getInstance().mni_resection_maps['Probability']['Right']
        resection_probability_map_ni = nib.load(map_filepath)
        resection_probability_map = resection_probability_map_ni.get_fdata()[:]

        residual, resectable, average = compute_resectability_index(volume=refined_image,
                                                                    resectability_map=resection_probability_map)
        result.resectability = NeuroResectabilityStatistics(resectable=resectable, residual=residual, index=average)
        
        # Computing cortical, subcortical, and infiltration profiles
        for s in ResourcesConfiguration.getInstance().neuro_features_cortical_structures:
            overlaps = compute_cortical_structures_location(volume=refined_image, reference=s)
            result.cortical[s] = NeuroCorticalStatistics(overlap=overlaps, distance=None)
        for s in ResourcesConfiguration.getInstance().neuro_features_subcortical_structures:
            overlaps, distances = compute_subcortical_structures_location(volume=refined_image,
                                                                          category='Main', reference=s)
            result.subcortical[s] = NeuroSubCorticalStatistics(overlap=overlaps, distance=distances)
        for s in ResourcesConfiguration.getInstance().neuro_features_braingrid:
            overlap_per_voxel, infiltrated_voxels = compute_braingrid_voxels_infiltration(volume=refined_image,
                                                                                           category='Main',
                                                                                           reference=s)
            result.infiltration[s] = NeuroInfiltrationStatistics(overlap=overlap_per_voxel, count=infiltrated_voxels)

        return result
    except Exception as e:
        raise ValueError(f"Structure features computation failed with: {e}")


def compute_cortical_structures_location(volume, reference='MNI'):
    logging.debug("Computing cortical structures location with {}.".format(reference))
    regions_data = ResourcesConfiguration.getInstance().cortical_structures['MNI'][reference]
    region_mask_ni = nib.load(regions_data['Mask'])
    region_mask = region_mask_ni.get_fdata()
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
        overlap = float(round(ratio_in_lobe * 100., 2))
        region_name = ''
        if reference == 'MNI':
            region_name = '-'.join(str(lobes_description.loc[lobes_description['Label'] == li]['Region'].values[0]).strip().split(' ')) + '_' + (str(lobes_description.loc[lobes_description['Label'] == li]['Laterality'].values[0]).strip() if str(lobes_description.loc[lobes_description['Label'] == li]['Laterality'].values[0]).strip() != 'None' else '')
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
        dist = -1.
        try:
            reg_tract_ni = nib.load(tracts_dict[tfn])
            reg_tract = reg_tract_ni.get_fdata()[:]
            reg_tract[reg_tract < tract_cutoff] = 0
            reg_tract[reg_tract >= tract_cutoff] = 1
            overlap_volume = np.logical_and(reg_tract, volume).astype('uint8')
            if reference == "BCB":
                distances_columns.append('distance_' + tfn.split('.')[0][:-4] + '_' + category)
                overlaps_columns.append('overlap_' + tfn.split('.')[0][:-4] + '_' + category)
            else:
                distances_columns.append('distance_' + tfn.split('.')[0] + '_' + category)
                overlaps_columns.append('overlap_' + tfn.split('.')[0] + '_' + category)
            if np.count_nonzero(overlap_volume) != 0:
                distances[tfn] = dist
                overlaps[tfn] = float((np.count_nonzero(overlap_volume) / np.count_nonzero(volume)) * 100.)
            else:
                if np.count_nonzero(reg_tract) > 0:
                    dist = compute_hd95(volume, reg_tract, voxelspacing=reg_tract_ni.header.get_zooms(), connectivity=1)
                distances[tfn] = dist
                overlaps[tfn] = 0.
        except Exception:
            print("Tumor distance to subcortical structure could not be computed.")
            print(traceback.format_exc())
            distances[tfn] = dist
    return overlaps, distances


def compute_braingrid_voxels_infiltration(volume, category=None, reference='Voxels'):
    logging.debug("Computing BrainGrid infiltration with {}.".format(reference))
    regions_data = ResourcesConfiguration.getInstance().braingrid_structures['MNI'][reference]
    region_mask_ni = nib.load(regions_data['Mask'])
    region_mask = region_mask_ni.get_fdata()
    lobes_description = pd.read_csv(regions_data['Description'])

    total_voxels_labels = np.unique(region_mask)[1:]  # Removing the background label with value 0.
    overlap_per_voxel = {}
    infiltrated_voxels = 0
    for li in total_voxels_labels:
        overlap = volume[region_mask == li]
        ratio_in_voxel = np.count_nonzero(overlap) / np.count_nonzero(volume)
        overlap = float(round(ratio_in_voxel * 100., 2))
        region_name = ''
        if reference == 'Voxels':
            region_name = '_'.join(lobes_description.loc[lobes_description['Label'] == li]['Region'].values[0].strip().split(' '))
        overlap_per_voxel[region_name] = overlap
        if overlap > 0:
            infiltrated_voxels += 1

    return overlap_per_voxel, infiltrated_voxels


def compute_surgical_report(brain_preop_fn: str, brain_postop_fn: str, tumor_preop_fn: str, tumor_postop_fn: str,
                            necrosis_preop_fn: str, necrosis_postop_fn: str, report, flairchanges_preop_fn: str = None,
                            flairchanges_postop_fn: str = None, cavity_postop_fn: str = None) -> None:
    """
    Update the report in-place with the computed values.
    What do the RANO guidelines say about contrast-enhancing versus not, regarding the assessment?
    How to check for supramaximal resection? => beyond CE tumor borders
    Is it correct to compare the tumorcore preop and tumorCE postop?
    """
    try:
        preop_brain_annotation_ni = nib.load(brain_preop_fn)
        postop_brain_annotation_ni = nib.load(brain_postop_fn)
        preop_brain_volume, _ = compute_volume(preop_brain_annotation_ni.get_fdata()[:], preop_brain_annotation_ni.header.get_zooms())
        postop_brain_volume, _ = compute_volume(postop_brain_annotation_ni.get_fdata()[:], postop_brain_annotation_ni.header.get_zooms())

        preop_annotation_ni = nib.load(tumor_preop_fn)
        postop_annotation_ni = nib.load(tumor_postop_fn)
        preop_volume, _ = compute_volume(preop_annotation_ni.get_fdata()[:], preop_annotation_ni.header.get_zooms())
        postop_volume, _ = compute_volume(postop_annotation_ni.get_fdata()[:], postop_annotation_ni.header.get_zooms())

        flairchanges_preop_volume = None
        if flairchanges_preop_fn is not None:
            flairchanges_preop_ni = nib.load(flairchanges_preop_fn)
            flairchanges_preop_volume, _ = compute_volume(flairchanges_preop_ni.get_fdata()[:],
                                                       flairchanges_preop_ni.header.get_zooms())

        flairchanges_postop_volume = None
        if flairchanges_postop_fn is not None:
            flairchanges_postop_ni = nib.load(flairchanges_postop_fn)
            flairchanges_postop_volume, _ = compute_volume(flairchanges_postop_ni.get_fdata()[:],
                                                        flairchanges_postop_ni.header.get_zooms())
        necrosis_preop_volume = None
        if necrosis_preop_fn is not None:
            necrosis_preop_ni = nib.load(necrosis_preop_fn)
            necrosis_preop_volume, _ = compute_volume(necrosis_preop_ni.get_fdata()[:],
                                                       necrosis_preop_ni.header.get_zooms())
        necrosis_postop_volume = None
        if necrosis_postop_fn is not None:
            necrosis_postop_ni = nib.load(necrosis_postop_fn)
            necrosis_postop_volume, _ = compute_volume(necrosis_postop_ni.get_fdata()[:],
                                                       necrosis_postop_ni.header.get_zooms())
        cavity_postop_volume = None
        if cavity_postop_fn is not None:
            cavity_postop_ni = nib.load(cavity_postop_fn)
            cavity_postop_volume, _ = compute_volume(cavity_postop_ni.get_fdata()[:], cavity_postop_ni.header.get_zooms())

        eor = ((preop_volume - postop_volume) / preop_volume) * 100.
        report.statistics.tumor_volume_preop = preop_volume
        report.statistics.tumor_volume_postop = postop_volume
        report.statistics.extent_of_resection = eor
        report.statistics.flairchanges_volume_preop = flairchanges_preop_volume
        report.statistics.flairchanges_volume_postop = flairchanges_postop_volume
        report.statistics.necrosis_volume_preop = necrosis_preop_volume
        report.statistics.necrosis_volume_postop = necrosis_postop_volume
        report.statistics.cavity_volume_postop = cavity_postop_volume
        report.statistics.brain_volume_preop = preop_brain_volume
        report.statistics.brain_volume_postop = postop_brain_volume
        report.statistics.brain_volume_change =  ((preop_brain_volume - postop_brain_volume) / preop_brain_volume) * 100.
        report.statistics.tumor_to_brain_ratio_preop = (preop_volume / preop_brain_volume) * 100.
        report.statistics.tumor_to_brain_ratio_postop = (postop_volume / postop_brain_volume) * 100.

        if flairchanges_preop_volume and flairchanges_postop_volume:
            eor_flair =  ((flairchanges_preop_volume - flairchanges_postop_volume) / flairchanges_preop_volume) * 100.
            report.statistics.extent_of_resection_flair = eor_flair

        if necrosis_preop_volume and necrosis_postop_volume:
            eor_necro =  ((necrosis_preop_volume - necrosis_postop_volume) / necrosis_preop_volume) * 100.
            report.statistics.necrosis_volume_change = eor_necro

        if eor > 99.9:
            report.statistics.resection_category = ResectionCategoryType.ComR
        elif eor >= 95.0 and postop_volume <= 10.0:
            report.statistics.resection_category = ResectionCategoryType.NeaR
        elif eor >= 80.0 and postop_volume <= 50.0:
            report.statistics.resection_category = ResectionCategoryType.SubR
        else:
            report.statistics.resection_category = ResectionCategoryType.ParR
    except Exception as e:
        raise ValueError(f"Surgical report computation failed with {e}\n")

def compute_acquisition_infos(radiological_volumes: List[str]) -> NeuroAcquisitionInfo:
    """
    @TODO. If multiple scans for a same sequence, only the info for the "best" sequence should be reported. The best
    scan being the one also used for running the segmentation models (e.g., highest resolution/smallest spacing).
    """
    t1c_stats = None
    t1w_stats = None
    t2f_stats = None
    t2w_stats = None
    for v in radiological_volumes:
        rad_vol_nib = nib.load(v.usable_input_filepath)
        stats = NeuroRadiologicalVolumeStatistics(dim_x=rad_vol_nib.shape[0], dim_y=rad_vol_nib.shape[1],
                                                  dim_z=rad_vol_nib.shape[2], spac_x=rad_vol_nib.header.get_zooms()[0],
                                                  spac_y=rad_vol_nib.header.get_zooms()[1],
                                                  spac_z=rad_vol_nib.header.get_zooms()[2])
        if v.get_sequence_type_enum() == MRISequenceType.T1c:
            t1c_stats = stats
        elif v.get_sequence_type_enum() == MRISequenceType.T1w:
            t1w_stats = stats
        elif v.get_sequence_type_enum() == MRISequenceType.FLAIR:
            t2f_stats = stats
        elif v.get_sequence_type_enum() == MRISequenceType.T2:
            t2w_stats = stats
    infos = NeuroAcquisitionInfo(t1c_stats=t1c_stats, t1w_stats=t1w_stats, t2w_stats=t2w_stats, t2f_stats=t2f_stats)
    return infos
