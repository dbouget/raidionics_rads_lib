import configparser
import logging
import math
import traceback
from pathlib import PurePath
import numpy as np
import sys, os, shutil
import scipy.ndimage.morphology as smo
import nibabel as nib
import subprocess
from typing import List
from skimage import measure
from scipy.ndimage.measurements import label, find_objects
from skimage.measure import regionprops
from scipy.ndimage.measurements import center_of_mass

from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType
from ..Utils.io import load_nifti_volume
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.segmentation_parser import collect_segmentation_model_parameters


def perform_brain_extraction(image_filepath: str, method: str = 'deep_learning') -> str:
    """
    @DEPRECATED? The brain segmentation is performed as a stand-alone pipeline step
    The brain extraction process.

    Parameters
    ----------
    image_filepath : str
        Filepath of the patient input MRI volume.
    method : str
        Skull stripping method to use to choose from [ants, deep_learning]. In ants mode the ANTs library is used
        to perform the task, and in deep_learning mode a custom brain segmentation model is used.
        AT THE TIME, ONLY THE deep_learning MODE IS IMPLEMENTED AND AVAILABLE!
    Returns
    -------
    str
        Full filepath of the newly created brain mask.
    """
    # Creating temporary folder to delete when all is done
    tmp_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, 'tmp')
    os.makedirs(tmp_folder, exist_ok=True)

    brain_predictions_file = None
    if method == 'deep_learning':
        brain_predictions_file = perform_custom_brain_extraction(image_filepath, tmp_folder)
    else:
        pass

    return brain_predictions_file


def perform_custom_brain_extraction(image_filepath: str, folder: str) -> str:
    """
    @DEPRECATED?
    The custom brain segmentation is performed by using the pre-trained model followed by skull-stripping.

    Parameters
    ----------
    image_filepath : str
        Filepath of the patient input MRI volume.
    folder : str
        Destination folder in which the brain mask will be saved.
    Returns
    -------
    str
        Full filepath of the newly created brain mask.
    """
    brain_config_filename = ''
    dump_brain_mask_filepath = ''
    try:
        brain_config = configparser.ConfigParser()
        brain_config.add_section('System')
        brain_config.set('System', 'gpu_id', ResourcesConfiguration.getInstance().gpu_id)
        brain_config.set('System', 'input_filename', image_filepath)
        brain_config.set('System', 'output_folder', ResourcesConfiguration.getInstance().output_folder)
        brain_config.set('System', 'model_folder', os.path.join(os.path.dirname(ResourcesConfiguration.getInstance().model_folder), 'MRI_Brain'))
        brain_config.add_section('Runtime')
        brain_config.set('Runtime', 'reconstruction_method', 'thresholding')
        brain_config.set('Runtime', 'reconstruction_order', 'resample_first')
        brain_config_filename = os.path.join(os.path.dirname(ResourcesConfiguration.getInstance().config_filename), 'brain_config.ini')
        with open(brain_config_filename, 'w') as outfile:
            brain_config.write(outfile)

        log_level = logging.getLogger().level
        log_str = 'warning'
        if log_level == 10:
            log_str = 'debug'
        elif log_level == 20:
            log_str = 'info'
        elif log_level == 40:
            log_str = 'error'

        # if os.name == 'nt':
        #     script_path_parts = list(PurePath(os.path.realpath(__file__)).parts[:-3] + ('raidionics_seg_lib', 'main.py',))
        #     script_path = PurePath()
        #     for x in script_path_parts:
        #         script_path = script_path.joinpath(x)
        #     subprocess.check_call([sys.executable, '{script}'.format(script=script_path), '-c',
        #                      '{config}'.format(config=brain_config_filename), '-v', log_str])
        # else:
        #     script_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-2]) + '/raidionics_seg_lib/main.py'
        #     subprocess.check_call(['python3', '{script}'.format(script=script_path), '-c',
        #                      '{config}'.format(config=brain_config_filename), '-v', log_str])
        from raidionicsseg.fit import run_model
        run_model(brain_config_filename)
    except Exception as e:
        logging.error("Automatic brain segmentation failed with: {}.\n".format(traceback.format_exc()))
        if os.path.exists(brain_config_filename):
            os.remove(brain_config_filename)
        raise ValueError("Impossible to perform automatic brain segmentation.\n")

    try:
        brain_mask_filename = os.path.join(ResourcesConfiguration.getInstance().output_folder, 'labels_Brain.nii.gz')
        brain_mask_ni = load_nifti_volume(brain_mask_filename)
        brain_mask = brain_mask_ni.get_fdata()[:].astype('uint8')

        # The automatic segmentation should be clean, but just in case, only the largest component is retained.
        labels, nb_components = label(brain_mask)
        brain_objects_properties = regionprops(labels)
        brain_object = brain_objects_properties[0]
        brain_component = np.zeros(brain_mask.shape).astype('uint8')
        brain_component[brain_object.bbox[0]:brain_object.bbox[3],
        brain_object.bbox[1]:brain_object.bbox[4],
        brain_object.bbox[2]:brain_object.bbox[5]] = 1

        dump_brain_mask = brain_mask & brain_component
        dump_brain_mask_ni = nib.Nifti1Image(dump_brain_mask, affine=brain_mask_ni.affine)
        if os.name == 'nt':
            path_parts = list(PurePath(os.path.realpath(folder)).parts[:-1] + ('input_brain_mask.nii.gz',))
            dump_brain_mask_filepath = PurePath()
            for x in path_parts:
                dump_brain_mask_filepath = dump_brain_mask_filepath.joinpath(x)
            dump_brain_mask_filepath = str(dump_brain_mask_filepath)
        else:
            dump_brain_mask_filepath = os.path.join('/'.join(folder.split('/')[:-1]), 'input_brain_mask.nii.gz')
        nib.save(dump_brain_mask_ni, dump_brain_mask_filepath)
        os.remove(brain_config_filename)
    except Exception as e:
        logging.error("Skull stripping operation failed with: {}.\n".format(traceback.format_exc()))
        if os.path.exists(brain_config_filename):
            os.remove(brain_config_filename)
        raise ValueError("Impossible to perform skull stripping.\n")

    return dump_brain_mask_filepath


def perform_brain_masking(image_filepath, mask_filepath, output_folder):
    """
    Set to 0 any voxel that does not belong to the brain mask.
    :param image_filepath:
    :param mask_filepath:
    :return: masked_image_filepath
    """
    os.makedirs(output_folder, exist_ok=True)
    image_ni = load_nifti_volume(image_filepath)
    brain_mask_ni = load_nifti_volume(mask_filepath)

    image = image_ni.get_fdata()[:]
    brain_mask = brain_mask_ni.get_fdata()[:]
    image[brain_mask == 0] = 0

    masked_input_filepath = os.path.join(output_folder, os.path.basename(image_filepath).split('.')[0] + '_masked.nii.gz')
    nib.save(nib.Nifti1Image(image, affine=image_ni.affine), masked_input_filepath)
    return masked_input_filepath


def perform_brain_clipping(image_filepath, mask_filepath):
    """
    Identify the tighest bounding box around the brain mask and set to 0 any voxel outside that bounding box.
    :param image_filepath:
    :param mask_filepath:
    :return: masked_image_filepath
    """
    pass


def perform_brain_overlap_refinement(predictions_filepath: str, brain_mask_filepath: str):
    """
    In-place refinement of the predictions.

    Parameters
    ----------
    predictions_filepath
    brain_mask_filepath

    Returns
    -------

    """
    try:
        pred_nib = nib.load(predictions_filepath)
        brain_mask_nib = nib.load(brain_mask_filepath)
        pred = pred_nib.get_fdata()[:]
        brain_mask = brain_mask_nib.get_fdata()[:].astype('uint8')

        pred_binary = np.zeros(pred.shape, dtype='uint8')
        pred_binary[pred > 1e-3] = 1
        cc_pred_bin = measure.label(pred_binary)
        obj_labels = np.unique(cc_pred_bin)[1:]
        final_pred = np.zeros(pred.shape, dtype='float32')
        for l in range(0, len(obj_labels)):
            obj_pred = np.zeros(pred_binary.shape, dtype='uint8')
            obj_pred[cc_pred_bin == (l + 1)] = 1
            overlap = np.count_nonzero(obj_pred & brain_mask) > 0
            if overlap:
                label_pred = np.where(cc_pred_bin == (l + 1), pred, 0).astype("float32")
                final_pred = final_pred + label_pred
        final_pred_nib = nib.Nifti1Image(final_pred, affine=pred_nib.affine, header=pred_nib.header)
        nib.save(final_pred_nib, predictions_filepath)
    except Exception as e:
        raise ValueError("Brain overlap refinement failed with: {}.".format(e))

def perform_segmentation_global_consistency_refinement(annotation_files: dict, timestamp: str,
                                                       tumor_general_type: str = "contrast-enhancing")  -> dict:
    """
    @TODO. Should also ship the preop segmentation for postop refinement (e.g. preop necrosis still in postop),
    but then the preop T1 and postop T1 should be co-registered...
    """
    tumorcore_anno_fn = None
    tumorcore_anno_nib = None
    tumorcore_anno = None
    tumor_ce_anno_fn = None
    tumor_ce_anno_nib = None
    tumor_ce_anno = None
    cavity_anno_fn = None
    cavity_anno_nib = None
    cavity_anno = None
    flair_changes_anno_fn = None
    flair_changes_anno_nib = None
    flair_changes_anno = None
    necrosis_anno_fn = None
    necrosis_anno_nib = None
    necrosis_anno = None
    output_annotation_files = {}

    for a in list(annotation_files.keys()):
        if a == str(AnnotationClassType.Tumor):
            tumorcore_anno_fn = annotation_files[a]
            tumorcore_anno_nib = nib.load(tumorcore_anno_fn)
            tumorcore_anno = tumorcore_anno_nib.get_fdata()[:]
        elif a == str(AnnotationClassType.Cavity):
            cavity_anno_fn = annotation_files[a]
            cavity_anno_nib = nib.load(cavity_anno_fn)
            cavity_anno = cavity_anno_nib.get_fdata()[:]
        elif a == str(AnnotationClassType.TumorCE):
            tumor_ce_anno_fn = annotation_files[a]
            tumor_ce_anno_nib = nib.load(tumor_ce_anno_fn)
            tumor_ce_anno = tumor_ce_anno_nib.get_fdata()[:]
        elif a == str(AnnotationClassType.FLAIRChanges):
            flair_changes_anno_fn = annotation_files[a]
            flair_changes_anno_nib = nib.load(flair_changes_anno_fn)
            flair_changes_anno = flair_changes_anno_nib.get_fdata()[:]
        elif a == str(AnnotationClassType.Necrosis):
            necrosis_anno_fn = annotation_files[a]
            necrosis_anno_nib = nib.load(necrosis_anno_fn)
            necrosis_anno = necrosis_anno_nib.get_fdata()[:]

    if timestamp == 1:
        if tumor_general_type == "contrast-enhancing":
            if cavity_anno is not None and necrosis_anno is not None:
                # High chances that both segmented the same thing, would have to clean up!
                # Disabling necrosis segmention in postop for the time being, as the model was trained on preop only.
                pass

            if cavity_anno is not None and flair_changes_anno is not None:
                refined_tumorce = np.zeros(tumor_ce_anno.shape).astype("uint8")
                refined_tumorce[(tumor_ce_anno != 0) & (cavity_anno == 0)] = 1
                nib.save(nib.Nifti1Image(refined_tumorce, affine=tumor_ce_anno_nib.affine, header=tumor_ce_anno_nib.header),
                         tumor_ce_anno_fn)

                output_annotation_files[AnnotationClassType.TumorCE] = tumor_ce_anno_fn
                new_flair_changes = np.zeros(flair_changes_anno.shape).astype('uint8')
                new_flair_changes[flair_changes_anno == 1] = 1
                new_flair_changes[tumor_ce_anno == 1] = 1
                new_flair_changes[cavity_anno == 1] = 0
                nib.save(nib.Nifti1Image(new_flair_changes, flair_changes_anno_nib.affine, flair_changes_anno_nib.header),
                         flair_changes_anno_fn)
                output_annotation_files[AnnotationClassType.FLAIRChanges] = flair_changes_anno_fn

                new_cavity = np.zeros(cavity_anno.shape).astype('uint8')
                new_cavity[flair_changes_anno == 1] = 1
                # combined_anno = np.zeros(tumor_ce_anno.shape).astype("uint8")
                # combined_anno[flair_changes_anno == 1] = 1
                # combined_anno[cavity_anno == 1] = 2
                # combined_anno[refined_tumorce == 1] = 3
                # # @TODO. Saving the combined file in case?
        elif tumor_general_type == "non contrast-enhancing":
            if cavity_anno is not None:
                refined_flairchanges = np.zeros(flair_changes_anno.shape).astype("uint8")
                refined_flairchanges[flair_changes_anno == 1] = 1
                refined_flairchanges[cavity_anno == 1] = 0
                nib.save(
                    nib.Nifti1Image(refined_flairchanges, affine=flair_changes_anno_nib.affine,
                                    header=flair_changes_anno_nib.header), flair_changes_anno_fn)
    elif timestamp == 0:
        if tumor_general_type == "contrast-enhancing":
            if cavity_anno is None and flair_changes_anno is None and tumor_ce_anno is None and necrosis_anno is None:
                # Only the preop tumor core is available, no context refinement can be performed for a ce tumor.
                output_annotation_files[AnnotationClassType.Tumor] = tumorcore_anno_fn
            elif cavity_anno is None and flair_changes_anno is not None and tumor_ce_anno is None and necrosis_anno is None:
                # Only the preop tumor is available, no context refinement can be performed for a non-ce tumor.
                output_annotation_files[AnnotationClassType.FLAIRChanges] = flair_changes_anno_fn
            elif (flair_changes_anno is not None and cavity_anno is None and tumor_ce_anno is None and necrosis_anno is None
                  and tumorcore_anno is not None):
                # Should the FLAIR changes just be on the outskirt of the tumor core (at least for contrast-enhancing tumors)?
                new_flair_changes = np.zeros(flair_changes_anno.shape).astype('uint8')
                new_flair_changes[flair_changes_anno == 1] = 1
                new_flair_changes[tumorcore_anno == 1] = 0
                nib.save(nib.Nifti1Image(new_flair_changes, flair_changes_anno_nib.affine, flair_changes_anno_nib.header),
                         flair_changes_anno_fn)
                output_annotation_files[AnnotationClassType.Tumor] = tumorcore_anno_fn
                output_annotation_files[AnnotationClassType.FLAIRChanges] = flair_changes_anno_fn
            elif flair_changes_anno is None and cavity_anno is None and tumor_ce_anno is None and necrosis_anno is not None:
                new_necrosis = np.zeros(necrosis_anno.shape).astype('uint8')
                new_necrosis[(tumorcore_anno == 1) & (necrosis_anno == 1)] = 1
                nib.save(nib.Nifti1Image(new_necrosis, necrosis_anno_nib.affine, necrosis_anno_nib.header),
                         necrosis_anno_fn)
                new_et = np.zeros(necrosis_anno.shape).astype('uint8')
                new_et[(tumorcore_anno == 1) & (necrosis_anno == 0)] = 1
                new_et_fn = os.path.join(os.path.dirname(necrosis_anno_fn), os.path.basename(necrosis_anno_fn.split('_annotation')[0]) + '_annotation-TumorCE.nii.gz')
                nib.save(nib.Nifti1Image(new_et, necrosis_anno_nib.affine, necrosis_anno_nib.header),
                         new_et_fn)
                output_annotation_files[AnnotationClassType.Tumor] = tumorcore_anno_fn
                output_annotation_files[AnnotationClassType.Necrosis] = necrosis_anno_fn
                output_annotation_files[AnnotationClassType.TumorCE] = new_et_fn
            elif flair_changes_anno is not None and cavity_anno is None and tumor_ce_anno is None and necrosis_anno is not None:
                new_necrosis = np.zeros(necrosis_anno.shape).astype('uint8')
                new_necrosis[(tumorcore_anno == 1) & (necrosis_anno == 1)] = 1
                nib.save(nib.Nifti1Image(new_necrosis, necrosis_anno_nib.affine, necrosis_anno_nib.header),
                         necrosis_anno_fn)
                new_flair_changes = np.zeros(flair_changes_anno.shape).astype('uint8')
                new_flair_changes[flair_changes_anno == 1] = 1
                new_flair_changes[tumorcore_anno == 1] = 1
                nib.save(nib.Nifti1Image(new_flair_changes, flair_changes_anno_nib.affine, flair_changes_anno_nib.header),
                         flair_changes_anno_fn)
                new_edema = np.zeros(necrosis_anno.shape).astype('uint8')
                new_edema[flair_changes_anno == 1] = 1
                new_edema[tumorcore_anno == 1] = 0
                new_edema_fn = os.path.join(os.path.dirname(necrosis_anno_fn), os.path.basename(necrosis_anno_fn.split('_annotation')[0]) + '_annotation-Edema.nii.gz')
                nib.save(nib.Nifti1Image(new_edema, necrosis_anno_nib.affine, necrosis_anno_nib.header),
                         new_edema_fn)
                output_annotation_files[AnnotationClassType.Tumor] = tumorcore_anno_fn
                output_annotation_files[AnnotationClassType.Necrosis] = necrosis_anno_fn
                output_annotation_files[AnnotationClassType.FLAIRChanges] = flair_changes_anno_fn
                output_annotation_files[AnnotationClassType.Edema] = new_edema_fn
            else:
                pass
        elif tumor_general_type == "non contrast-enhancing":
            if cavity_anno is None and tumorcore_anno is None and tumor_ce_anno is None and necrosis_anno is None:
                # Only the preop FLAIR changes is available, no context refinement can be performed for a non-ce tumor.
                output_annotation_files[AnnotationClassType.FLAIRChanges] = flair_changes_anno

    return output_annotation_files


def select_candidates(input_array):
    """
    Perform a connected components analysis to identify the stand-alone objects in both the ground truth and
    binarized prediction volumes. Objects with a number of voxels below the limit set in self.tiny_objects_removal_threshold
    are discarded, in both instances. Safe way to handle potential noise in the ground truth, especially if a
    third-party software (e.g. 3DSlicer) was used.
    """
    from scipy.ndimage import measurements
    from skimage.measure import regionprops
    from copy import deepcopy

    if input_array is None:
        return None, None

    # Cleaning the too small objects that might be noise in the ground truth
    labels = measurements.label(input_array)[0]
    refined_image = deepcopy(labels)
    for c in range(1, np.max(labels)+1):
        if np.count_nonzero(labels == c) < 50:
            refined_image[refined_image == c] = 0
    refined_image[refined_image != 0] = 1
    labels = measurements.label(refined_image)[0]
    candidates = regionprops(labels)

    return labels, candidates

def compute_dice(volume1, volume2):
    dice = 0.
    if np.sum(volume1[volume2 == 1]) != 0:
        dice = (np.sum(volume1[volume2 == 1]) * 2.0) / (np.sum(volume1) + np.sum(volume2))
    return dice


import numpy as np


def compute_3d_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two 3D bounding boxes.

    box1, box2: (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    # Compute intersection box
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    z_min_inter = max(box1[2], box2[2])
    x_max_inter = min(box1[3], box2[3])
    y_max_inter = min(box1[4], box2[4])
    z_max_inter = min(box1[5], box2[5])

    # Compute intersection volume
    inter_dim_x = max(0, x_max_inter - x_min_inter)
    inter_dim_y = max(0, y_max_inter - y_min_inter)
    inter_dim_z = max(0, z_max_inter - z_min_inter)
    intersection_volume = inter_dim_x * inter_dim_y * inter_dim_z

    # Compute volume of each bounding box
    vol_box1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol_box2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

    # Compute union volume
    union_volume = vol_box1 + vol_box2 - intersection_volume

    # Compute IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0
    return iou
