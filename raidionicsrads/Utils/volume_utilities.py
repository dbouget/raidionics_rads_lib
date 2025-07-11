import numpy as np
from copy import deepcopy
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes, measurements
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, ball
from .configuration_parser import *


def crop_MR(volume, parameters):
    original_volume = np.copy(volume)
    volume[volume >= 0.2] = 1
    volume[volume < 0.2] = 0
    volume = volume.astype(np.uint8)
    volume = binary_fill_holes(volume).astype(np.uint8)
    regions = regionprops(volume)
    min_row, min_col, min_depth, max_row, max_col, max_depth = regions[0].bbox
    print('cropping params', min_row, min_col, min_depth, max_row, max_col, max_depth)

    cropped_volume = original_volume[min_row:max_row, min_col:max_col, min_depth:max_depth]
    bbox = [min_row, min_col, min_depth, max_row, max_col, max_depth]

    return cropped_volume, bbox


def resize_volume(volume, new_slice_size, slicing_plane, order=1):
    new_volume = None
    if len(new_slice_size) == 2:
        if slicing_plane == 'axial':
            new_val = int(volume.shape[2] * (new_slice_size[1] / volume.shape[1]))
            new_volume = resize(volume, (new_slice_size[0], new_slice_size[1], new_val), order=order)
        elif slicing_plane == 'sagittal':
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_val, new_slice_size[0], new_slice_size[1]), order=order)
        elif slicing_plane == 'coronal':
            new_val = new_slice_size[0]
            new_volume = resize(volume, (new_slice_size[0], new_val, new_slice_size[1]), order=order)
    elif len(new_slice_size) == 3:
        new_volume = resize(volume, new_slice_size, order=order)
    return new_volume


def padding_for_inference(data, slab_size, slicing_plane):
    new_data = data
    if slicing_plane == 'axial':
        missing_dimension = (slab_size - (data.shape[2] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, 0), (0, missing_dimension), (0, 0)), mode='edge')
    elif slicing_plane == 'sagittal':
        missing_dimension = (slab_size - (data.shape[0] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, missing_dimension), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif slicing_plane == 'coronal':
        missing_dimension = (slab_size - (data.shape[1] % slab_size)) % slab_size
        if missing_dimension != 0:
            new_data = np.pad(data, ((0, 0), (0, missing_dimension), (0, 0), (0, 0)), mode='edge')

    return new_data, missing_dimension


def padding_for_inference_both_ends(data, slab_size, slicing_plane):
    new_data = data
    padding_val = int(slab_size / 2)
    if slicing_plane == 'axial':
        new_data = np.pad(data, ((0, 0), (0, 0), (padding_val, padding_val), (0, 0)), mode='edge')
    elif slicing_plane == 'sagittal':
        new_data = np.pad(data, ((padding_val, padding_val), (0, 0), (0, 0), (0, 0)), mode='edge')
    elif slicing_plane == 'coronal':
        new_data = np.pad(data, ((0, 0), (padding_val, padding_val), (0, 0), (0, 0)), mode='edge')

    return new_data


def volume_masking(volume, mask, output_filename):
    """
    Masks out everything outside.
    :param volume:
    :param mask:
    :param output_filename:
    :return:
    """
    pass


def volume_cropping(volume, mask, output_filename):
    """
    Crops the initial volume with the tighest bounding box around the mask.
    :param volume:
    :param mask:
    :param output_filename:
    :return:
    """
    pass


def prediction_binary_dilation(prediction_filepath: str, arg: int) -> None:
    """
    Perform iterative dilation over a binary segmentation mask. The dilation process continues until a volume
    increase exceeding the provided arg is reached.\n
    The dilation is not applied over the whole mask, but over each focus after performing a connected component step.

    :param prediction_filepath: Filepath where the segmentation mask to dilate is stored
    :param arg: Volume increase percentage to reach for stopping the dilation process.
    :return: Nothing, the dilated volume is saved in place.
    """
    pred_ni = nib.load(prediction_filepath)
    pred = pred_ni.get_fdata()[:].astype('uint8')
    pred_volume_initial = np.count_nonzero(pred) * np.prod(pred_ni.header.get_zooms()[0:3]) * 1e-3
    res = np.zeros(pred.shape, dtype=pred.dtype)

    if np.count_nonzero(pred) == 0:
        logging.warning("[SegmentationRefinement] Step was skipped - Segmentation file is empty!")
        res = pred

    # Identifying the different focus, for potential volume-based dilation
    detection_labels = measurements.label(pred)[0]
    for c in range(1, np.max(detection_labels) + 1):
        focus_img = np.zeros(detection_labels.shape)
        focus_img[detection_labels == c] = 1
        initial_focus_volume_ml = np.count_nonzero(focus_img) * pred_volume_initial
        kernel = ball(radius=1)
        stop_flag = False
        while not stop_flag:
            ori_focus_img = deepcopy(focus_img)
            focus_img = binary_dilation(focus_img, footprint=kernel)
            focus_volume_ml = np.count_nonzero(focus_img) * pred_volume_initial
            if ((focus_volume_ml - initial_focus_volume_ml) / initial_focus_volume_ml) * 100 > arg:
                focus_img = ori_focus_img
                stop_flag = True
        seg_dil = focus_img.astype('uint8')
        res[seg_dil == 1] = 1

    res_ni = nib.Nifti1Image(res.astype('uint8'), affine=pred_ni.affine, header=pred_ni.header)
    nib.save(res_ni, prediction_filepath)
