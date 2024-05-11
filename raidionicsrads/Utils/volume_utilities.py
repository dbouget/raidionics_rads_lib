import numpy as np
from copy import deepcopy
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


def __intensity_normalization_CT(volume, parameters):
    result = np.copy(volume)

    result[volume < parameters.intensity_clipping_values[0]] = parameters.intensity_clipping_values[0]
    result[volume > parameters.intensity_clipping_values[1]] = parameters.intensity_clipping_values[1]

    min_val = np.min(result)
    max_val = np.max(result)
    if (max_val - min_val) != 0:
        result = (result - min_val) / (max_val - min_val)

    return result


def __intensity_normalization_MRI(volume, parameters):
    #result = np.zeros(shape=volume.shape)
    #original = np.copy(volume)

    result = deepcopy(volume).astype('float32')
    if parameters.intensity_clipping_range[1] - parameters.intensity_clipping_range[0] != 100:
        limits = np.percentile(volume, q=parameters.intensity_clipping_range)
        result[volume < limits[0]] = limits[0]
        result[volume > limits[1]] = limits[1]

    if parameters.normalization_method == 'zeromean':
        mean_val = np.mean(result)
        var_val = np.std(result)
        tmp = (result - mean_val) / var_val
        result = tmp
    else:
        min_val = np.min(result)
        max_val = np.max(result)
        if (max_val - min_val) != 0:
            tmp = (result - min_val) / (max_val - min_val)
            result = tmp
    # else:
    #     result = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    return result


def intensity_normalization(volume, parameters):
    if parameters.imaging_modality == ImagingModalityType.CT:
        return __intensity_normalization_CT(volume, parameters)
    elif parameters.imaging_modality == ImagingModalityType.MRI:
        return __intensity_normalization_MRI(volume, parameters)


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


def prediction_binary_dilation(prediction: np.ndarray, voxel_volume: float, arg: int) -> np.ndarray:
    """
    Perform iterative dilation over a binary segmentation mask. The dilation process continues until a volume
    increase exceeding the provided arg is reached.\n
    The dilation is not applied over the whole mask, but over each focus after performing a connected component step.

    :param prediction: Binary segmentation mask to dilate
    :param voxel_volume: Size of one volume voxel in cubic ml.
    :param arg: Volume increase percentage to reach for stopping the dilation process.
    :return: Dilated prediction as a numpy array, with same dimensions as the original volume
    """
    # @TODO. Should assert that the prediction file is binary
    res = np.zeros(prediction.shape)

    if np.count_nonzero(prediction) == 0:
        logging.warning("[SegmentationRefinement] Step was skipped - Segmentation file is empty!")
        res = prediction

    # Identifying the different focus, for potential volume-based dilation
    detection_labels = measurements.label(prediction)[0]
    for c in range(1, np.max(detection_labels) + 1):
        focus_img = np.zeros(detection_labels.shape)
        focus_img[detection_labels == c] = 1
        initial_focus_volume_ml = np.count_nonzero(focus_img) * voxel_volume
        kernel = ball(radius=1)
        stop_flag = False
        while not stop_flag:
            ori_focus_img = deepcopy(focus_img)
            focus_img = binary_dilation(focus_img, footprint=kernel)
            focus_volume_ml = np.count_nonzero(focus_img) * voxel_volume
            if ((focus_volume_ml - initial_focus_volume_ml) / initial_focus_volume_ml) * 100 > arg:
                focus_img = ori_focus_img
                stop_flag = True
        seg_dil = focus_img.astype('uint8')
        res[seg_dil == 1] = 1

    return res

def convert_braingrid_to_mni():
    import nibabel as nib
    from nibabel.processing import resample_to_output
    input_folder = '/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/BrainGrid/BrainGrid_voxels'
    output_folder = '/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/BrainGrid/BrainGrid_voxels_correct'
    input_files = []

    for _, _, files in os.walk(input_folder):
        for f in files:
            if ".nii.gz" in f:
                input_files.append(f)
        break

    atlas_ni = nib.load('/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii')
    atlas = atlas_ni.get_fdata()[:]
    for f in input_files:
        fn = os.path.join(input_folder, f)
        region_ni = nib.load(fn)
        region_res_ni = resample_to_output(region_ni, (1,1,1), order=0)
        region_res = region_res_ni.get_fdata()
        final_region_res = np.zeros(atlas.shape)
        final_region_res[22:177, 24:211, 22:157] = region_res
        final_region_res_ni = nib.Nifti1Image(final_region_res, atlas_ni.affine, atlas_ni.header)
        nib.save(final_region_res_ni, os.path.join(output_folder, f))

def convert_braingrid_wm_to_mni():
    import nibabel as nib
    from nibabel.processing import resample_to_output
    input_folder = '/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/BrainGrid/BrainGrid_white_matter_atlas'
    output_folder = '/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/BrainGrid/BrainGrid_white_matter_atlas_correct'
    input_files = []

    for _, _, files in os.walk(input_folder):
        for f in files:
            if ".nii" in f:
                input_files.append(f)
        break

    atlas_ni = nib.load('/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii')
    atlas = atlas_ni.get_fdata()[:]
    for f in input_files:
        try:
            fn = os.path.join(input_folder, f)
            region_ni = nib.load(fn)
            region_res_ni = resample_to_output(region_ni, (1, 1, 1), order=0)
            region_res = region_res_ni.get_fdata()
            final_region_res = np.zeros(atlas.shape)
            final_region_res[22:177, 24:211, 22:157] = np.round(region_res)
            final_region_res_ni = nib.Nifti1Image(np.round(final_region_res).astype('uint8'), atlas_ni.affine, atlas_ni.header)
            nib.save(final_region_res_ni, os.path.join(output_folder, f.replace(' ', '_').replace('.trk', '')))
        except Exception as e:
            print("Can't convert {}".format(f))
def create_braingrid_atlas():
    import nibabel as nib
    input_folder = '/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/BrainGrid/BrainGrid_voxels_correct'
    atlas_ni = nib.load('/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii')
    atlas = atlas_ni.get_fdata()[:]

    braingrid_atlas = np.zeros(atlas.shape).astype('uint8')
    index = 1
    for a in range(1, 5):
        for c in range(1, 4):
            for s in range(1, 5):
                file_fn = os.path.join(input_folder, "A"+str(a)+"C"+str(c)+"S"+str(s)+".nii.gz")
                region_ni = nib.load(file_fn)
                region_res = region_ni.get_fdata()[:]
                braingrid_atlas[region_res > 0.99] = index
                index += 1

    braingrid_atlas_ni = nib.Nifti1Image(braingrid_atlas, atlas_ni.affine, atlas_ni.header)
    nib.save(braingrid_atlas_ni, os.path.join(input_folder, "braingrid_atlas.nii.gz"))

def create_braingrid_whitematter_atlas():
    import nibabel as nib
    import pandas as pd
    input_folder = '/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/BrainGrid/White_matter_atlas'
    atlas_ni = nib.load('/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii')
    atlas = atlas_ni.get_fdata()[:]

    input_files = []

    for _, _, files in os.walk(input_folder):
        for f in files:
            if ".nii.gz" in f:
                input_files.append(f)
        break

    braingrid_atlas = np.zeros(atlas.shape).astype('uint8')
    index = 1
    descriptions = []
    for f in input_files:
        file_fn = os.path.join(input_folder, f)
        region_ni = nib.load(file_fn)
        region_res = region_ni.get_fdata()[:]
        braingrid_atlas[region_res == 1] = index
        descriptions.append([index, f.split('.')[0]])
        index += 1

    braingrid_atlas_ni = nib.Nifti1Image(braingrid_atlas, atlas_ni.affine, atlas_ni.header)
    nib.save(braingrid_atlas_ni, os.path.join(input_folder, "braingrid_white_matter_atlas.nii.gz"))
    descriptions_df = pd.DataFrame(descriptions, columns=['Label', 'Region'])
    descriptions_df.to_csv('/home/dbouget/Documents/Code/Private/raidionics_rads_lib/raidionicsrads/Atlases/BrainGrid/braingrid_subcortical_structures_description.csv', index=False)
