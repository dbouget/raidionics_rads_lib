import os
import json
import configparser
import logging
import sys
import subprocess
import traceback
import nibabel as nib
import numpy as np
import shutil
import platform


def test_segmentation_pipeline_package(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running segmentation pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "results", "output_seg_package")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'neuro_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', os.path.join(test_dir, "patients",
                                                               "patient-UnitTest1", "inputs"))
        rads_config.set('System', 'output_folder', output_folder)
        rads_config.set('System', 'model_folder', os.path.join(test_dir, "models"))
        rads_config.set('System', 'pipeline_filename', os.path.join(output_folder,
                                                                    'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'thresholding')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config_filename = os.path.join(output_folder, 'rads_config.ini')
        with open(rads_config_filename, 'w') as outfile:
            rads_config.write(outfile)

        pipeline_json = {}
        step_index = 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Classification"
        pipeline_json[step_str]["inputs"] = {}  # Empty input means running it on all existing data for the patient
        pipeline_json[step_str]["target"] = ["MRSequence"]
        pipeline_json[step_str]["model"] = "MRI_SequenceClassifier"
        pipeline_json[step_str]["description"] = "Classification of the MRI sequence type for all input scans."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Segmentation"
        pipeline_json[step_str]["inputs"] = {}
        pipeline_json[step_str]["inputs"]["0"] = {}
        pipeline_json[step_str]["inputs"]["0"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["0"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["0"]["labels"] = None
        pipeline_json[step_str]["inputs"]["0"]["space"] = {}
        pipeline_json[step_str]["inputs"]["0"]["space"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["0"]["space"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["target"] = "Brain"
        pipeline_json[step_str]["model"] = "MRI_Brain"
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Brain segmentation in T1-CE (T0)."

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running segmentation pipeline unit test.\n")
        try:
            from raidionicsrads.compute import run_rads
            run_rads(rads_config_filename)
        except Exception as e:
            raise ValueError(f"Error during the package call with {e}")

        logging.info("Collecting and comparing results.\n")
        segmentation_pred_filename = os.path.join(output_folder, 'T0',
                                                  'input1_annotation-Brain_MRI_Brain.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No brain segmentation mask was generated.\n"
        segmentation_gt_filename = os.path.join(test_dir, "patients", "patient-UnitTest1", "verif",
                                                'T0', 'input1_annotation-Brain.nii.gz')
        segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
        segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
        logging.info(f"Ground truth and prediction arrays difference: {np.count_nonzero(abs(segmentation_gt - segmentation_pred))} pixels")
        # assert np.array_equal(segmentation_pred,
        #                       segmentation_gt), "Ground truth and prediction arrays are not identical"
    except Exception as e:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError(f"Error during segmentation pipeline unit test with {e}\n{traceback.format_exc()}")

    logging.info("Segmentation pipeline unit test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

def test_segmentation_pipeline_cli(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running segmentation pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "results", "output_seg_cli")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'neuro_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', os.path.join(test_dir, "patients",
                                                               "patient-UnitTest1", "inputs"))
        rads_config.set('System', 'output_folder', output_folder)
        rads_config.set('System', 'model_folder', os.path.join(test_dir, "models"))
        rads_config.set('System', 'pipeline_filename', os.path.join(output_folder,
                                                                    'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'thresholding')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config_filename = os.path.join(output_folder, 'rads_config.ini')
        with open(rads_config_filename, 'w') as outfile:
            rads_config.write(outfile)

        pipeline_json = {}
        step_index = 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Classification"
        pipeline_json[step_str]["inputs"] = {}  # Empty input means running it on all existing data for the patient
        pipeline_json[step_str]["target"] = ["MRSequence"]
        pipeline_json[step_str]["model"] = "MRI_SequenceClassifier"
        pipeline_json[step_str]["description"] = "Classification of the MRI sequence type for all input scans."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Segmentation"
        pipeline_json[step_str]["inputs"] = {}
        pipeline_json[step_str]["inputs"]["0"] = {}
        pipeline_json[step_str]["inputs"]["0"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["0"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["0"]["labels"] = None
        pipeline_json[step_str]["inputs"]["0"]["space"] = {}
        pipeline_json[step_str]["inputs"]["0"]["space"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["0"]["space"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["target"] = "Brain"
        pipeline_json[step_str]["model"] = "MRI_Brain"
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Brain segmentation in T1-CE (T0)."

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Standardized reporting CLI unit test started.\n")
        try:
            import platform
            if platform.system() == 'Windows':
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'debug'], shell=True)
            elif platform.system() == 'Darwin' and platform.processor() == 'arm':
                subprocess.check_call(['python3', '-m', 'raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'debug'])
            else:
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'debug'])
        except Exception as e:
            raise ValueError(f"Error during segmentation pipeline CLI test with {e}\n")

        logging.info("Collecting and comparing results.\n")
        segmentation_pred_filename = os.path.join(output_folder, 'T0', 'input1_annotation-Brain_MRI_Brain.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No brain segmentation mask was generated.\n"
        segmentation_gt_filename = os.path.join(test_dir, "patients", "patient-UnitTest1", "verif",
                                                'T0', 'input1_annotation-Brain.nii.gz')
        segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
        segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
        logging.info(
            f"Ground truth and prediction arrays difference: {np.count_nonzero(abs(segmentation_gt - segmentation_pred))} pixels")
        # assert np.array_equal(segmentation_pred,
        #                       segmentation_gt), "Ground truth and prediction arrays are not identical"
    except Exception as e:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError(f"Error during segmentation pipeline unit test with {e}\n{traceback.format_exc()}")

    logging.info("Segmentation pipeline CLI unit test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test_segmentation_pipeline_package_mediastinum(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "results", "output_seg_package_medi")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'mediastinum_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', os.path.join(test_dir, "patients",
                                                               "patient-UnitTest3-Mediastinum", "inputs"))
        rads_config.set('System', 'output_folder', output_folder)
        rads_config.set('System', 'model_folder', os.path.join(test_dir, "models"))
        rads_config.set('System', 'pipeline_filename', os.path.join(output_folder,
                                                                    'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'thresholding')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config_filename = os.path.join(output_folder, 'rads_config.ini')
        with open(rads_config_filename, 'w') as outfile:
            rads_config.write(outfile)

        pipeline_json = {}
        step_index = 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'CT_Tumor'
        pipeline_json[step_str]["timestamp"] = 0
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best tumor segmentation model for existing inputs"

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        lungs_mask_fn = os.path.join(test_dir, "patients", "patient-UnitTest3-Mediastinum", "inputs", "T0",
                                                "1_CT_HR_label-lungs.nii.gz")
        logging.debug(f"Lungs mask already existing at: {lungs_mask_fn}")
        if os.path.exists(lungs_mask_fn):
            os.remove(lungs_mask_fn)
        logging.info("Running segmentation pipeline unit test.\n")
        try:
            from raidionicsrads.compute import run_rads
            run_rads(rads_config_filename)
        except Exception as e:
            raise ValueError(f"Error during the package call with {e}")

        logging.info("Collecting and comparing results.\n")
        segmentation_pred_filename = os.path.join(output_folder, 'T0',
                                                  '1_CT_HR_annotation-Tumor.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No tumor segmentation mask was generated.\n"
        segmentation_gt_filename = os.path.join(test_dir, "patients", "patient-UnitTest3-Mediastinum", "verif", "T0",
                                                "1_CT_HR_labels-Tumor.nii.gz")
        segmentation_pred_nib = nib.load(segmentation_pred_filename)
        segmentation_gt_nib = nib.load(segmentation_gt_filename)
        pred_volume = np.count_nonzero(segmentation_pred_nib.get_fdata()[:]) * np.prod(
            segmentation_pred_nib.header.get_zooms()[0:3]) * 1e-3
        gt_volume = np.count_nonzero(segmentation_gt_nib.get_fdata()[:]) * np.prod(
            segmentation_gt_nib.header.get_zooms()[0:3]) * 1e-3
        logging.info(f"Volume difference: {abs(pred_volume - gt_volume)}\n")
        assert abs(pred_volume - gt_volume) < 1., \
            "Ground truth and prediction arrays are very different"
    except Exception as e:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError(f"Error during segmentation pipeline unit test with {e}\n{traceback.format_exc()}")

    logging.info("Segmentation pipeline unit test succeeded.\n")
    # if os.path.exists(output_folder):
    #     shutil.rmtree(output_folder)