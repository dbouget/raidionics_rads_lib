import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import nibabel as nib
import numpy as np


def test_postoperative_segmentation_pipeline_package(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running postoperative segmentation pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "results", "output_package")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        using_skull_stripped_inputs = True

        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'neuro_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', os.path.join(test_dir, "patients",
                                                               'patient-UnitTest2', "inputs"))
        rads_config.set('System', 'output_folder', output_folder)
        rads_config.set('System', 'model_folder', os.path.join(test_dir, "models"))
        rads_config.set('System', 'pipeline_filename', os.path.join(output_folder, 'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'probabilities')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config.set('Runtime', 'use_stripped_data', 'True' if using_skull_stripped_inputs else 'False')
        rads_config_filename = os.path.join(output_folder, 'rads_config.ini')
        with open(rads_config_filename, 'w') as outfile:
            rads_config.write(outfile)

        # Prepare the underlying pipeline
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
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'MRI_TumorCE_Postop'
        pipeline_json[step_str]["timestamp"] = 1
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best rest enhancing tumor segmentation model for existing inputs"

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4)

        logging.info("Running segmentation pipeline unit test.\n")
        from raidionicsrads.compute import run_rads
        run_rads(rads_config_filename)

        logging.info("Collecting and comparing results.\n")
        segmentation_pred_filename = os.path.join(output_folder, 'T1',
                                                  'postop_t1gd_annotation-TumorCE_MRI_TumorCE_Postop.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No tumor segmentation mask was generated.\n"
        segmentation_gt_filename = os.path.join(test_dir, "patients", "patient-UnitTest2", "verif",
                                                'T1', 'postop_t1gd_annotation-TumorCE.nii.gz')
        segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
        segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
        assert np.count_nonzero(abs(segmentation_pred - segmentation_gt)) < 200, \
            "Ground truth and prediction arrays are very different"
    except Exception as e:
        logging.error(f"Error during segmentation pipeline unit test with: {e}\n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError("Error during segmentation pipeline unit test with.\n")

    logging.info("Postoperative segmentation pipeline package test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

def test_postoperative_segmentation_pipeline_cli(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running postoperative segmentation pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "results", "output_cli")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        using_skull_stripped_inputs = True

        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'neuro_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', os.path.join(test_dir, "patients",
                                                               'patient-UnitTest2', "inputs"))
        rads_config.set('System', 'output_folder', output_folder)
        rads_config.set('System', 'model_folder', os.path.join(test_dir, "models"))
        rads_config.set('System', 'pipeline_filename', os.path.join(output_folder,
                                                                    'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'probabilities')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config.set('Runtime', 'use_stripped_data', 'True' if using_skull_stripped_inputs else 'False')
        rads_config_filename = os.path.join(output_folder, 'rads_config.ini')
        with open(rads_config_filename, 'w') as outfile:
            rads_config.write(outfile)

        # Prepare the underlying pipeline
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
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'MRI_TumorCE_Postop'
        pipeline_json[step_str]["timestamp"] = 1
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best rest enhancing tumor segmentation model for existing inputs"

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4)

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
        segmentation_pred_filename = os.path.join(output_folder, 'T1',
                                                  'postop_t1gd_annotation-TumorCE_MRI_TumorCE_Postop.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No tumor segmentation mask was generated.\n"
        segmentation_gt_filename = os.path.join(test_dir, "patients", "patient-UnitTest2", "verif",
                                                'T1', 'postop_t1gd_annotation-TumorCE.nii.gz')
        segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
        segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
        assert np.count_nonzero(abs(segmentation_pred - segmentation_gt)) < 200, \
            "Ground truth and prediction arrays are very different"
    except Exception as e:
        logging.error(f"Error during segmentation pipeline unit test with: {e}\n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError("Error during segmentation pipeline unit test with.\n")

    logging.info("Postoperative segmentation pipeline CLI test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)