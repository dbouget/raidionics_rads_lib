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


def test_segmentation_pipeline_docker(test_dir, tmp_path):
    """
    Testing the CLI within a Docker container for the segmentation pipeline unit test, running on CPU.
    The latest Docker image is being hosted at: dbouget/raidionics-rads:v1.3.1-py39-cpu

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Preparing configuration file.\n")
    try:
        image_name = "dbouget/raidionics-rads:v1.3.1-py39-cpu"
        if os.environ.get("GITHUB_ACTIONS"):
            image_name = "dbouget/raidionics-rads:" + os.environ["IMAGE_TAG"]

        output_folder = os.path.join(tmp_path, "results")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        test_raw_input_fn = os.path.join(test_dir, "patients")
        tmp_test_input_fn = os.path.join(tmp_path, "patients")
        if os.path.exists(tmp_test_input_fn):
            shutil.rmtree(tmp_test_input_fn)
        shutil.copytree(test_raw_input_fn, tmp_test_input_fn)
        test_raw_models_fn = os.path.join(test_dir, "models")
        tmp_test_model_fn = os.path.join(tmp_path, "models")
        if os.path.exists(tmp_test_model_fn):
            shutil.rmtree(tmp_test_model_fn)
        shutil.copytree(test_raw_models_fn, tmp_test_model_fn)

        using_skull_stripped_inputs = False
        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'neuro_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', '/workspace/resources/patients/patient-UnitTest1/inputs')
        rads_config.set('System', 'output_folder',  '/workspace/resources/results')
        rads_config.set('System', 'model_folder', '/workspace/resources/models')
        rads_config.set('System', 'pipeline_filename', '/workspace/resources/results/test_pipeline.json')
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'thresholding')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config.set('Runtime', 'use_stripped_data', 'True' if using_skull_stripped_inputs else 'False')
        rads_config_filename = os.path.join(tmp_path, "results", 'rads_config.ini')
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
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'MRI_Brain'
        pipeline_json[step_str]["timestamp"] = 0
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best brain segmentation model for existing inputs"

        with open(os.path.join(tmp_path, "results", 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running segmentation pipeline unit test in Docker container.\n")
        try:
            import platform
            cmd_docker = ['docker', 'run', '-v', '{}:/workspace/resources'.format(tmp_path),
                          '--network=host', '--ipc=host']
            if not os.environ.get("GITHUB_ACTIONS") and sys.platform != "win32":
                cmd_docker.extend(['--user', str(os.geteuid())])
            elif os.environ.get("GITHUB_ACTIONS"):
                cmd_docker.extend(['-u',f"{os.getuid()}:{os.getgid()}"])
            cmd_docker.extend([image_name, '-c', '/workspace/resources/results/rads_config.ini', '-v', 'debug'])
            logging.info("Executing the following Docker call: {}".format(cmd_docker))
            if platform.system() == 'Windows':
                subprocess.check_call(cmd_docker, shell=True)
            else:
                subprocess.check_call(cmd_docker, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            raise ValueError(f"{e}")

        logging.info("Collecting and comparing results.\n")
        segmentation_pred_filename = os.path.join(tmp_path, "results", 'T0',
                                                  'input1_annotation-Brain_MRI_Brain.nii.gz')
        assert os.path.exists(segmentation_pred_filename), "No brain segmentation mask was generated.\n"
        segmentation_gt_filename = os.path.join(tmp_path, "patients", "patient-UnitTest1", "verif",
                                                'T0', 'input1_annotation-Brain.nii.gz')
        segmentation_pred = nib.load(segmentation_pred_filename).get_fdata()[:]
        segmentation_gt = nib.load(segmentation_gt_filename).get_fdata()[:]
        assert np.array_equal(segmentation_pred,
                              segmentation_gt), "Ground truth and prediction arrays are not identical"
    except Exception as e:
        logging.error(f"Error during segmentation pipeline unit test in Docker container with: {e}\n {traceback.format_exc()}.\n")
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        raise ValueError("Error during segmentation pipeline unit test in Docker container.\n")

    logging.info("Segmentation pipeline unit test in Docker container succeeded.\n")
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)