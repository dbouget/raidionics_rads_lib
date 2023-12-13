import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile

try:
    import requests
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'requests'])
    import requests


def postoperative_segmentation_pipeline_test():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running segmentation pipeline unit test.\n")
    logging.info("Downloading unit test resources.\n")
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    patient_dir = os.path.join(test_dir, 'patients')
    os.makedirs(patient_dir)
    output_dir = os.path.join(test_dir, 'results')
    os.makedirs(output_dir)
    models_dir = os.path.join(test_dir, 'models')
    os.makedirs(models_dir)

    try:
        test_patient_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsRADSLib-UnitTest2.zip'
        brain_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Raidionics-MRI_Brain-ONNX-v12.zip'
        gbm_preop_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Raidionics-MRI_GBM-ONNX-v12.zip'
        gbm_postop_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Raidionics-MRI_GBM_Postop_FV_5p-ONNX_v12.zip'

        archive_dl_dest = os.path.join(test_dir, 'inference_patient.zip')
        if not os.path.exists(archive_dl_dest):
            headers = {}
            response = requests.get(test_patient_url, headers=headers, stream=True)
            response.raise_for_status()
            if response.status_code == requests.codes.ok:
                with open(archive_dl_dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1048576):
                        f.write(chunk)
            with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
                zip_ref.extractall(patient_dir)

        archive_dl_dest = os.path.join(test_dir, 'brain-model.zip')
        if not os.path.exists(archive_dl_dest):
            headers = {}
            response = requests.get(brain_model_url, headers=headers, stream=True)
            response.raise_for_status()
            if response.status_code == requests.codes.ok:
                with open(archive_dl_dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1048576):
                        f.write(chunk)
            with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
                zip_ref.extractall(models_dir)

        archive_dl_dest = os.path.join(test_dir, 'gbm_preop-model.zip')
        if not os.path.exists(archive_dl_dest):
            headers = {}
            response = requests.get(gbm_preop_model_url, headers=headers, stream=True)
            response.raise_for_status()
            if response.status_code == requests.codes.ok:
                with open(archive_dl_dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1048576):
                        f.write(chunk)
            with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
                zip_ref.extractall(models_dir)

        archive_dl_dest = os.path.join(test_dir, 'gbm_postop-model.zip')
        if not os.path.exists(archive_dl_dest):
            headers = {}
            response = requests.get(gbm_postop_model_url, headers=headers, stream=True)
            response.raise_for_status()
            if response.status_code == requests.codes.ok:
                with open(archive_dl_dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1048576):
                        f.write(chunk)
            with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
    except Exception as e:
        logging.error("Error during resources download with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during resources download.\n")

    logging.info("Preparing configuration file.\n")
    try:
        using_skull_stripped_inputs = True

        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'neuro_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', os.path.join(patient_dir, 'patient-UnitTest2'))
        rads_config.set('System', 'output_folder', output_dir)
        rads_config.set('System', 'model_folder', models_dir)
        rads_config.set('System', 'pipeline_filename', os.path.join(test_dir, 'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'probabilities')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config.set('Runtime', 'use_stripped_data', 'True' if using_skull_stripped_inputs else 'False')
        rads_config_filename = os.path.join(output_dir, 'rads_config.ini')
        with open(rads_config_filename, 'w') as outfile:
            rads_config.write(outfile)

        # Prepare the underlying pipeline
        pipeline_json = {}
        step_index = 1
        step_str = str(step_index)
        if not using_skull_stripped_inputs:
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
        pipeline_json[step_str]["target"] = "Tumor"
        pipeline_json[step_str]["model"] = "MRI_GBM"
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Tumor segmentation in T1-CE (T0)."

        if not using_skull_stripped_inputs:
            step_index = step_index + 1
            step_str = str(step_index)
            pipeline_json[step_str] = {}
            pipeline_json[step_str]["task"] = "Segmentation"
            pipeline_json[step_str]["inputs"] = {}
            pipeline_json[step_str]["inputs"]["0"] = {}
            pipeline_json[step_str]["inputs"]["0"]["timestamp"] = 1
            pipeline_json[step_str]["inputs"]["0"]["sequence"] = "T1-CE"
            pipeline_json[step_str]["inputs"]["0"]["labels"] = None
            pipeline_json[step_str]["inputs"]["0"]["space"] = {}
            pipeline_json[step_str]["inputs"]["0"]["space"]["timestamp"] = 1
            pipeline_json[step_str]["inputs"]["0"]["space"]["sequence"] = "T1-CE"
            pipeline_json[step_str]["target"] = "Brain"
            pipeline_json[step_str]["model"] = "MRI_Brain"
            pipeline_json[step_str]["format"] = "thresholding"
            pipeline_json[step_str]["description"] = "Brain segmentation in T1-CE (T1)."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = 1
        pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["description"] = "Registration from T1-CE (T0) to T1-CE (T1)"

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Apply registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = 1
        pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["direction"] = "forward"
        pipeline_json[step_str]["description"] = "Apply registration from T1-CE (T0) to T1-CE (T1)"

        additional_postop_sequences = ['T1-w', 'FLAIR']
        for seq in additional_postop_sequences:
            if not using_skull_stripped_inputs:
                step_index = step_index + 1
                step_str = str(step_index)
                pipeline_json[step_str] = {}
                pipeline_json[step_str]["task"] = "Segmentation"
                pipeline_json[step_str]["inputs"] = {}
                pipeline_json[step_str]["inputs"]["0"] = {}
                pipeline_json[step_str]["inputs"]["0"]["timestamp"] = 1
                pipeline_json[step_str]["inputs"]["0"]["sequence"] = seq
                pipeline_json[step_str]["inputs"]["0"]["labels"] = None
                pipeline_json[step_str]["inputs"]["0"]["space"] = {}
                pipeline_json[step_str]["inputs"]["0"]["space"]["timestamp"] = 1
                pipeline_json[step_str]["inputs"]["0"]["space"]["sequence"] = seq
                pipeline_json[step_str]["target"] = "Brain"
                pipeline_json[step_str]["model"] = "MRI_Brain"
                pipeline_json[step_str]["format"] = "thresholding"
                pipeline_json[step_str]["description"] = "Brain segmentation in {} (T1).".format(seq)

            step_index = step_index + 1
            step_str = str(step_index)
            pipeline_json[step_str] = {}
            pipeline_json[step_str]["task"] = "Registration"
            pipeline_json[step_str]["moving"] = {}
            pipeline_json[step_str]["moving"]["timestamp"] = 1
            pipeline_json[step_str]["moving"]["sequence"] = seq
            pipeline_json[step_str]["fixed"] = {}
            pipeline_json[step_str]["fixed"]["timestamp"] = 1
            pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
            pipeline_json[step_str]["description"] = "Registration from {} (T1) to T1-CE (T1)".format(seq)

            step_index = step_index + 1
            step_str = str(step_index)
            pipeline_json[step_str] = {}
            pipeline_json[step_str]["task"] = "Apply registration"
            pipeline_json[step_str]["moving"] = {}
            pipeline_json[step_str]["moving"]["timestamp"] = 1
            pipeline_json[step_str]["moving"]["sequence"] = seq
            pipeline_json[step_str]["fixed"] = {}
            pipeline_json[step_str]["fixed"]["timestamp"] = 1
            pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
            pipeline_json[step_str]["direction"] = "forward"
            pipeline_json[step_str]["description"] = "Apply registration from {} (T1) to T1-CE (T1)".format(seq)

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Segmentation"
        pipeline_json[step_str]["inputs"] = {}
        pipeline_json[step_str]["inputs"]["0"] = {}
        pipeline_json[step_str]["inputs"]["0"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["0"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["0"]["labels"] = None
        pipeline_json[step_str]["inputs"]["0"]["space"] = {}
        pipeline_json[step_str]["inputs"]["0"]["space"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["0"]["space"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["1"] = {}
        pipeline_json[step_str]["inputs"]["1"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["1"]["sequence"] = "T1-w"
        pipeline_json[step_str]["inputs"]["1"]["labels"] = None
        pipeline_json[step_str]["inputs"]["1"]["space"] = {}
        pipeline_json[step_str]["inputs"]["1"]["space"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["1"]["space"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["2"] = {}
        pipeline_json[step_str]["inputs"]["2"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["2"]["sequence"] = "FLAIR"
        pipeline_json[step_str]["inputs"]["2"]["labels"] = None
        pipeline_json[step_str]["inputs"]["2"]["space"] = {}
        pipeline_json[step_str]["inputs"]["2"]["space"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["2"]["space"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["3"] = {}
        pipeline_json[step_str]["inputs"]["3"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["3"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["3"]["labels"] = None
        pipeline_json[step_str]["inputs"]["3"]["space"] = {}
        pipeline_json[step_str]["inputs"]["3"]["space"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["3"]["space"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["4"] = {}
        pipeline_json[step_str]["inputs"]["4"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["4"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["inputs"]["4"]["labels"] = "Tumor"
        pipeline_json[step_str]["inputs"]["4"]["space"] = {}
        pipeline_json[step_str]["inputs"]["4"]["space"]["timestamp"] = 1
        pipeline_json[step_str]["inputs"]["4"]["space"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["target"] = "Tumor"
        pipeline_json[step_str]["model"] = "MRI_GBM_Postop_FV_5p"
        pipeline_json[step_str]["description"] = "Tumor segmentation in multiple sequences (T1)."

        with open(os.path.join(test_dir, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4)

        logging.info("Running segmentation pipeline unit test.\n")
        from raidionicsrads.compute import run_rads
        run_rads(rads_config_filename)

        logging.info("Collecting and comparing results.\n")
        automatic_seg_filename = os.path.join(output_dir, 'T1', 'input_t1gd_annotation-Tumor.nii.gz')
        if not os.path.exists(automatic_seg_filename):
            logging.error("Segmentation pipeline unit test failed, no segmentation was generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Segmentation pipeline unit test failed, no segmentation was generated.\n")
    except Exception as e:
        logging.error("Error during segmentation pipeline unit test with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during segmentation pipeline unit test with.\n")

    logging.info("Segmentation pipeline unit test succeeded.\n")
    shutil.rmtree(test_dir)


postoperative_segmentation_pipeline_test()
