import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile


def test_registration_pipeline_package(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running registration pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_reg_package")
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

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Segmentation"
        pipeline_json[step_str]["inputs"] = {}
        pipeline_json[step_str]["inputs"]["0"] = {}
        pipeline_json[step_str]["inputs"]["0"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["0"]["sequence"] = "FLAIR"
        pipeline_json[step_str]["inputs"]["0"]["labels"] = None
        pipeline_json[step_str]["inputs"]["0"]["space"] = {}
        pipeline_json[step_str]["inputs"]["0"]["space"]["timestamp"] = 0
        pipeline_json[step_str]["inputs"]["0"]["space"]["sequence"] = "FLAIR"
        pipeline_json[step_str]["target"] = "Brain"
        pipeline_json[step_str]["model"] = "MRI_Brain"
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Brain segmentation in FLAIR (T0)."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "FLAIR"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = 0
        pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["description"] = "Registration from FLAIR (T0) to T1CE (T0)."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Apply registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "FLAIR"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = 0
        pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["direction"] = "forward"
        pipeline_json[step_str]["description"] = "Apply registration from FLAIR (T0) to T1CE (T0)."

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running registration pipeline unit test.\n")
        from raidionicsrads.compute import run_rads
        run_rads(rads_config_filename)

        logging.info("Collecting and comparing results.\n")
        transform_dir = os.listdir(os.path.join(output_folder, "Transforms"))
        assert len(transform_dir) > 0, "No transform folder was generated"
        registered_inputs = os.listdir(os.path.join(output_folder, "T0", "T0_T1c_space"))
        assert len(registered_inputs) > 0, "No registered files were generated"

        logging.info("Registration CLI unit test succeeded.\n")
    except Exception as e:
        logging.error(f"Error during registration pipeline unit test with: {e}\n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError("Error during registration pipeline unit test with.\n")

    logging.info("Registration pipeline unit test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def registration_pipeline_test_cli(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running registration pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_reg_cli")
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
        rads_config.set('System', 'pipeline_filename', os.path.join(output_folder, 'test_pipeline.json'))
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

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = 0
        pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["description"] = "Registration from T1CE (T0) to T1CE (T0)."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Apply registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = 0
        pipeline_json[step_str]["fixed"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["direction"] = "forward"
        pipeline_json[step_str]["description"] = "Apply registration from T1CE (T0) to T1CE (T0)."

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Registration CLI unit test started.\n")
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
            logging.error(f"Error during registration pipeline CLI unit test with: {e}\n {traceback.format_exc()}.\n")
            raise ValueError("Error during registration pipeline CLI unit test.\n")

        logging.info("Collecting and comparing results.\n")
        transform_dir = os.listdir(os.path.join(output_folder, "Transforms"))
        assert len(transform_dir) > 0, "No transform folder was generated"
        registered_inputs = os.listdir(os.path.join(output_folder, "T0", "T0_T1c_space"))
        assert len(registered_inputs) > 0, "No registered files were generated"

        logging.info("Registration CLI unit test succeeded.\n")
    except Exception as e:
        logging.error(f"Error during registration pipeline unit test with: {e}\n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError("Error during registration pipeline unit test with.\n")

    logging.info("Registration pipeline unit test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)