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
    import gdown
    if int(gdown.__version__.split('.')[0]) < 4 or int(gdown.__version__.split('.')[1]) < 4:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'requests==2.28.2'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
    import requests
    import gdown


def registration_pipeline_docker_test():
    """
    Testing the CLI within a Docker container for the registration pipeline unit test, running on CPU.
    The latest Docker image is being hosted at: dbouget/raidionics-rads:v1.1-py38-cpu

    Returns
    -------

    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running registration pipeline unit test in Docker container.\n")
    logging.info("Downloading unit test resources.\n")
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    patient_dir = os.path.join(test_dir, 'patient')
    output_dir = os.path.join(test_dir, 'results')
    os.makedirs(output_dir)
    models_dir = os.path.join(test_dir, 'models')
    os.makedirs(models_dir)

    try:
        test_image_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Samples-RaidionicsRADSLib-UnitTest1.zip'  # Test patient
        seq_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Raidionics-MRI_Sequence_Classifier-ONNX-v12.zip'
        brain_model_url = 'https://github.com/raidionics/Raidionics-models/releases/download/1.2.0/Raidionics-MRI_Brain-ONNX-v12.zip'

        archive_dl_dest = os.path.join(test_dir, 'inference_patient.zip')
        headers = {}
        response = requests.get(test_image_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(test_dir)

        archive_dl_dest = os.path.join(test_dir, 'seq-model.zip')
        headers = {}
        response = requests.get(seq_model_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(models_dir)

        archive_dl_dest = os.path.join(test_dir, 'brain-model.zip')
        headers = {}
        response = requests.get(brain_model_url, headers=headers, stream=True)
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
        rads_config = configparser.ConfigParser()
        rads_config.add_section('Default')
        rads_config.set('Default', 'task', 'neuro_diagnosis')
        rads_config.set('Default', 'caller', '')
        rads_config.add_section('System')
        rads_config.set('System', 'gpu_id', "-1")
        rads_config.set('System', 'input_folder', patient_dir.replace(test_dir, '/workspace/resources'))
        rads_config.set('System', 'output_folder', output_dir.replace(test_dir, '/workspace/resources'))
        rads_config.set('System', 'model_folder', models_dir.replace(test_dir, '/workspace/resources'))
        rads_config.set('System', 'pipeline_filename', os.path.join('/workspace/resources', 'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'thresholding')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config_filename = os.path.join(output_dir, 'rads_config.ini')
        with open(rads_config_filename, 'w') as outfile:
            rads_config.write(outfile)

        pipeline_json = {}
        step_index = 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Classification"
        pipeline_json[step_str]["inputs"] = {}  # Empty input means running it on all existing data for the patient
        pipeline_json[step_str]["model"] = "MRI_Sequence_Classifier"
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

        with open(os.path.join(test_dir, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running registration pipeline unit test in Docker container.\n")
        try:
            import platform
            cmd_docker = ['docker', 'run', '-v', '{}:/workspace/resources'.format(test_dir),
                          '--network=host', '--ipc=host', '--user', str(os.geteuid()), 'dbouget/raidionics-rads:v1.1-py38-cpu',
                          '-c', '/workspace/resources/results/rads_config.ini', '-v', 'debug']
            logging.info("Executing the following Docker call: {}".format(cmd_docker))
            if platform.system() == 'Windows':
                subprocess.check_call(cmd_docker, shell=True)
            else:
                subprocess.check_call(cmd_docker)
        except Exception as e:
            logging.error("Error during registration pipeline unit test in Docker container with: \n {}.\n".format(traceback.format_exc()))
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            raise ValueError("Error during registration pipeline unit test in Docker container.\n")

        logging.info("Collecting and comparing results.\n")
        # @TODO. How to check/compare results?

        logging.info("Registration unit test in Docker container succeeded.\n")
    except Exception as e:
        logging.error("Error during registration pipeline unit test in Docker container with: \n {}.\n".format(traceback.format_exc()))
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        raise ValueError("Error during registration pipeline unit test in Docker container with.\n")

    logging.info("Registration pipeline unit test in Docker container succeeded.\n")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


registration_pipeline_docker_test()
