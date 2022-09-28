import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback

try:
    import gdown
    if int(gdown.__version__.split('.')[0]) < 4 or int(gdown.__version__.split('.')[1]) < 4:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
    import gdown


def segmentation_pipeline_test():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running segmentation pipeline unit test.\n")
    logging.info("Downloading unit test resources.\n")
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    patient_dir = os.path.join(test_dir, 'patient')
    os.makedirs(patient_dir)
    output_dir = os.path.join(test_dir, 'results')
    os.makedirs(output_dir)
    models_dir = os.path.join(test_dir, 'models')
    os.makedirs(models_dir)

    try:
        test_image_url = 'https://drive.google.com/uc?id=1WWKheweJ8bbNCZbz7ZdnI5_P6xKZTkaL'  # Test patient
        seq_model_url = 'https://drive.google.com/uc?id=1DJc41omBVMM48HD4FKur8fVRFwvEa2t7'  # MRI sequence model
        brain_model_url = 'https://drive.google.com/uc?id=1FLsBz5_-w8yt6K-QmgXDMGD-v85Fl1QT'  # Brain model

        archive_dl_dest = os.path.join(test_dir, 'inference_patient.zip')
        gdown.cached_download(url=test_image_url, path=archive_dl_dest)
        gdown.extractall(path=archive_dl_dest, to=test_dir)

        archive_dl_dest = os.path.join(test_dir, 'seq-model.zip')
        gdown.cached_download(url=seq_model_url, path=archive_dl_dest)
        gdown.extractall(path=archive_dl_dest, to=models_dir)

        archive_dl_dest = os.path.join(test_dir, 'brain-model.zip')
        gdown.cached_download(url=brain_model_url, path=archive_dl_dest)
        gdown.extractall(path=archive_dl_dest, to=models_dir)
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
        rads_config.set('System', 'input_folder', patient_dir)
        rads_config.set('System', 'output_folder', output_dir)
        rads_config.set('System', 'model_folder', models_dir)
        rads_config.set('System', 'pipeline_filename', os.path.join(test_dir, 'test_pipeline.json'))
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
        pipeline_json[step_str]["description"] = "Brain segmentation in T1-CE (T0)."

        with open(os.path.join(test_dir, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running segmentation pipeline unit test.\n")
        from raidionicsrads.compute import run_rads
        run_rads(rads_config_filename)

        logging.info("Collecting and comparing results.\n")
        automatic_seg_filename = os.path.join(output_dir, 'T0', 'input1_annotation-Brain.nii.gz')
        if not os.path.exists(automatic_seg_filename):
            logging.error("Segmentation pipeline unit test failed, no segmentation was generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Segmentation pipeline unit test failed, no segmentation was generated.\n")

        logging.info("Standardized reporting CLI unit test started.\n")
        try:
            import platform
            if platform.system() == 'Windows':
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'debug'], shell=True)
            else:
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'debug'])
        except Exception as e:
            logging.error("Error during segmentation pipeline CLI unit test with: \n {}.\n".format(traceback.format_exc()))
            shutil.rmtree(test_dir)
            raise ValueError("Error during segmentation pipeline CLI unit test.\n")

        logging.info("Collecting and comparing results.\n")
        automatic_seg_filename = os.path.join(output_dir, 'T0', 'input1_annotation-Brain.nii.gz')
        if not os.path.exists(automatic_seg_filename):
            logging.error("Segmentation pipeline CLI unit test failed, no segmentation was generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Segmentation pipeline CLI unit test failed, no segmentation was generated.\n")
        logging.info("Standardized reporting CLI unit test succeeded.\n")
    except Exception as e:
        logging.error("Error during segmentation pipeline unit test with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during segmentation pipeline unit test with.\n")

    logging.info("Segmentation pipeline unit test succeeded.\n")
    shutil.rmtree(test_dir)


segmentation_pipeline_test()
