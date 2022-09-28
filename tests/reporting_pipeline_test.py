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


def reporting_pipeline_test():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running standard reporting unit test.\n")
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
        test_image_url = 'https://drive.google.com/uc?id=1WWKheweJ8bbNCZbz7ZdnI5_P6xKZTkaL'  # Test patient
        seq_model_url = 'https://drive.google.com/uc?id=1DJc41omBVMM48HD4FKur8fVRFwvEa2t7'  # MRI sequence model
        brain_model_url = 'https://drive.google.com/uc?id=1FLsBz5_-w8yt6K-QmgXDMGD-v85Fl1QT'  # Brain model
        test_model_url = 'https://drive.google.com/uc?id=1-uUgFQDQxDDrSrkNljrw2hD4DO6q7p-b'  # HGGlioma model

        archive_dl_dest = os.path.join(test_dir, 'inference_patient.zip')
        gdown.cached_download(url=test_image_url, path=archive_dl_dest)
        gdown.extractall(path=archive_dl_dest, to=test_dir)

        archive_dl_dest = os.path.join(test_dir, 'seq-model.zip')
        gdown.cached_download(url=seq_model_url, path=archive_dl_dest)
        gdown.extractall(path=archive_dl_dest, to=models_dir)

        archive_dl_dest = os.path.join(test_dir, 'brain-model.zip')
        gdown.cached_download(url=brain_model_url, path=archive_dl_dest)
        gdown.extractall(path=archive_dl_dest, to=models_dir)

        archive_dl_dest = os.path.join(test_dir, 'model.zip')
        gdown.cached_download(url=test_model_url, path=archive_dl_dest)
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
        rads_config.add_section('Neuro')
        rads_config.set('Neuro', 'cortical_features', 'MNI')
        rads_config.set('Neuro', 'subcortical_features', '')
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
        pipeline_json[step_str]["model"] = "MRI_HGGlioma_P2"
        pipeline_json[step_str]["description"] = "Tumor segmentation in T1-CE (T0)."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = -1
        pipeline_json[step_str]["fixed"]["sequence"] = "MNI"
        pipeline_json[step_str]["description"] = "Registration T1-CE (T0) to MNI."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Apply registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = -1
        pipeline_json[step_str]["fixed"]["sequence"] = "MNI"
        pipeline_json[step_str]["direction"] = "forward"
        pipeline_json[step_str]["description"] = "Apply registration from T1-CE (T0) to MNI."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Apply registration"
        pipeline_json[step_str]["moving"] = {}
        pipeline_json[step_str]["moving"]["timestamp"] = 0
        pipeline_json[step_str]["moving"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["fixed"] = {}
        pipeline_json[step_str]["fixed"]["timestamp"] = -1
        pipeline_json[step_str]["fixed"]["sequence"] = "MNI"
        pipeline_json[step_str]["direction"] = "inverse"
        pipeline_json[step_str]["description"] = "Apply inverse registration from MNI to T1-CE (T0)."

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = "Features computation"
        pipeline_json[step_str]["input"] = {}
        pipeline_json[step_str]["input"]["timestamp"] = 0
        pipeline_json[step_str]["input"]["sequence"] = "T1-CE"
        pipeline_json[step_str]["target"] = "Tumor"
        pipeline_json[step_str]["space"] = "MNI"
        pipeline_json[step_str]["description"] = "Tumor features computation from T1-CE (T0) in MNI space."

        with open(os.path.join(test_dir, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running standardized reporting unit test.\n")
        from raidionicsrads.compute import run_rads
        run_rads(rads_config_filename)

        logging.info("Collecting and comparing results.\n")
        report_filename = os.path.join(output_dir, 'neuro_clinical_report.json')
        if not os.path.exists(report_filename):
            logging.error("Reporting pipeline unit test failed, no report was generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Reporting pipeline unit test failed, no report was generated.\n")

        logging.info("Reporting pipeline CLI unit test started.\n")
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
            logging.error("Error during reporting pipeline CLI unit test with: \n {}.\n".format(traceback.format_exc()))
            shutil.rmtree(test_dir)
            raise ValueError("Error during reporting pipeline CLI unit test.\n")

        logging.info("Collecting and comparing results.\n")
        report_filename = os.path.join(output_dir, 'neuro_clinical_report.json')
        if not os.path.exists(report_filename):
            logging.error("Reporting pipeline CLI unit test failed, no report was generated.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Reporting pipeline CLI unit test failed, no report was generated.\n")
        logging.info("Reporting pipeline CLI unit test succeeded.\n")
    except Exception as e:
        logging.error("Error during reporting pipeline unit test with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during reporting pipeline unit test with.\n")

    logging.info("Reporting pipeline unit test succeeded.\n")
    shutil.rmtree(test_dir)


reporting_pipeline_test()
