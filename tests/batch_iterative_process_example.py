import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import argparse
import platform
from tqdm import tqdm


def batch_iterative_process_example():
    """
    Example code for running a pipeline iteratively over a set of patients. Custom pipelines can be created, the
    examples provided in the notebooks are a good place to start.

    All trained models, necessary for running the segmentation tasks selected in the pipeline, must be manually
    downloaded (https://github.com/dbouget/Raidionics-models/releases/tag/1.3.0-rc), extracted and placed within an overall
    models folder.

    For using Docker as processing backend, please first download the following image: dbouget/raidionics-rads:v1.3-py39-cpu

    The expected patient data structure is as follows:
    └── path/to/data/cohort/
        └── Pat001/
            ├── T0/
                ├── Pat001_t1c_pre.nii.gz
                ├── Pat001_t1c_pre_label_brain.nii.gz
                └── Pat001_t1w_pre.nii.gz
            └── T1/
                ├── Pat001_t1c_post.nii.gz
                ├── Pat001_t1c_post_label_brain.nii.gz
                ├── Pat001_t1w_post.nii.gz
                ├── Pat001_t2f_post.nii.gz
                └── Pat001_t2w_post.nii.gz
        [...]
        └── PatXXX/
            ├── T0/
                ├── PatXXX_t1c_pre.nii.gz
                └── PatXXX_t1c_pre_label_brain.nii.gz
            └── T1/
                ├── PatXXX_t1c_post.nii.gz
                └── PatXXX_t1c_post_label_brain.nii.gz
    Images inside each patient folder are expected to be in nifti format (nii.gz), and placed inside their respective
    timestamp subfolders: e.g., T0 folder for preoperative data and T1 folder for early postoperative data.
    For already existing segmentation/annotation files, the -label_target suffix must be provided (i.e., label_tumor),
    appended to the same name as the MRI image it corresponds to (e.g., input0.nii.gz and input0_label_tumor.nii.gz).
    The list of support label names can be viewed in raidionicsrads > Utils > AnnotationStructure.py (AnnotationClassType)

    Parameters
    ----------
    --input Folder path containing the patient cohort to process.
    --output Destination folder where the processed results will be dumped.
    --models Folder path where all trained models are located on disk.
    --backend Indication to either perform all processing directly (assuming located inside a proper venv) or inside
    a Docker container. To select from [local, docker].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='input', help='Path to the input patients folder')
    parser.add_argument('--output', metavar='output', help='Path to save the predictions')
    parser.add_argument('--models', metavar='models', help='Path to the trained models folder')
    parser.add_argument('--backend', metavar='backend', help='Favored processing approach, either venv or Docker',
                        choices=['local', 'docker'])
    parser.add_argument('--verbose', help="To specify the level of verbose, Default: warning", type=str,
                        choices=['debug', 'info', 'warning', 'error'], default='warning')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)
    cohort_folderpath = args.input
    dest_folderpath = args.output
    models_folderpath = args.models
    process_backend = args.backend

    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARNING)

    if args.verbose == 'debug':
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose == 'info':
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose == 'error':
        logging.getLogger().setLevel(logging.ERROR)

    patients = []
    for _, dirs, _ in os.walk(cohort_folderpath):
        for d in dirs:
            patients.append(d)
        break

    for pat in tqdm(patients):
        tmp_folder = ''
        try:
            # Setting up directories
            input_pat_folder = os.path.join(cohort_folderpath, pat)
            dest_pat_folder = os.path.join(dest_folderpath, pat)
            if os.path.exists(dest_pat_folder):
                print("Skipping inference for patient {}. Delete destination folder beforehand".format(pat))
                continue

            os.makedirs(dest_pat_folder)
            tmp_folder = os.path.join(dest_folderpath, 'tmp')
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            os.makedirs(tmp_folder)

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
            pipeline_json[step_str]["model"] = 'MRI_TumorCore'
            pipeline_json[step_str]["timestamp"] = 0
            pipeline_json[step_str]["format"] = "thresholding"
            pipeline_json[step_str][
                "description"] = "Identifying the best rest tumor core segmentation model for existing inputs"

            with open(os.path.join(dest_pat_folder, 'batch_iterative_pipeline.json'), 'w', newline='\n') as outfile:
                json.dump(pipeline_json, outfile, indent=4)

            # Setting up the configuration file
            rads_config = configparser.ConfigParser()
            rads_config.add_section('Default')
            rads_config.set('Default', 'task', 'neuro_diagnosis')
            rads_config.set('Default', 'caller', '')
            rads_config.add_section('System')
            rads_config.set('System', 'gpu_id', "-1")
            rads_config.set('System', 'input_folder', input_pat_folder)
            rads_config.set('System', 'output_folder', dest_pat_folder)
            rads_config.set('System', 'model_folder', models_folderpath)
            rads_config.set('System', 'pipeline_filename', os.path.join(dest_pat_folder, 'batch_iterative_pipeline.json'))
            rads_config.add_section('Runtime')
            rads_config.set('Runtime', 'reconstruction_method', 'probabilities')  # thresholding, probabilities
            rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
            rads_config.set('Runtime', 'use_preprocessed_data', 'False')

            rads_config_filename = os.path.join(dest_pat_folder, 'rads_config.ini')
            with open(rads_config_filename, 'w') as outfile:
                rads_config.write(outfile)

            # Running the process
            if process_backend == 'local':
                rads_config_filename = os.path.join(tmp_folder, 'rads_config.ini')
                with open(rads_config_filename, 'w') as outfile:
                    rads_config.write(outfile)
                if platform.system() == 'Windows':
                    subprocess.check_call(['raidionicsrads',
                                           '{config}'.format(config=rads_config_filename),
                                           '--verbose', args.verbose], shell=True)
                else:
                    subprocess.check_call(['raidionicsrads',
                                           '{config}'.format(config=rads_config_filename),
                                           '--verbose', args.verbose])
            elif process_backend == 'docker':
                # @OBS. For using Docker, the mounted folder should contain all necessary resources
                # (i.e., models folder in addition to input/output folders).
                # The following is an example where those folders are copied inside a temporary folder to be fed to
                # the Docker image (lots of copying overhead). The process can be greatly improved speed-wise!
                docker_folder = os.path.join(tmp_folder, 'docker')
                os.makedirs(docker_folder)

                shutil.copytree(src=models_folderpath, dst=os.path.join(docker_folder, 'models'))
                shutil.copytree(src=input_pat_folder, dst=os.path.join(docker_folder, 'inputs'))
                shutil.copytree(src=dest_pat_folder, dst=os.path.join(docker_folder, 'outputs'))

                rads_config.set('System', 'input_folder', '/workspace/resources/inputs')
                rads_config.set('System', 'output_folder', '/workspace/resources/outputs')
                rads_config.set('System', 'model_folder', '/workspace/resources/models')
                rads_config.set('System', 'pipeline_filename',
                                '/workspace/resources/outputs/batch_iterative_pipeline.json')
                rads_config_filename = os.path.join(docker_folder, 'outputs', 'rads_config.ini')
                with open(rads_config_filename, 'w') as outfile:
                    rads_config.write(outfile)
                cmd_docker = ['docker', 'run', '-v', '{}:/workspace/resources'.format(docker_folder),
                              '--network=host', '--ipc=host', '--user', str(os.geteuid()),
                              'dbouget/raidionics-rads:v1.3-py39-cpu',
                              '-c', f'/workspace/resources/outputs/rads_config.ini', '-v', args.verbose]
                if platform.system() == 'Windows':
                    subprocess.check_call(cmd_docker, shell=True)
                else:
                    subprocess.check_call(cmd_docker)
            else:
                logging.error("Backend option not supported, please select from [local, docker]")
                return

            shutil.copytree(src=os.path.join(docker_folder, 'outputs'), dst=dest_pat_folder, dirs_exist_ok=True)
            # Clean-up
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
        except Exception:
            print("Patient {} failed.".format(pat))
            print(traceback.format_exc())
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)

            continue


batch_iterative_process_example()
