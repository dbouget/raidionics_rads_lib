import os
import yaml
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import argparse
import platform
from copy import deepcopy


def mlcube_brats():
    """
    Example code for running a pipeline iteratively over a set of patients. Custom pipelines can be performed by
    feeding a local json file to the following line below in the code:
    rads_config.set('System', 'pipeline_filename', os.path.join(models_folderpath, 'MRI_Tumor_Postop', 'pipeline.json'))

    All trained models, necessary for running the segmentation tasks selected in the pipeline, must be manually
    downloaded (https://github.com/dbouget/Raidionics-models/releases/tag/1.2.0), extracted and placed within an overall
    models folder.

    Parameters
    ----------
    --input Folder path containing sub-folders, one for each patient, to process. Images inside each patient folder are
    expected to be in nifti format (nii.gz) and their names must contain their MRI sequence type (i.e., t1gd for T1-CE,
    t1 for T1-w, flair for FLAIR) or the -label_target suffix for annotations (i.e., label_tumor) appended to the same
    name as the MRI image it corresponds to.
    --output Destination folder where the processed results will be dumped. Sub-folders, one for each patient, will be
    automatically generated.
    --models Folder path where all trained models are located on disk.
    --backend Indication to either perform all processing directly (assuming located inside a proper venv) or inside
    a Docker container. To chose from [local, docker].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='data_dir', help='Path to the input patients folder')
    parser.add_argument('--output_dir', metavar='output_dir', help='Path to save the predictions')
    parser.add_argument('--tumor_type', metavar='models', help='Tumor type to segment')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)
    input_folderpath = args.data_dir
    dest_folderpath = args.output_dir
    tumor_type = args.tumor_type

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    patients = []
    for _, dirs, _ in os.walk(input_folderpath):
        for d in dirs:
            patients.append(d)
        break

    if len(patients) == 0:
        # Not a cohort but a single patient
        patients.append(os.path.basename(input_folderpath))
        input_folderpath = os.path.dirname(input_folderpath)

    for pat in patients:
        tmp_folder = ''
        revamped_input_folder = ''
        try:
            # Setting up directories
            input_pat_folder = os.path.join(input_folderpath, pat)
            dest_pat_folder = os.path.join(dest_folderpath, pat)
            if os.path.exists(dest_pat_folder):
                print("Skipping inference for patient {}. Delete destination folder beforehand".format(pat))
                continue

            os.makedirs(dest_pat_folder)
            tmp_folder = os.path.join(dest_folderpath, 'tmp')
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            os.makedirs(tmp_folder)

            # Revamping the input folder for proper naming conventions
            revamped_input_folder = os.path.join(dest_folderpath, 'fix_input')
            logging.info("Running inference on {}".format(revamped_input_folder))
            logging.info("Inference results will be stored in {}".format(dest_pat_folder))

            if os.path.exists(revamped_input_folder):
                shutil.rmtree(revamped_input_folder)
            os.makedirs(revamped_input_folder)
            seq_files = []
            for _, _, files in os.walk(input_pat_folder):
                for f in files:
                    seq_files.append(f)
                break

            os.makedirs(os.path.join(revamped_input_folder, "T0"))
            for sf in seq_files:
                if "t1c" in sf:
                    new_name = deepcopy(sf)
                    new_name.replace("t1c", "t1_gd")
                    shutil.copyfile(src=os.path.join(input_pat_folder, sf),
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))
                if "t1w" in sf:
                    new_name = deepcopy(sf)
                    new_name.replace("t1w", "t1_woc")
                    shutil.copyfile(src=os.path.join(input_pat_folder, sf),
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))
                if "t2f" in sf:
                    new_name = deepcopy(sf)
                    new_name.replace("t2f", "flair")
                    shutil.copyfile(src=os.path.join(input_pat_folder, sf),
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))

                if "t2w" in sf:
                    new_name = deepcopy(sf)
                    new_name.replace("t2w", "t2")
                    shutil.copyfile(src=os.path.join(input_pat_folder, sf),
                                    dst=os.path.join(revamped_input_folder, "T0", new_name))

            # Setting up the configuration file
            rads_config = configparser.ConfigParser()
            rads_config.add_section('Default')
            rads_config.set('Default', 'task', 'neuro_diagnosis')
            rads_config.set('Default', 'caller', '')
            rads_config.add_section('System')
            rads_config.set('System', 'gpu_id', "-1")
            rads_config.set('System', 'input_folder', revamped_input_folder)
            rads_config.set('System', 'output_folder', dest_pat_folder)
            rads_config.set('System', 'model_folder', "/workspace/models/MRI_HGG_Brats")
            rads_config.set('System', 'pipeline_filename', '/workspace/models/MRI_HGG_Brats/pipeline.json')
            rads_config.add_section('Runtime')
            rads_config.set('Runtime', 'reconstruction_method', 'thresholding')  # thresholding, probabilities
            rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
            rads_config.set('Runtime', 'use_stripped_data', 'True')
            rads_config.set('Runtime', 'use_registered_data', 'True')

            # Running the process
            rads_config_filename = os.path.join(tmp_folder, 'rads_config.ini')
            with open(rads_config_filename, 'w') as outfile:
                rads_config.write(outfile)
            if platform.system() == 'Windows':
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'info'], shell=True)
            else:
                subprocess.check_call(['raidionicsrads',
                                       '{config}'.format(config=rads_config_filename),
                                       '--verbose', 'info'])
            # Clean-up
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            if os.path.exists(revamped_input_folder):
                shutil.rmtree(revamped_input_folder)
        except Exception:
            print("Patient {} failed.".format(pat))
            print(traceback.format_exc())
            if os.path.exists(tmp_folder):
                shutil.rmtree(tmp_folder)
            if os.path.exists(revamped_input_folder):
                shutil.rmtree(revamped_input_folder)
            continue


mlcube_brats()
