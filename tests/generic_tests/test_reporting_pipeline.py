import os
import json
import shutil
import configparser
import logging
import sys
import subprocess
import traceback


def test_reporting_pipeline_package(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running segmentation pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_reporting_package")
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
                                                               "patient-UnitTest2", "inputs"))
        rads_config.set('System', 'output_folder', output_folder)
        rads_config.set('System', 'model_folder', os.path.join(test_dir, "models"))
        rads_config.set('System', 'pipeline_filename', os.path.join(output_folder,
                                                                    'test_pipeline.json'))
        rads_config.add_section('Runtime')
        rads_config.set('Runtime', 'reconstruction_method', 'thresholding')
        rads_config.set('Runtime', 'reconstruction_order', 'resample_first')
        rads_config.set('Runtime', 'use_stripped_data', 'True' if using_skull_stripped_inputs else 'False')
        rads_config.add_section('Neuro')
        rads_config.set('Neuro', 'cortical_features', 'MNI')
        ## Might not want to run it while debugging the CIs as it is quite time-consuming
        # rads_config.set('Neuro', 'subcortical_features', 'BCB')
        rads_config.set('Neuro', 'subcortical_features', 'BrainGrid')
        rads_config.set('Neuro', 'braingrid_features', 'Voxels')
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
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'MRI_TumorCE_Postop'
        pipeline_json[step_str]["timestamp"] = 1
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best rest enhancing tumor segmentation model for existing inputs"

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = 'Reporting selection'
        pipeline_json[step_str]["scope"] = 'standalone'
        pipeline_json[step_str]["timestamps"] = [1]
        pipeline_json[step_str]["tumor_type"] = "contrast-enhancing"
        pipeline_json[step_str]["description"] = "Identifying the reporting method for existing inputs"

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running standardized reporting unit test.\n")
        from raidionicsrads.compute import run_rads
        run_rads(rads_config_filename)

        logging.info("Collecting and comparing results.\n")
        report_filename_json = os.path.join(output_folder, "reporting", "T1", 'neuro_clinical_report.json')
        assert os.path.exists(report_filename_json), "No report clinical generated in json format."
        report_filename_csv = os.path.join(output_folder, "reporting", "T1", 'neuro_clinical_report.csv')
        assert os.path.exists(report_filename_csv), "No report clinical generated in csv format."
        report_filename_txt = os.path.join(output_folder, "reporting", "T1", 'neuro_clinical_report.txt')
        assert os.path.exists(report_filename_txt), "No report clinical generated in text format."
        tumorce_anno_filename = os.path.join(output_folder, "T1", 'postop_t1gd_annotation-TumorCE_MRI_TumorCE_Postop.nii.gz')
        assert os.path.exists(tumorce_anno_filename), "No TumorCE segmentation was generated over the postop image"
        mni_structures_filename = os.path.join(output_folder, "T1", "Cortical-structures", "MNI_MNI_atlas.nii.gz")
        assert os.path.exists(mni_structures_filename), "No MNI cortical structures atlas was generated"

    except Exception as e:
        logging.error(f"Error during reporting pipeline unit test with: {e}\n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError("Error during reporting pipeline unit test with.\n")

    logging.info("Reporting pipeline unit test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


def test_reporting_pipeline_surgical(test_dir):
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running segmentation pipeline unit test.\n")

    logging.info("Preparing configuration file.\n")
    try:
        output_folder = os.path.join(test_dir, "output_reporting_surgical")
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
                                                               "patient-UnitTest2", "inputs"))
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
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'MRI_TumorCore'
        pipeline_json[step_str]["timestamp"] = 0
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best tumor core segmentation model for existing inputs"

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = 'Reporting selection'
        pipeline_json[step_str]["scope"] = 'standalone'
        pipeline_json[step_str]["timestamps"] = [0]
        pipeline_json[step_str]["tumor_type"] = "contrast-enhancing"
        pipeline_json[step_str]["description"] = "Identifying the reporting method for existing inputs"

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'MRI_TumorCE_Postop'
        pipeline_json[step_str]["timestamp"] = 1
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best rest enhancing tumor segmentation model for existing inputs"

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = 'Model selection'
        pipeline_json[step_str]["model"] = 'MRI_FLAIRChanges'
        pipeline_json[step_str]["timestamp"] = 1
        pipeline_json[step_str]["format"] = "thresholding"
        pipeline_json[step_str]["description"] = "Identifying the best FLAIR changes segmentation model for existing inputs"


        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = 'Reporting selection'
        pipeline_json[step_str]["scope"] = 'standalone'
        pipeline_json[step_str]["timestamps"] = [1]
        pipeline_json[step_str]["tumor_type"] = "contrast-enhancing"
        pipeline_json[step_str]["description"] = "Identifying the reporting method for existing inputs"

        step_index = step_index + 1
        step_str = str(step_index)
        pipeline_json[step_str] = {}
        pipeline_json[step_str]["task"] = 'Reporting selection'
        pipeline_json[step_str]["scope"] = 'surgical'
        pipeline_json[step_str]["timestamps"] = [0, 1]
        pipeline_json[step_str]["tumor_type"] = "contrast-enhancing"
        pipeline_json[step_str]["description"] = "Identifying the reporting method for existing inputs"

        with open(os.path.join(output_folder, 'test_pipeline.json'), 'w', newline='\n') as outfile:
            json.dump(pipeline_json, outfile, indent=4, sort_keys=True)

        logging.info("Running standardized reporting unit test.\n")
        from raidionicsrads.compute import run_rads
        run_rads(rads_config_filename)

        logging.info("Collecting and comparing results.\n")
        report_filename_preop_json = os.path.join(output_folder, "reporting", "T0", 'neuro_clinical_report.json')
        assert os.path.exists(report_filename_preop_json), "No preop report clinical generated in json format."
        report_filename_postop_json = os.path.join(output_folder, "reporting", "T1", 'neuro_clinical_report.json')
        assert os.path.exists(report_filename_postop_json), "No postop report clinical generated in json format."
        report_filename_surgical_json = os.path.join(output_folder, "reporting", 'neuro_surgical_report.json')
        assert os.path.exists(report_filename_surgical_json), "No surgical report generated in json format."

    except Exception as e:
        logging.error(f"Error during reporting pipeline unit test with: {e}\n {traceback.format_exc()}.\n")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        raise ValueError("Error during reporting pipeline unit test with.\n")

    logging.info("Reporting pipeline unit test succeeded.\n")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
