from .Utils.configuration_parser import ResourcesConfiguration
import time
import traceback
import logging
import os
import json
import numpy as np
from .Utils.DataStructures.PatientStructure import PatientParameters
from .Utils.DataStructures.AnnotationStructure import Annotation, AnnotationClassType
from .Utils.utilities import get_type_from_enum_name
from .Pipelines.PipelineStructure import Pipeline
from .Pipelines.ClassificationStep import ClassificationStep


def run_rads(config_filename: str, logging_filename: str = None) -> None:
    """

    """
    ResourcesConfiguration.getInstance().set_environment(config_path=config_filename)
    if logging_filename:
        # logging.basicConfig(filename=logging_filename, filemode='a',
        #                     format="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s", datefmt='%d/%m/%Y %H.%M')
        # logging.getLogger().setLevel(logging.DEBUG)
        logger = logging.getLogger()
        handler = logging.FileHandler(filename=logging_filename, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter(fmt="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s",
                                               datefmt='%d/%m/%Y %H.%M'))
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    executed_pipeline_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "executed_pipeline.json")
    completed_pipeline_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "executed_pipeline_completed.json")
    existing_pipeline = None
    if os.path.exists(completed_pipeline_fn):
        with open(completed_pipeline_fn, 'r') as infile:
            existing_pipeline = json.load(infile)

    logging.info("Starting pipeline for file: {}.".format(ResourcesConfiguration.getInstance().pipeline_filename))
    start = time.time()
    pip = Pipeline(ResourcesConfiguration.getInstance().pipeline_filename)
    try:
        patient_parameters = PatientParameters(id="Patient",
                                               patient_filepath=ResourcesConfiguration.getInstance().input_folder)
    except Exception as e:
        logging.error("""[Backend error] Patient data setup phase of failed with:\n{}""".format(e))
        logging.debug("Traceback: {}.".format(traceback.format_exc()))
        return
    try:
        patient_parameters = pip.setup(patient_parameters=patient_parameters)
    except Exception as e:
        logging.error("""[Backend error] Patient data setup phase for models in automatic selection failed with:\n{}""".format(e))
        logging.debug("Traceback: {}.".format(traceback.format_exc()))
        return

    with open(executed_pipeline_fn, 'r') as infile:
        new_pipeline = json.load(infile)
    steps_computed = _steps_already_computed(new_pipeline, existing_pipeline)
    _apply_step_skip_decisions(pip, new_pipeline, steps_computed, patient_parameters)
    logging.info("Steps already computed: {}/{}.".format(sum(steps_computed.values()), len(steps_computed)))
    
    try:
        patient_parameters = pip.execute(patient_parameters=patient_parameters)
        with open(completed_pipeline_fn, 'w', newline='\n') as outfile:
            json.dump(new_pipeline, outfile, indent=4)
        pip.cleanup()
    except Exception as e:
        logging.error("""[Backend error] Patient data execution phase of failed with:\n{}""".format(e))
        logging.debug("Traceback: {}.".format(traceback.format_exc()))
        pip.cleanup()
        return
    logging.info('Total elapsed time for executing the pipeline: {} seconds.'.format(time.time() - start))


def run_folder_inspection(config_filename: str, logging_filename: str = None) -> None:
    # The user could go and manually check if stuff is correct before running the actual pipeline
    # Only if direct use, stuff will be assumed correct if coming from Raidionics,
    # or can be called from there and inspect in the GUI?
    # @TODO. I think it should not be a stand-alone method, rather a stand-alone pipeline.json or a step inside another
    # But there's need for a way to dump/communicate the info to Raidionics.
    ResourcesConfiguration.getInstance().set_environment(config_path=config_filename)
    if logging_filename:
        logging.basicConfig(filename=logging_filename, filemode='a',
                            format="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s", datefmt='%d/%m/%Y %H.%M')
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        patient_parameters = PatientParameters(id="Patient",
                                               patient_filepath=ResourcesConfiguration.getInstance().input_folder)
    except Exception as e:
        logging.error("""[Backend error] Patient data setup phase of failed with:\n{}""".format(e))
        logging.debug("Traceback: {}.".format(traceback.format_exc()))
        return

    class_json = {}
    class_json["task"] = "classification"
    class_json["inputs"] = {}  # Empty input means running it on all existing data for the patient
    class_json["model"] = "MRI_Sequence_Classifier"
    class_json["description"] = "Classification of the MRI sequence type for all input scans."

    logging.info("Starting sequence classification pipeline.")
    start = time.time()

    try:
        classification = ClassificationStep(class_json)
        classification.setup(patient_parameters)
        patient_parameters = classification.execute()
    except Exception as e:
        logging.error("""[Backend error] Classification step setup or execution phase failed with: {}""".format(e))
        logging.debug("Traceback: {}.".format(traceback.format_exc()))
        return
    # @TODO. Should dump it differently, or arrange filenames for re-use in Raidionics, or return the updated
    # patient_parameters if running another real pipeline straight after.
    logging.info('Total elapsed time for executing the pipeline: {} seconds.'.format(time.time() - start))

def preview_pipeline(config_filename: str, sequences_declaration: dict, logging_filename: str = None) -> bool:
    """
    Builds executed_pipeline.json for a given pipeline.json and a declared set of MR
    sequences, without running classification or any actual inference.
    """
    ResourcesConfiguration.getInstance().set_environment(config_path=config_filename)
    if logging_filename:
        logging.basicConfig(filename=logging_filename, filemode='a', format="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s", datefmt='%d/%m/%Y %H.%M')
        logging.getLogger().setLevel(logging.DEBUG)

    completed_pipeline_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "executed_pipeline_completed.json")
    existing_pipeline = None
    if os.path.exists(completed_pipeline_fn):
        with open(completed_pipeline_fn, 'r') as infile:
            existing_pipeline = json.load(infile)

    try:
        patient_parameters = PatientParameters(id="Patient", declared_sequences=sequences_declaration)
    except Exception as e:
        logging.error("""[Backend error] Patient data setup phase of failed with:\n{}""".format(e))
        logging.debug("Traceback: {}.".format(traceback.format_exc()))
        return False

    logging.info("Starting pipeline preview for file: {}.".format(ResourcesConfiguration.getInstance().pipeline_filename))
    start = time.time()
    try:
        pip = Pipeline(ResourcesConfiguration.getInstance().pipeline_filename)
        pip.mark_sequence_as_known()
        pip.setup(patient_parameters=patient_parameters)
    except Exception as e:
        logging.error("""[Backend error] Pipeline preview setup phase failed with:\n{}""".format(e))
        logging.debug("Traceback: {}.".format(traceback.format_exc()))
        return False
    logging.info('Total elapsed time for building the pipeline preview: {} seconds.'.format(time.time() - start))

    executed_pipeline_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "executed_pipeline.json")
    with open(executed_pipeline_fn, 'r') as infile:
        new_pipeline = json.load(infile)
    steps_computed = _steps_already_computed(new_pipeline, existing_pipeline)
    already_computed = all(steps_computed.values())
    logging.info("Steps already computed: {}/{}.".format(sum(steps_computed.values()), len(steps_computed)))
    logging.info("Pipeline already fully computed: {}.".format(already_computed))
    return already_computed

def _step_signature(step: dict) -> str:
    return json.dumps(step, sort_keys=True)

def _steps_already_computed(new_pipeline: dict, existing_pipeline: dict) -> dict:
    """
    For each step in the new plan, checks whether an identical step (matched by full
    content, not by key position) already exists somewhere in a previous run's plan.
    """
    if existing_pipeline is None:
        return {k: False for k in new_pipeline}
    existing_signatures = {_step_signature(v) for v in existing_pipeline.values()}
    return {k: _step_signature(v) in existing_signatures for k, v in new_pipeline.items()}

def _reload_existing_segmentation(step: dict, patient_parameters) -> bool:
    """
    For a Segmentation step already covered by a previous run, reconstructs the annotation
    filename it would have written (same convention as SegmentationStep) and, if found on
    disk, registers it into patient_parameters so downstream steps (e.g. Segmentation
    refinement) can find it without recomputing.
    """
    input_json = step["inputs"]["0"]
    volume_uid = patient_parameters.get_radiological_volume_uid(timestamp=input_json["timestamp"], sequence=input_json["sequence"])

    if volume_uid == "-1":
        return False
    volume  = patient_parameters.get_radiological_volume(volume_uid)

    for target in step["target"]:
        annotation_class = get_type_from_enum_name(AnnotationClassType, target)
        if annotation_class == -1:
            return False
        if len(patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid, annotation_class=annotation_class)) > 0:
            continue # already registered, e.g. by an earlier step in this same run

        anno_fn = os.path.join(volume.output_folder, os.path.basename(volume.raw_input_filepath).split('.')[0] + '_annotation-' + target + '_' + step["model"].split('/')[0] + '.nii.gz')

        if not os.path.exists(anno_fn):
            return False

        non_available_uid = True
        anno_uid = None
        while non_available_uid:
            anno_uid = 'A' + str(np.random.randint(0, 10000))
            if anno_uid not in patient_parameters.get_all_annotations_uids():
                non_available_uid = False

        annotation = Annotation(uid=anno_uid, input_filename=anno_fn, output_folder=volume.output_folder, radiological_volume_uid=volume_uid, annotation_class=target)
        patient_parameters.include_annotation(anno_uid, annotation)
    return True

def _apply_step_skip_decisions(pip, new_pipeline: dict, steps_computed: dict, patient_parameters) -> None:
    """
    For each step already covered by a previous run, attempts to reload its expected existing
    output into patient_parameters so downstream steps still find their input, and only then
    marks it skippable. Steps whose output can't be reloaded are left to run for real, which
    is always safe.
    """
    for step_key in sorted(new_pipeline.keys(), key=int):
        if not steps_computed.get(step_key) or step_key not in pip._steps:
            continue
        step_json = new_pipeline[step_key]
        task = step_json.get("task")
        step_obj = pip._steps[step_key]

        if task == "Classification":
            continue  # always re-run for real; required to assign sequence types

        elif task == "Segmentation":
            if _reload_existing_segmentation(step_json, patient_parameters):
                step_obj.skip = True

        elif task == "Segmentation refinement":
            input_json = step_json["inputs"]["0"]
            volume_uid = patient_parameters.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                                         sequence=input_json["sequence"])
            annotation_class = get_type_from_enum_name(AnnotationClassType, input_json["labels"]) if input_json.get("labels") else -1
            if volume_uid != "-1" and annotation_class != -1 and len(
                    patient_parameters.get_all_annotations_uids_class_radiological_volume(
                        volume_uid=volume_uid, annotation_class=annotation_class)) > 0:
                step_obj.skip = True

        elif task == "Features computation":
            report_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "reporting", "T" + str(step_json["timestamp"]), "neuro_clinical_report.json")
            if os.path.exists(report_fn):
                step_obj.skip = True

        elif task == "Surgical reporting":
            report_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "reporting", "neuro_surgical_report.json")
            if os.path.exists(report_fn):
                step_obj.skip = True
                