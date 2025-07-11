from .Utils.configuration_parser import ResourcesConfiguration
import time
import traceback
import logging
from .Utils.DataStructures.PatientStructure import PatientParameters
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
    try:
        patient_parameters = pip.execute(patient_parameters=patient_parameters)
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
