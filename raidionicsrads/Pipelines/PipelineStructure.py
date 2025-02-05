import json
import logging
import os.path
import time
import traceback

from aenum import Enum, unique
from raidionicsseg.Utils.configuration_parser import ConfigResources

from ..Utils.utilities import get_type_from_string
from ..Utils.configuration_parser import ResourcesConfiguration
from .ClassificationStep import ClassificationStep
from .SegmentationStep import SegmentationStep
from .SegmentationRefinementStep import SegmentationRefinementStep
from .RegistrationStep import RegistrationStep
from .RegistrationDeployerStep import RegistrationDeployerStep
from .FeaturesComputationStep import FeaturesComputationStep
from .SurgicalReportingStep import SurgicalReportingStep
from .ModelSelectionStep import ModelSelectionStep


@unique
class TaskType(Enum):
    """

    """
    _init_ = 'value string'

    Class = 0, 'Classification'
    Seg = 1, 'Segmentation'
    Reg = 2, 'Registration'
    AReg = 3, "Apply registration"
    FComp = 4, "Features computation"
    SRep = 5, "Surgical reporting"
    SegRef = 6, "Segmentation refinement"
    ModSelec = 7, "Model selection"

    def __str__(self):
        return self.string


class Pipeline:
    """
    Class defining how an MRI volume should be handled.
    """
    _input_filepath = ""  # Full filepath to the current pipeline, stored in a json file
    _pipeline_json = {}  # Loaded pipeline from the aforementioned json file, stored as a dictionary
    _steps = {}  # Internal pipeline steps, inherited from AbstractPipelineStep, matching the steps inside the json dict.

    def __init__(self, input_filename: str) -> None:
        self.__reset()
        self._input_filepath = input_filename
        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._input_filepath = ""
        self._pipeline_json = {}
        self._steps = {}

    def __init_from_scratch(self):
        """

        Returns
        -------

        """
        with open(self._input_filepath, 'r') as infile:
            self._pipeline_json = json.load(infile)

        self.__parse_pipeline_steps(pipeline=self._pipeline_json, initial=True)

    def __parse_pipeline_steps(self, pipeline: {}, initial: bool = True) -> None:
        self._steps = {}
        for i, s in enumerate(list(pipeline.keys())):
            task = get_type_from_string(TaskType, pipeline[s]["task"])
            step = None
            if task == TaskType.Class:
                step = ClassificationStep(pipeline[s])
                if pipeline[s]["target"][0] == "MRSequence" and not initial:
                    step.skip = True
            elif task == TaskType.Seg:
                step = SegmentationStep(pipeline[s])
            elif task == TaskType.SegRef:
                step = SegmentationRefinementStep(pipeline[s])
            elif task == TaskType.Reg:
                step = RegistrationStep(pipeline[s])
            elif task == TaskType.AReg:
                step = RegistrationDeployerStep(pipeline[s])
            elif task == TaskType.FComp:
                step = FeaturesComputationStep(pipeline[s])
            elif task == TaskType.SRep:
                step = SurgicalReportingStep(pipeline[s])
            elif task == TaskType.ModSelec:
                step = ModelSelectionStep(pipeline[s])
            if step:
                self._steps[str(i)] = step
            else:
                logging.warning(f"Step dismissed because task could not be matched.")

    def setup(self, patient_parameters) -> None:
        """
        @TODO. Should not consider all classification tasks the same, the initial exception is only for the sequence
        classification which is mandatory for further disambiguation in model selection...
        @TODO. How to propagate down the probabilities/thresholding decision for the segmentation models (is it
        enough with the main_config.ini parameter?

        Parameters
        ----------
        patient_parameters

        Returns
        -------

        """
        logging.info('LOG: Pipeline setup - {} steps.'.format(len(self._steps)))
        final_pipeline = {}
        final_count = 0
        for s in list(self._steps.keys()):
            try:
                if self._steps[s].get_task() in [str(TaskType.Class), str(TaskType.ModSelec)]:
                    start = time.time()
                    logging.info("LOG: Pipeline - {desc} - Begin ({curr}/{tot})".format(
                        desc=self._steps[s].step_description,
                        curr=str(int(s) + 1),
                        tot=len(self._steps)))
                    try:
                        self._steps[s].setup(patient_parameters)
                    except Exception as e:
                        logging.error("""[Backend error] Setup phase of {} failed with:\n{}""".format(
                            self._steps[s].step_json, e))
                        logging.debug("Traceback: {}.".format(traceback.format_exc()))
                        break
                    try:
                        if self._steps[s].get_task() == str(TaskType.Class):
                            patient_parameters = self._steps[s].execute()
                            final_count = final_count + 1
                            final_count_str = str(final_count)
                            final_pipeline[final_count_str] = {}
                            final_pipeline[final_count_str] = self._steps[s].step_json
                        else:
                            task_optimal_pipeline = self._steps[s].execute()
                            for top in task_optimal_pipeline.keys():
                                final_count = final_count + 1
                                final_count_str = str(final_count)
                                final_pipeline[final_count_str] = {}
                                final_pipeline[final_count_str] = task_optimal_pipeline[top]
                    except Exception as e:
                        logging.error("""[Backend error] Execution phase of {} failed with:\n{}""".format(
                            self._steps[s].step_json, e))
                        logging.debug("Traceback: {}.".format(traceback.format_exc()))
                        break
                    logging.info('LOG: Pipeline - {desc} - Runtime: {time} seconds.'.format(
                        desc=self._steps[s].step_description,
                        time=time.time() - start))
                    logging.info("LOG: Pipeline - {desc} - End ({curr}/{tot})".format(
                        desc=self._steps[s].step_description,
                        curr=str(int(s) + 1),
                        tot=len(self._steps)))
                else:
                    final_count = final_count + 1
                    final_count_str = str(final_count)
                    final_pipeline[final_count_str] = {}
                    final_pipeline[final_count_str] = self._steps[s]
            except Exception as e:
                logging.error("""[Backend error] setup phase of {} failed with:\n{}""".format(
                    self._steps[s].step_json, e))
                logging.debug("Traceback: {}.".format(traceback.format_exc()))
        self.__parse_pipeline_steps(pipeline=final_pipeline, initial=False)

        # Writing on disk the actual/final pipeline (for info and reuse in Raidionics)
        executed_pipeline_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "executed_pipeline.json")
        with open(executed_pipeline_fn, 'w', newline='\n') as outfile:
            json.dump(final_pipeline, outfile, indent=4)
        return patient_parameters

    def execute(self, patient_parameters):
        logging.info('LOG: Pipeline - {} steps.'.format(len(self._steps)))
        for s in list(self._steps.keys()):
            start = time.time()
            logging.info("LOG: Pipeline - {desc} - Begin ({curr}/{tot})".format(desc=self._steps[s].step_description,
                                                                                curr=str(int(s) + 1),
                                                                                tot=len(self._steps)))
            try:
                self._steps[s].setup(patient_parameters)
            except Exception as e:
                logging.error("""[Backend error] Setup phase of {} failed with:\n{}""".format(
                    self._steps[s].step_json, e))
                logging.debug("Traceback: {}.".format(traceback.format_exc()))
                break
            try:
                patient_parameters = self._steps[s].execute()
            except Exception as e:
                logging.error("""[Backend error] Execution phase of {} failed with:\n{}""".format(
                    self._steps[s].step_json, e))
                logging.debug("Traceback: {}.".format(traceback.format_exc()))
                break
            logging.info('LOG: Pipeline - {desc} - Runtime: {time} seconds.'.format(desc=self._steps[s].step_description,
                                                                                    time=time.time() - start))
            logging.info("LOG: Pipeline - {desc} - End ({curr}/{tot})".format(desc=self._steps[s].step_description,
                                                                              curr=str(int(s) + 1),
                                                                              tot=len(self._steps)))
        return patient_parameters

    def cleanup(self):
        for s in list(self._steps.keys()):
            self._steps[s].cleanup()