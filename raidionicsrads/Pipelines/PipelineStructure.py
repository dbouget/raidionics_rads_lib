import json
import logging

from aenum import Enum, unique
from ..Utils.utilities import get_type_from_string
from .ClassificationStep import ClassificationStep
from .SegmentationStep import SegmentationStep
from .RegistrationStep import RegistrationStep
from .RegistrationDeployerStep import RegistrationDeployerStep


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

    def __str__(self):
        return self.string


class Pipeline:
    """
    Class defining how an MRI volume should be handled.
    """
    _input_filepath = ""
    _pipeline_json = {}
    _steps = {}

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
        with open(self._input_filepath, 'r') as infile:
            self._pipeline_json = json.load(infile)

        for i, s in enumerate(list(self._pipeline_json.keys())):
            task = get_type_from_string(TaskType, self._pipeline_json[s]["task"])
            step = None
            if task == TaskType.Class:
                step = ClassificationStep(self._pipeline_json[s])
            elif task == TaskType.Seg:
                step = SegmentationStep(self._pipeline_json[s])
            elif task == TaskType.Reg:
                step = RegistrationStep(self._pipeline_json[s])
            elif task == TaskType.AReg:
                step = RegistrationDeployerStep(self._pipeline_json[s])

            if step:
                self._steps[str(i)] = step
            else:
                logging.warning("Step dismissed because task could not be matched.")

    def execute(self, patient_parameters):
        logging.info('LOG: Reporting - {} steps.'.format(len(self._steps)))
        for s in list(self._steps.keys()):
            logging.info("LOG: Reporting - {desc} - Begin ({curr}/{tot})".format(desc=self._pipeline_json[str(int(s)+1)]['description'], curr=s, tot=len(self._steps)))
            self._steps[s].setup(patient_parameters)
            patient_parameters = self._steps[s].execute()
            logging.info("LOG: Reporting - {desc} - End ({curr}/{tot})".format(desc=self._pipeline_json[str(int(s) + 1)]['description'], curr=s, tot=len(self._steps)))

        return patient_parameters
