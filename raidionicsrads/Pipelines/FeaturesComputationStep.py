import json
from .AbstractPipelineStep import AbstractPipelineStep
from ..NeuroDiagnosis.neuro_diagnostics import compute_statistics


class FeaturesComputationStep(AbstractPipelineStep):
    _patient_parameters = None

    def __init__(self, step_json: dict) -> None:
        super(FeaturesComputationStep, self).__init__(step_json=step_json)
        self.__reset()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._patient_parameters = None

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters

    def execute(self):
        return self._patient_parameters
