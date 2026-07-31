import json
from abc import ABC, abstractmethod


class AbstractPipelineStep(ABC):
    _step_json = {}
    _step_description = None
    _skip = False  # Boolean flag indicating if the step should be skipped during the pipeline execution phase
    _inclusion = "required"  # A step can have two modes: required or optional
    _dry_run = False  # Boolean flag for running the pipeline in preview mode where no computation is performed

    def __init__(self, step_json: dict) -> None:
        self._reset()
        self._step_json = step_json
        self._step_description = step_json["description"]
        self._inclusion = step_json["inclusion"] if "inclusion" in step_json.keys() else "required"

    def _reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._step_json = {}
        self._step_description = None
        self._skip = False
        self._dry_run = False

    @property
    def step_json(self) -> dict:
        return self._step_json

    @step_json.setter
    def step_json(self, step: dict) -> None:
        self._step_json = step

    @property
    def step_description(self) -> dict:
        return self._step_description

    @property
    def skip(self) -> bool:
        return self._skip

    @skip.setter
    def skip(self, state: bool) -> None:
        self._skip = state

    @property
    def inclusion(self) -> str:
        return self._inclusion

    @inclusion.setter
    def inclusion(self, value: str) -> None:
        if value in ["required", "optional"]:
            self._inclusion = value

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, state: bool) -> None:
        self._dry_run = state

    @abstractmethod
    def setup(self, patient_parameters):
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def get_task(self) -> str:
        return self._step_json["task"] if "task" in self._step_json.keys() else None
    # @TODO. Should there be a step assess method, before executing the pipeline, ensuring all inputs are available
    # for all steps.
