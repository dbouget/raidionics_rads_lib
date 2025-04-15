import os
import shutil
from copy import deepcopy

import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
import json
from typing import List

from ..Utils.utilities import get_type_from_string, get_type_from_enum_name
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.PatientStructure import PatientParameters
from ..Utils.DataStructures.AnnotationStructure import Annotation, AnnotationClassType, BrainTumorType


class ReportingSelectionStep(AbstractPipelineStep):
    """


    """
    _patient_parameters = None  # Overall patient parameters, updated on-the-fly
    _scope_reporting = None # Type of reporting, either standalone (at a single timestamp) or surgical (pre/post)
    _timestamps = None

    def __init__(self, step_json: dict):
        super(ReportingSelectionStep, self).__init__(step_json=step_json)
        self.__reset()
        step_keys = list(self._step_json.keys())
        self._scope_reporting = self._step_json["scope"] if "scope" in step_keys else None
        self._timestamps = self._step_json["timestamps"] if "timestamps" in step_keys else []

    def __reset(self):
        self._patient_parameters = None
        self._scope_reporting = None

    @property
    def scope_reporting(self) -> str:
        return self._scope_reporting

    @scope_reporting.setter
    def scope_reporting(self, value: str) -> None:
        self._scope_reporting = value

    @property
    def timestamps(self) -> List[int]:
        return self._timestamps

    @timestamps.setter
    def timestamps(self, value: List[int]) -> None:
        self._timestamps = value

    def setup(self, patient_parameters: PatientParameters) -> None:
        """
        Sanity check that all requirements are met for running the model selection step.

        Parameters
        ----------
        patient_parameters: PatientParameters
            Placeholder for the current patient data, which will be updated with the results of this step.
        """
        self._patient_parameters = patient_parameters
        try:
            if self.scope_reporting is None or self.scope_reporting not in ["standalone", "surgical"]:
                raise ValueError(f"Provided scope for identifying the reporting setup is not recognized,"
                                 f" with {self.scope_reporting}")
        except Exception as e:
            raise ValueError(f"[ReportingSelectionStep] setup failed with: {e}.")

    def execute(self) -> {}:
        """
        Executes the current step.

        Returns
        -------
        dict
            Dictionary containing all the steps for the selected model, which will be appended to the rest of the
            existing pipeline.
        """
        try:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                reporting_pipeline = self.__generate_neuro_reporting_pipeline()
            else:
                reporting_pipeline = self.__generate_mediastinum_reporting_pipeline()
        except Exception as e:
            raise ValueError(f"[ReportingSelectionStep] failed with: {e}.")

        return reporting_pipeline

    def cleanup(self):
        pass

    def __generate_neuro_reporting_pipeline(self) -> dict:
        pip = {}
        pip_num_int = 0
        try:
            if self.scope_reporting == "standalone":
                pip_num_int = pip_num_int + 1
                pip_num = str(pip_num_int)
                pip[pip_num] = {}
                pip[pip_num]["task"] = 'Registration'
                pip[pip_num]["moving"] = {}
                pip[pip_num]["moving"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["moving"]["sequence"] = "FLAIR"
                pip[pip_num]["fixed"] = {}
                pip[pip_num]["fixed"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["fixed"]["sequence"] = "T1-CE"
                pip[pip_num]["inclusion"] = "optional"
                pip[pip_num]["description"] = "Registration from FLAIR (T0) to T1CE (T0)"


                pip_num_int = pip_num_int + 1
                pip_num = str(pip_num_int)
                pip[pip_num] = {}
                pip[pip_num]["task"] = 'Apply registration'
                pip[pip_num]["moving"] = {}
                pip[pip_num]["moving"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["moving"]["sequence"] = "FLAIR"
                pip[pip_num]["fixed"] = {}
                pip[pip_num]["fixed"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["fixed"]["sequence"] = "T1-CE"
                pip[pip_num]["direction"] = "forward"
                pip[pip_num]["inclusion"] = "optional"
                pip[pip_num]["description"] = "Apply registration from FLAIR (T0) to T1CE (T0)"

                pip_num_int = pip_num_int + 1
                pip_num = str(pip_num_int)
                pip[pip_num] = {}
                pip[pip_num]["task"] = 'Segmentation refinement'
                pip[pip_num]["inputs"] = {}
                pip[pip_num]["inputs"]["0"] = {}
                pip[pip_num]["inputs"]["0"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["inputs"]["0"]["sequence"] = "T1-CE"
                pip[pip_num]["inputs"]["0"]["labels"] = "Tumor"
                pip[pip_num]["inputs"]["0"]["space"] = {}
                pip[pip_num]["inputs"]["0"]["space"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["inputs"]["0"]["space"]["sequence"] = "T1-CE"
                pip[pip_num]["operation"] = "global_context"
                pip[pip_num]["args"] = {}
                pip[pip_num]["description"] = "Global segmented structures context refinement in T1CE (T0)"

                pip_num_int = pip_num_int + 1
                pip_num = str(pip_num_int)
                pip[pip_num] = {}
                pip[pip_num]["task"] = 'Registration'
                pip[pip_num]["moving"] = {}
                pip[pip_num]["moving"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["moving"]["sequence"] = "T1-CE"
                pip[pip_num]["fixed"] = {}
                pip[pip_num]["fixed"]["timestamp"] = -1
                pip[pip_num]["fixed"]["sequence"] = "MNI"
                pip[pip_num]["description"] = "Registration from T1CE (T0) to MNI"

                pip_num_int = pip_num_int + 1
                pip_num = str(pip_num_int)
                pip[pip_num] = {}
                pip[pip_num]["task"] = 'Apply registration'
                pip[pip_num]["moving"] = {}
                pip[pip_num]["moving"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["moving"]["sequence"] = "T1-CE"
                pip[pip_num]["fixed"] = {}
                pip[pip_num]["fixed"]["timestamp"] = -1
                pip[pip_num]["fixed"]["sequence"] = "MNI"
                pip[pip_num]["direction"] = "forward"
                pip[pip_num]["description"] = "Apply registration from T1CE (T0) to MNI"

                pip_num_int = pip_num_int + 1
                pip_num = str(pip_num_int)
                pip[pip_num] = {}
                pip[pip_num]["task"] = 'Apply registration'
                pip[pip_num]["moving"] = {}
                pip[pip_num]["moving"]["timestamp"] = self.timestamps[0]
                pip[pip_num]["moving"]["sequence"] = "T1-CE"
                pip[pip_num]["fixed"] = {}
                pip[pip_num]["fixed"]["timestamp"] = -1
                pip[pip_num]["fixed"]["sequence"] = "MNI"
                pip[pip_num]["direction"] = "inverse"
                pip[pip_num]["description"] = "Apply inverse registration from MNI to T1CE (T0)"

                pip_num_int = pip_num_int + 1
                pip_num = str(pip_num_int)
                pip[pip_num] = {}
                pip[pip_num]["task"] = "Features computation"
                pip[pip_num]["timestamp"] = self.timestamps[0]
                pip[pip_num]["target"] = ["Tumor"]
                pip[pip_num]["space"] = "MNI"
                pip[pip_num]["description"] = "Standardized features computation for timestamp 0"
        except Exception as e:
            raise ValueError(f"{e}")
        return pip

    def __generate_mediastinum_reporting_pipeline(self) -> dict:
        pass