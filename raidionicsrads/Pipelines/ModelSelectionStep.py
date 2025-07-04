import os
import shutil
from copy import deepcopy

import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
import json

from ..Utils.utilities import get_type_from_string, get_type_from_enum_name
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.PatientStructure import PatientParameters
from ..Utils.DataStructures.AnnotationStructure import Annotation, AnnotationClassType, BrainTumorType


class ModelSelectionStep(AbstractPipelineStep):
    """
    For each model, a subset of models has been trained based on the provided inputs.
    The identification of the best fitting model for the current patient is performed here and the corresponding
    pipeline json file is generated matching the required inputs.

    @TODO. Should this step be generalized in case the targeted input or timestamp has to be adjusted on-the-fly
    from the default pipeline.json found on disk?
    """
    _base_model_name = None  # Basename of the folder containing all the sub-models to choose from.
    _patient_parameters = None  # Overall patient parameters, updated on-the-fly
    _working_folder = None  # Temporary directory on disk to store inputs/outputs for the segmentation
    _target_timestamp = None # Timestamp for the inputs to run the model on (if on-the-fly adaptation is needed)
    _predictions_format = None

    def __init__(self, step_json: dict):
        super(ModelSelectionStep, self).__init__(step_json=step_json)
        self.__reset()
        step_keys = list(self._step_json.keys())
        self._base_model_name = self._step_json["model"] if "model" in step_keys else None
        self._target_timestamp = int(self._step_json["timestamp"]) if "timestamp" in step_keys else None
        self._predictions_format = self._step_json["format"] if "format" in step_keys else None
        self.sequences_names_intern = ["T1-CE", "T1-w", "FLAIR", "T2", "High-resolution"]
        self.sequences_names_models = ["t1c", "t1w", "t2f", "t2w", "hr"]

    def __reset(self):
        self._base_model_name = None
        self._patient_parameters = None
        self._working_folder = None
        self._target_timestamp = None
        self._predictions_format = None

    @property
    def target_timestamp(self) -> int:
        return self._target_timestamp

    @target_timestamp.setter
    def target_timestamp(self, ts: int) -> None:
        self._target_timestamp = ts

    def setup(self, patient_parameters: PatientParameters) -> None:
        """
        Sanity check that all requirements are met for running the model selection step.

        Parameters
        ----------
        patient_parameters: PatientParameters
            Placeholder for the current patient data, which will be updated with the results of this step.
        """
        self._patient_parameters = patient_parameters

        self._working_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, "modelselection_tmp")
        os.makedirs(self._working_folder, exist_ok=True)
        try:
            base_model_path = os.path.join(ResourcesConfiguration.getInstance().model_folder, self._base_model_name)
            if self._base_model_name is None or not os.path.exists(base_model_path) or not os.path.isdir(base_model_path):
                raise ValueError(f"Provided input model directory does not exist on disk with value {base_model_path}")
        except Exception as e:
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            self.skip = True
            raise ValueError(f"[ModelSelectionStep] setup failed with: {e}.")

    def execute(self) -> {}:
        """
        Executes the current step.

        Returns
        -------
        dict
            Dictionary containing all the steps for the selected model, which will be appended to the rest of the
            existing pipeline.
        """
        if self.skip:
            logging.info("Model selection step skipped, no matching combination of available models and"
                         " provided inputs was found!")
            return

        try:
            base_model_path = os.path.join(ResourcesConfiguration.getInstance().model_folder, self._base_model_name)
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                model_name = self.__identify_model_from_mri_inputs(base_model_path=base_model_path)
            else:
                model_name = self.__identify_model_from_ct_inputs(base_model_path=base_model_path)
        except Exception as e:
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError(f"[ModelSelectionStep] failed with: {e}.")

        if model_name is None:
            raise ValueError(f"[ModelSelectionStep] failed, no model could be selected.")
        model_pipeline_fn = os.path.join(ResourcesConfiguration.getInstance().model_folder, model_name, "pipeline.json")
        model_pipeline = None
        with open(model_pipeline_fn, 'r') as infile:
            model_pipeline = json.load(infile)
        # @TODO. Same issue for the brain model that can segment over any MR sequence, have to find a way to adjust the
        # value on-the-fly here (most likely only use-case when coming from Raidionics). Or should Raidionics deal with it?

        if self.target_timestamp is not None:
            # If the timestamp is left unspecified (i.e., for some generic models (brain, flair changes), it should be
            # modified inside the pipeline based on self.timestamp.
            adjusted_model_pipeline = deepcopy(model_pipeline)
            for st in list(model_pipeline.keys()):
                if model_pipeline[st]["task"] in ["Segmentation", "Segmentation refinement"]:
                    for i in list(model_pipeline[st]["inputs"].keys()):
                        if model_pipeline[st]["inputs"][i]["timestamp"] == -1:
                            adjusted_model_pipeline[st]["inputs"][i]["timestamp"] = self.target_timestamp
                        if model_pipeline[st]["inputs"][i]["space"]["timestamp"] == -1:
                            adjusted_model_pipeline[st]["inputs"][i]["space"]["timestamp"] = self.target_timestamp
                    if "format" not in list(model_pipeline[st].keys()):
                        adjusted_model_pipeline[st]["format"] = self._predictions_format
                elif model_pipeline[st]["task"] in ["Registration", "Apply registration"]:
                    adjusted_model_pipeline[st]["moving"]["timestamp"] = self.target_timestamp
                    adjusted_model_pipeline[st]["fixed"]["timestamp"] = self.target_timestamp
                adjusted_model_pipeline[st]["inclusion"] = self.inclusion
            model_pipeline = adjusted_model_pipeline

        if os.path.exists(self._working_folder):
            shutil.rmtree(self._working_folder)

        return model_pipeline

    def cleanup(self):
        if self._working_folder is not None and os.path.exists(self._working_folder):
            shutil.rmtree(self._working_folder)

    def __identify_model_from_mri_inputs(self, base_model_path: str) -> str:
        """
        For each model, a subset of models has been trained based on the provided inputs.
        The identification of the best fitting model for the current patient is performed here.

        Returns
        -------
        str
            The folder name on disk matching best the set of inputs for the current patient.
        """
        try:
            final_model_name = None
            # Have to check first if the model is eligible (e.g., MRI_Brain is not)
            if self._base_model_name.split('_')[0] == "CT" or self._base_model_name == "MRI_Brain":
                return base_model_path
            else:
                # For eligible models, list the submodels and match to list of available inputs.
                eligible_models = []
                for _, dirs, _ in os.walk(base_model_path):
                    for d in dirs:
                        eligible_models.append(d)
                    break

            eligible_models = sorted(eligible_models, key=len)
            timestamp = self.target_timestamp

            existing_inputs = self._patient_parameters.get_all_radiological_volumes_for_timestamp(timestamp=timestamp)
            existing_sequences = sorted(np.unique([i.get_sequence_type_str() for i in existing_inputs]))
            for m in eligible_models:
                mseq = m.split('_')
                complete = True
                for s in mseq:
                    if s == "t1c" and "T1-CE" not in existing_sequences:
                        complete = False
                        break
                    elif s == "t1w" and "T1-w" not in existing_sequences:
                        complete = False
                        break
                    elif s == "t2f" and "FLAIR" not in existing_sequences:
                        complete = False
                        break
                    elif s == "t2w" and "T2" not in existing_sequences:
                        complete = False
                        break
                    elif s in ["t1d", "t2d"]:
                        continue
                if complete:
                    final_model_name = os.path.join(self._base_model_name, m)
            if final_model_name is None:
                raise ValueError("Could not identify any model matching the set of MR scan inputs.")
        except Exception as e:
            raise ValueError(e)
        return final_model_name

    def __identify_model_from_ct_inputs(self, base_model_path: str) -> str:
        """

        Parameters
        ----------
        base_model_path

        Returns
        -------

        """
        try:
            final_model_name = None
            # Have to check first if the model is eligible (e.g., MRI models are not)
            if self._base_model_name.split('_')[0] == "MRI":
                return base_model_path
            else:
                # For eligible models, list the submodels and match to list of available inputs.
                eligible_models = []
                for _, dirs, _ in os.walk(base_model_path):
                    for d in dirs:
                        eligible_models.append(d)
                    break

            eligible_models = sorted(eligible_models, key=len)
            timestamp = self.target_timestamp

            existing_inputs = self._patient_parameters.get_all_radiological_volumes_for_timestamp(timestamp=timestamp)
            existing_sequences = sorted(np.unique([i.get_sequence_type_str() for i in existing_inputs]))
            for m in eligible_models:
                mseq = m.split('_')
                complete = True
                for s in mseq:
                    if s == "hr" and "High-resolution" not in existing_sequences:
                        complete = False
                        break
                if complete:
                    final_model_name = os.path.join(self._base_model_name, m)
            if final_model_name is None:
                raise ValueError("Could not identify any model matching the set of MR scan inputs.")
        except Exception as e:
            raise ValueError(e)
        return final_model_name
