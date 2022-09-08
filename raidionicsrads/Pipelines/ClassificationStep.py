import os
import shutil

import nibabel as nib
import pandas as pd
import logging
import configparser
import traceback
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep


class ClassificationStep(AbstractPipelineStep):
    _input_volume_uid = None
    _model_name = None
    _patient_parameters = None
    _input_volume_filepath = None

    def __init__(self, step_json: dict):
        super(ClassificationStep, self).__init__(step_json=step_json)
        self.__reset()
        self._model_name = self._step_json["target"]

    def __reset(self):
        self._input_volume_uid = None
        self._model_name = None
        self._patient_parameters = None
        self._input_volume_filepath = None

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters

    def execute(self):
        if len(self._step_json["input"].keys()) == 0:
            for volume in self._patient_parameters._radiological_volumes.keys():
                self._input_volume_uid = volume
                self._input_volume_filepath = self._patient_parameters._radiological_volumes[volume]._usable_input_filepath
                self.__perform_classification()
        else:
            # self._step_json["input"] # @TODO. Have to query the proper radiological volume from the info in there.
            self._input_volume_uid = ""
            self._input_volume_filepath = ""
            self.__perform_classification()
        return self._patient_parameters

    def __perform_classification(self) -> None:
        """

        """
        if False:  # @TODO. Might be a flag to not perform the classification if the user did the work manually
            return
        else:
            classification_config_filename = ""
            try:
                tmp_dir = os.path.join(ResourcesConfiguration.getInstance().output_folder, "classification_tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                classification_config = configparser.ConfigParser()
                classification_config.add_section('System')
                classification_config.set('System', 'gpu_id', ResourcesConfiguration.getInstance().gpu_id)
                classification_config.set('System', 'input_filename', self._input_volume_filepath)
                classification_config.set('System', 'output_folder', tmp_dir)
                classification_config.set('System', 'model_folder', os.path.join(ResourcesConfiguration.getInstance().model_folder, self._model_name))
                classification_config.add_section('Runtime')
                classification_config.set('Runtime', 'reconstruction_method', 'thresholding')
                classification_config.set('Runtime', 'reconstruction_order', 'resample_first')
                classification_config.add_section('Neuro')
                classification_config.set('Neuro', 'brain_segmentation_filename', '')
                classification_config_filename = os.path.join(tmp_dir, 'classification_config.ini')
                with open(classification_config_filename, 'w') as outfile:
                    classification_config.write(outfile)

                log_level = logging.getLogger().level
                log_str = 'warning'
                if log_level == 10:
                    log_str = 'debug'
                elif log_level == 20:
                    log_str = 'info'
                elif log_level == 40:
                    log_str = 'error'

                from raidionicsseg.fit import run_model
                run_model(classification_config_filename)
            except Exception as e:
                logging.error("Automatic classification failed with: {}.".format(traceback.format_exc()))
                if os.path.exists(classification_config_filename):
                    os.remove(classification_config_filename)
                raise ValueError("Impossible to perform classification.")

            # @TODO. Must read the classification results and populate the patient parameters.
            classification_results_filename = os.path.join(tmp_dir, 'classification-results.csv')
            classification_results_df = pd.read_csv(classification_results_filename)
            final_class = classification_results_df.values[classification_results_df[classification_results_df.columns[1]].idxmax(), 0]
            self._patient_parameters._radiological_volumes[self._input_volume_uid].set_sequence_type(final_class)

            shutil.rmtree(tmp_dir)
            return

