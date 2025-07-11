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
    """
    The only use-case is an MRI sequence type classifier for now, to run on all inputs as a preprocessing step.
    @TODO. Will have to be made more generic in the future, especially for tumor type classification soon.
    """
    _model_name = None
    _patient_parameters = None
    _working_folder = None
    _input_volume_uid = None
    _input_volume_filepath = None

    def __init__(self, step_json: dict):
        super(ClassificationStep, self).__init__(step_json=step_json)
        self.__reset()
        self._model_name = self._step_json["model"]

    def __reset(self):
        self._model_name = None
        self._patient_parameters = None
        self._working_folder = None
        self._input_volume_uid = None
        self._input_volume_filepath = None

    @property
    def working_folder(self) -> str:
        return self._working_folder

    @working_folder.setter
    def working_folder(self, f: str) -> None:
        self._working_folder = f

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters

        if self.skip:
            return

        self.working_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, "classification_tmp")
        os.makedirs(self.working_folder, exist_ok=True)
        os.makedirs(os.path.join(self.working_folder, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(self.working_folder, 'outputs'), exist_ok=True)

    def execute(self):
        try:
            if self.skip:
                logging.info(f"Classification step not executed since marked as skippable.")
                return self._patient_parameters

            if len(self._step_json["inputs"].keys()) == 0:
                for volume_uid in self._patient_parameters.get_all_radiological_volume_uids():
                    self._input_volume_uid = volume_uid
                    self._input_volume_filepath = self._patient_parameters.get_radiological_volume(volume_uid=volume_uid).usable_input_filepath
                    new_fp = os.path.join(self.working_folder, 'inputs', 'input0.nii.gz')
                    shutil.copyfile(self._input_volume_filepath, new_fp)
                    self.__perform_classification()
            elif len(self._step_json["inputs"].keys()) == 1:
                # Brain tumor type classification use-case
                input_json = self._step_json["inputs"]["0"]

                if ((input_json["space"]["timestamp"] == input_json["timestamp"] and
                     input_json["space"]["sequence"] == input_json["sequence"]) or
                        ResourcesConfiguration.getInstance().predictions_use_registered_data):
                    volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                                                      sequence=input_json["sequence"])
                    if volume_uid == "-1":
                        raise ValueError("No radiological volume for {}.".format(input_json))
                    self._input_volume_uid = volume_uid
                    self._input_volume_filepath = self._patient_parameters.get_radiological_volume(volume_uid=volume_uid).usable_input_filepath
                    new_fp = os.path.join(self.working_folder, 'inputs', 'input0.nii.gz')
                    shutil.copyfile(self._input_volume_filepath, new_fp)
                    self.__perform_classification()
            else:  # Not a use-case for the moment.
                pass

            self.__process_classification_results()
        except Exception as e:
            if os.path.exists(self.working_folder):
                shutil.rmtree(self.working_folder)
            raise ValueError("[ClassificationStep] Automatic classification failed with: {}.".format(e))

        if os.path.exists(self.working_folder):
            shutil.rmtree(self.working_folder)

        return self._patient_parameters

    def cleanup(self):
        if self.working_folder is not None and os.path.exists(self.working_folder):
            shutil.rmtree(self.working_folder)

    def __perform_classification(self) -> None:
        """
        @TODO. Should hold the sequence class for each input, and dump a report/summary for Raidionics/the user.
        """
        try:
            tmp_dir = os.path.join(ResourcesConfiguration.getInstance().output_folder, "classification_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            classification_config = configparser.ConfigParser()
            classification_config.add_section('System')
            classification_config.set('System', 'gpu_id', ResourcesConfiguration.getInstance().gpu_id)
            classification_config.set('System', 'inputs_folder', os.path.join(self.working_folder, 'inputs'))
            classification_config.set('System', 'output_folder', os.path.join(self.working_folder, 'outputs'))
            classification_config.set('System', 'model_folder',
                                      os.path.join(ResourcesConfiguration.getInstance().model_folder, self._model_name))
            classification_config.add_section('Runtime')
            classification_config.set('Runtime', 'reconstruction_method', 'probabilities')
            classification_config.set('Runtime', 'reconstruction_order', 'resample_first')
            classification_config.add_section('Neuro')
            classification_config.set('Neuro', 'brain_segmentation_filename', '')
            classification_config_filename = os.path.join(os.path.join(self.working_folder, 'inputs'),
                                                          'classification_config.ini')
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
            raise ValueError(f"[ClassificationStep] Automatic classification failed with: {e}.")

        try:
            classification_results_filename = os.path.join(os.path.join(self.working_folder, 'outputs'),
                                                           'classification-results.csv')
            shutil.copyfile(classification_results_filename,
                            os.path.join(ResourcesConfiguration.getInstance().output_folder,
                                         self._step_json["target"][0] + '_classification_results_raw.csv'))
            classification_results_df = pd.read_csv(classification_results_filename)
            final_class = classification_results_df.values[classification_results_df[classification_results_df.columns[1]].idxmax(), 0]
            if self._step_json["target"][0] == "MRSequence":
                self._patient_parameters.get_radiological_volume(volume_uid=self._input_volume_uid).set_sequence_type(final_class)
            elif self._step_json["target"][0] == "BrainTumorType":
                # Can only store the brain tumor type info inside the patient report, later on
                pass
            else:
                raise ValueError(f"[ClassificationStep] Use-cases other than MRI sequence classification have not been"
                                 f" implemented yet!")
        except Exception as e:
            raise ValueError(f"[ClassificationStep] Classification results parsing failed with: {e}.")

        return

    def __process_classification_results(self) -> None:
        """
        Dumping the classification results for reloading afterwards
        Only/current use-case for now: MRI sequence classification.

        Returns
        -------

        """
        classification_results_filename = os.path.join(ResourcesConfiguration.getInstance().output_folder,
                                                       self._step_json["target"][0] + "_classification_results.csv")
        classes = []
        if self._step_json["target"][0] == "MRSequence":
            for volume_uid in self._patient_parameters.get_all_radiological_volume_uids():
                classes.append([os.path.basename(
                    self._patient_parameters.get_radiological_volume(volume_uid).raw_input_filepath),
                                self._patient_parameters.get_radiological_volume(volume_uid).get_sequence_type_str()])
            df = pd.DataFrame(classes, columns=['File', 'MRI sequence'])
            df.to_csv(classification_results_filename, index=False)
        elif self._step_json["target"][0] == "BrainTumorType":
            # The results file is on disk, ready to be used by to fill in the reporting.
            pass
        else:
            raise ValueError(f"[ClassificationStep] Use-cases other than MRI sequence classification and brain "
                             f" tumor type have not been implemented yet!")
        logging.info(f"Classification results written to {classification_results_filename}")