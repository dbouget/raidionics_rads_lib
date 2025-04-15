import os
import shutil
import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from ..Utils.ants_registration import *
from ..Processing.brain_processing import *
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType
from ..Utils.DataStructures.RegistrationStructure import Registration


class RegistrationStep(AbstractPipelineStep):
    """
    @TODO. Have to improve the different use-cases, and properly deal with potentially more than one atlas.
    @TODO. Should there be two flags (one for the co-registration between sequences, and one for the co-registration towards an atlas?
    """
    _patient_parameters = None  # Placeholder for all patient related data
    _moving_volume_uid = None  # Internal unique identifier for the radiological volume to register
    _fixed_volume_uid = None  # Internal unique identifier for the radiological volume to use as registration target
    _registration_method = None  # Unused for now, might be more than just SyN in the future?
    _moving_volume_filepath = None
    _fixed_volume_filepath = None
    _moving_mask_filepath = None
    _fixed_mask_filepath = None
    _registration_runner = None

    def __init__(self, step_json: dict):
        super(RegistrationStep, self).__init__(step_json=step_json)
        self.__reset()
        self._registration_runner = ANTsRegistration()

    def __reset(self):
        self._patient_parameters = None
        self._moving_volume_uid = None
        self._fixed_volume_uid = None
        self._registration_method = None
        self._moving_volume_filepath = None
        self._fixed_volume_filepath = None
        self._registration_runner = None
        self._moving_mask_filepath = None
        self._fixed_mask_filepath = None

    @property
    def moving_volume_uid(self) -> str:
        return self._moving_volume_uid

    @moving_volume_uid.setter
    def moving_volume_uid(self, uid: str) -> None:
        self._moving_volume_uid = uid

    @property
    def fixed_volume_uid(self) -> str:
        return self._fixed_volume_uid

    @fixed_volume_uid.setter
    def fixed_volume_uid(self, uid: str) -> None:
        self._fixed_volume_uid = uid

    def setup(self, patient_parameters):
        """

        """
        self._patient_parameters = patient_parameters
        try:
            if ResourcesConfiguration.getInstance().predictions_use_registered_data and self._step_json["fixed"]["sequence"] != 'MNI':
                self.skip = True
                return
            # Check if a registration instance exists for this combination
            if self._patient_parameters.get_registration_by_json(fixed=self._step_json["fixed"],
                                                                 moving=self._step_json["moving"]) is not None:
                self.skip = True
                return
            moving_volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=self._step_json["moving"]["timestamp"],
                                                                                     sequence=self._step_json["moving"]["sequence"])
            if moving_volume_uid != "-1":
                self.moving_volume_uid = moving_volume_uid
                self._moving_volume_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).usable_input_filepath
            elif self._step_json["moving"]["timestamp"] == -1:  # Atlas file
                self._moving_volume_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
            else:
                raise ValueError("[RegistrationStep] Requested registration moving input cannot be found for: {}".format(self._step_json["moving"]))
            if not os.path.exists(self._moving_volume_filepath):
                raise ValueError("[RegistrationStep] Registration moving input cannot be found on disk with: {}".format(self._moving_volume_filepath))

            fixed_volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=self._step_json["fixed"]["timestamp"],
                                                                                    sequence=self._step_json["fixed"]["sequence"])
            if fixed_volume_uid != "-1":
                self._fixed_volume_uid = fixed_volume_uid
                self._fixed_volume_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self._fixed_volume_uid).usable_input_filepath
            elif self._step_json["fixed"]["timestamp"] == -1:  # Atlas file
                self._fixed_volume_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
            else:
                raise ValueError("[RegistrationStep] Requested registration fixed input cannot be found for: {}".format(self._step_json["fixed"]))
            if not os.path.exists(self._fixed_volume_filepath):
                raise ValueError(f"[RegistrationStep] Registration fixed input cannot be found on "
                                 f"disk with: {self._fixed_volume_filepath}")

            # Checking if a manual registered file for the given combination was provided by the user
            moving_volume = self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid)
            if moving_volume.is_registered_volume_included(destination_space_uid=self._fixed_volume_uid):
                self.skip = True
                return
            self.skip = False
        except Exception as e:
            self.skip = True
            raise ValueError(f"[RegistrationStep] Setting up process failed with: {e}.")

    def execute(self):
        """

        """
        if self.skip:
            if ResourcesConfiguration.getInstance().predictions_use_registered_data and self._step_json["fixed"][
                "sequence"] != 'MNI':
                logging.info("Skipping registration - not necessary since using co-registered inputs")
                return self._patient_parameters
            elif self._patient_parameters.get_registration_by_json(fixed=self._step_json["fixed"],
                                                                 moving=self._step_json["moving"]) is not None:
                logging.info("Skipping registration - already existing")
                return self._patient_parameters
            elif self.inclusion == "optional":
                logging.info("Skipping registration - no matching input was found for the patient.")
                return self._patient_parameters
            else:
                logging.info("Skipping registration - either manually provided registered image for this instance, "
                             "or no matching input was found for the patient.")
                return self._patient_parameters

        try:
            fmf, mmf = self.__registration_preprocessing()
            self.__registration(fmf, mmf)
        except Exception as e:
            raise ValueError(f"[RegistrationStep] Process failed to run with: {e}.")

        return self._patient_parameters

    def cleanup(self):
        self._registration_runner.clear_cache()

    def __registration_preprocessing(self):
        """
        Generating masked version of both the fixed and moving inputs, for occluding irrelevant structures.
        For example the region outside the brain/lungs, or areas exhibiting cancer expressions.
        """
        fixed_masked_filepath = None
        moving_masked_filepath = None
        try:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                if self.fixed_volume_uid:
                    brain_anno = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(self.fixed_volume_uid, AnnotationClassType.Brain)
                    if len(brain_anno) != 0:
                        self._fixed_mask_filepath = self._patient_parameters.get_annotation(annotation_uid=brain_anno[0]).usable_input_filepath
                else:
                    self._fixed_mask_filepath = ResourcesConfiguration.getInstance().mni_atlas_brain_mask_filepath

                if self.moving_volume_uid:
                    brain_anno = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(self.moving_volume_uid, AnnotationClassType.Brain)
                    if len(brain_anno) != 0:
                        self._moving_mask_filepath = self._patient_parameters.get_annotation(annotation_uid=brain_anno[0]).usable_input_filepath
                else:
                    self._moving_mask_filepath = ResourcesConfiguration.getInstance().mni_atlas_brain_mask_filepath

                moving_masked_filepath = perform_brain_masking(image_filepath=self._moving_volume_filepath,
                                                               mask_filepath=self._moving_mask_filepath,
                                                               output_folder=self._registration_runner.registration_folder)
                fixed_masked_filepath = perform_brain_masking(image_filepath=self._fixed_volume_filepath,
                                                              mask_filepath=self._fixed_mask_filepath,
                                                              output_folder=self._registration_runner.registration_folder)
                return fixed_masked_filepath, moving_masked_filepath
        except Exception as e:
            raise ValueError(f"Preprocessing step failed to proceed with: {e}.")

    def __registration(self, fixed_filepath, moving_filepath):
        try:
            registration_method = 'SyN'
            logging.info("[RegistrationStep] Using {} ANTs backend.".format(ResourcesConfiguration.getInstance().system_ants_backend))
            if ResourcesConfiguration.getInstance().system_ants_backend == "cpp":
                logging.info("[RegistrationStep] ANTs root located in {}.".format(ResourcesConfiguration.getInstance().ants_root))
            try:
                self._registration_runner.compute_registration(fixed=fixed_filepath, moving=moving_filepath,
                                                               registration_method=registration_method)
            except Exception as e:
                raise RuntimeError(f"ANTs execution code failed with: {e}")

            non_available_uid = True
            reg_uid = None
            while non_available_uid:
                reg_uid = 'R' + str(np.random.randint(0, 10000))
                if reg_uid not in self._patient_parameters.get_all_annotations_uids():
                    non_available_uid = False

            if self.fixed_volume_uid is None:
                self.fixed_volume_uid = 'MNI'
            if self.moving_volume_uid is None:
                self.moving_volume_uid = 'MNI'

            registration = Registration(uid=reg_uid, fixed_uid=self.fixed_volume_uid, moving_uid=self.moving_volume_uid,
                                        fwd_paths=self._registration_runner.reg_transform['fwdtransforms'],
                                        inv_paths=self._registration_runner.reg_transform['invtransforms'],
                                        output_folder=ResourcesConfiguration.getInstance().output_folder)
            self._patient_parameters.include_registration(reg_uid, registration)
            self._registration_runner.clear_cache()
        except Exception as e:
            self._registration_runner.clear_cache()
            raise ValueError(f"[RegistrationStep] Registration failed with: {e}.")
