import os
import shutil
import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
import concurrent.futures
from tqdm import tqdm
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from ..Utils.ants_registration import *
from ..Processing.brain_processing import *
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType
from ..Utils.DataStructures.RegistrationStructure import Registration


class RegistrationDeployerStep(AbstractPipelineStep):
    """
    @TODO. Same issues to handle as in the main RegistrationStep.
    """
    _patient_parameters = None
    _moving_volume_uid = None
    _fixed_volume_uid = None
    _registration_runner = None
    _registration_instance = None
    _direction = None

    def __init__(self, step_json: dict):
        super(RegistrationDeployerStep, self).__init__(step_json=step_json)
        self.__reset()
        self._registration_runner = ANTsRegistration()
        self._direction = self._step_json["direction"]

    def __reset(self):
        self._patient_parameters = None
        self._moving_volume_uid = None
        self._fixed_volume_uid = None
        self._registration_runner = None
        self._registration_instance = None
        self._direction = None

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

    @property
    def registration_instance(self) -> Registration:
        return self._registration_instance

    @registration_instance.setter
    def registration_instance(self, registration: Registration) -> None:
        self._registration_instance = registration

    def setup(self, patient_parameters):
        """

        """
        try:
            self._patient_parameters = patient_parameters

            if (ResourcesConfiguration.getInstance().predictions_use_registered_data
                    and self._step_json["fixed"]["sequence"] != "MNI"):
                self.skip = True
                return

            moving_volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=self._step_json["moving"]["timestamp"],
                                                                                     sequence=self._step_json["moving"]["sequence"])
            fixed_volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=self._step_json["fixed"]["timestamp"],
                                                                                    sequence=self._step_json["fixed"]["sequence"])
            # Checking if a manual registered file for the given combination was provided by the user
            moving_volume = self._patient_parameters.get_radiological_volume(volume_uid=moving_volume_uid)
            # if moving_volume.is_registered_volume_included(destination_space_uid=fixed_volume_uid):
            #     self.skip = True
            #     return
            if moving_volume is None and self._inclusion == "optional":
                self.skip = True
                return

            self.registration_instance = self._patient_parameters.get_registration_by_json(fixed=self._step_json["fixed"],
                                                                                      moving=self._step_json["moving"])
            if self.registration_instance is None and moving_volume.is_registered_volume_included(destination_space_uid=fixed_volume_uid):
                self.skip = True
                return
            elif self.registration_instance is None:
                raise ValueError(f"No registration instance could be found between: {moving_volume_uid} and {fixed_volume_uid}")
            self.fixed_volume_uid = self.registration_instance.fixed_uid
            self.moving_volume_uid = self.registration_instance.moving_uid
            self._registration_runner.reg_transform['fwdtransforms'] = self.registration_instance.forward_filepaths
            self._registration_runner.reg_transform['invtransforms'] = self.registration_instance.inverse_filepaths
            self.skip = False
        except Exception as e:
            self.skip = True
            raise ValueError(f"[RegistrationDeployerStep] Step setup failed with: {e}.")

    def execute(self):
        """
        @TODO. Have to finish/improve how to deal with atlases, can't be the same process as regular volumes
        @TODO2. Have to take into account the registration direction from [forward, inverse].
        @TODO3. Have to deal with neuro/mediastinum, or more use-cases
        """
        try:
            if self.skip:
                if (ResourcesConfiguration.getInstance().predictions_use_registered_data
                        and self._step_json["fixed"]["sequence"] != "MNI"):
                    logging.debug("[RegistrationDeployerStep] Step skipped because pre-registered inputs are used.")
                    return self._patient_parameters
                elif self.inclusion == "optional":
                    logging.info("[RegistrationDeployerStep] Step skipped because no matching input was found for the "
                                 "patient and the step is optional.")
                    return self._patient_parameters
                else:
                    logging.debug("[RegistrationDeployerStep] Step skipped because manual registered input was provided.")
                    return self._patient_parameters

            if self._moving_volume_uid != 'MNI' and self._direction == 'forward':
                self.__apply_registration()
                self.__apply_registration_annotations()
            elif self._moving_volume_uid != 'MNI' and self._direction == 'inverse':
                self.__apply_registration_atlas_space() if ResourcesConfiguration.getInstance().num_workers == 1 else self.__apply_registration_atlas_space_parallel()

            self._registration_runner.clear_output_folder()
            return self._patient_parameters
        except Exception as e:
            raise ValueError(f"[RegistrationDeployerStep] Registration deployment failed with: {e}.")

    def cleanup(self):
        self._registration_runner.clear_output_folder()

    def __apply_registration(self):
        try:
            if self._patient_parameters.get_radiological_volume(volume_uid=self._moving_volume_uid).is_registered_volume_included(destination_space_uid=self.fixed_volume_uid):
                logging.info(f"Registered radiological volume already existing -- skipping the step")
                return

            fixed_filepath = None
            if self.fixed_volume_uid == 'MNI':
                fixed_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
            else:
                fixed_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self.fixed_volume_uid).usable_input_filepath

            moving_filepath = None
            if self.moving_volume_uid == 'MNI':
                moving_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
            else:
                moving_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).usable_input_filepath

            dest_base_folder = None
            if self.fixed_volume_uid == "MNI":
                dest_base_folder = self.fixed_volume_uid + '_space'
            else:
                dest_base_folder =   (self._patient_parameters.get_radiological_volume(volume_uid=self.fixed_volume_uid)._timestamp_id +
                                      '_'+ self._patient_parameters.get_radiological_volume(volume_uid=self.fixed_volume_uid).get_sequence_type_enum().name
                                      + '_space')
            fp = self._registration_runner.apply_registration_transform(moving=moving_filepath, fixed=fixed_filepath,
                                                                        interpolation='bSpline')
            new_fp = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                                  dest_base_folder, self.moving_volume_uid + '_Seq-' +
                                  self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid)._sequence_type.name +
                                  '_registered_to_' + self._fixed_volume_uid + '.nii.gz')
            os.makedirs(os.path.dirname(new_fp), exist_ok=True)
            shutil.copyfile(fp, new_fp)
            self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).include_registered_volume(filepath=new_fp,
                                                                                                                           registration_uid=self.registration_instance.unique_id,
                                                                                                                           destination_space_uid=self._fixed_volume_uid)
        except Exception as e:
            raise ValueError(f"Applying the registration on the radiological volume failed with: {e}.")

    def __apply_registration_annotations(self):
        try:
            fixed_filepath = None
            dest_base_folder = None
            if self.fixed_volume_uid == 'MNI':
                fixed_filepath = ResourcesConfiguration.getInstance().mni_atlas_filepath_T1
                dest_base_folder = self.fixed_volume_uid + '_space'
            else:
                fixed_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self.fixed_volume_uid).usable_input_filepath
                dest_base_folder =   (self._patient_parameters.get_radiological_volume(volume_uid=self.fixed_volume_uid)._timestamp_id +
                                      '_'+ self._patient_parameters.get_radiological_volume(volume_uid=self.fixed_volume_uid).get_sequence_type_enum().name
                                      + '_space')

            for anno in self._patient_parameters.get_all_annotations_uids_radiological_volume(volume_uid=self.moving_volume_uid):
                annotation = self._patient_parameters.get_annotation(annotation_uid=anno)
                if annotation.is_registered_volume_included(destination_space_uid=self.fixed_volume_uid):
                    logging.info(f"Registered annotation ({annotation.get_annotation_type_str()}) already existing -- skipping the step")
                    continue
                moving_filepath = annotation.usable_input_filepath

                fp = self._registration_runner.apply_registration_transform(moving=moving_filepath, fixed=fixed_filepath,
                                                                            interpolation='nearestNeighbor')
                new_fp = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                                      dest_base_folder,
                                      self.moving_volume_uid + '_label_' + annotation.get_annotation_type_name() +
                                      '_registered_to_' + self.fixed_volume_uid + '.nii.gz')
                os.makedirs(os.path.dirname(new_fp), exist_ok=True)
                shutil.copyfile(fp, new_fp)
                annotation.include_registered_volume(filepath=new_fp,
                                                     registration_uid=self.registration_instance.unique_id,
                                                     destination_space_uid=self.fixed_volume_uid)
            if self.fixed_volume_uid == 'MNI':
                # In addition, the other registered annotations towards the moving volume uid are parsed for an atlas
                # registration case. Only the extra annotations, not featured natively for the volume uid, are registered.
                for reganno in self._patient_parameters.get_all_registered_annotations_uids_radiological_volume(volume_uid=self.moving_volume_uid):
                    reg_annotation = self._patient_parameters.get_annotation(annotation_uid=reganno)
                    if len(self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=self.moving_volume_uid, annotation_class=reg_annotation.get_annotation_type_enum())) == 0:
                        if reg_annotation.is_registered_volume_included(destination_space_uid=self.fixed_volume_uid):
                            logging.info(
                                f"Registered annotation ({reg_annotation.get_annotation_type_str()}) already existing -- skipping the step")
                            continue
                        moving_filepath = reg_annotation.registered_volumes[self.moving_volume_uid]["filepath"]

                        fp = self._registration_runner.apply_registration_transform(moving=moving_filepath,
                                                                                    fixed=fixed_filepath,
                                                                                    interpolation='nearestNeighbor')
                        new_fp = os.path.join(self._patient_parameters.get_radiological_volume(
                            volume_uid=self.moving_volume_uid).output_folder,
                                              dest_base_folder,
                                              self.moving_volume_uid + '_label_' + reg_annotation.get_annotation_type_name() +
                                              '_registered_to_' + self.fixed_volume_uid + '.nii.gz')
                        os.makedirs(os.path.dirname(new_fp), exist_ok=True)
                        shutil.copyfile(fp, new_fp)
                        reg_annotation.include_registered_volume(filepath=new_fp,
                                                             registration_uid=self.registration_instance.unique_id,
                                                             destination_space_uid=self.fixed_volume_uid)

        except Exception as e:
            raise ValueError("Applying the registration on the annotation volume failed with: {}.".format(e))

    def __apply_registration_atlas_space(self):
        """
        @TODO. Have to include this info somehow inside the self._patient_parameters
        """
        try:
            fixed_filepath = self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).usable_input_filepath
            dump_folder = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                                       'Cortical-structures')
            os.makedirs(dump_folder, exist_ok=True)

            try:
                for s in ResourcesConfiguration.getInstance().neuro_features_cortical_structures:
                    self.__warp_cortical_structure(structure_name=s, fixed_filepath=fixed_filepath,
                                                   dump_folder=dump_folder)
                    # fp = self._registration_runner.apply_registration_inverse_transform(
                    #     moving=ResourcesConfiguration.getInstance().cortical_structures['MNI'][s]['Mask'],
                    #     fixed=fixed_filepath, interpolation='nearestNeighbor', label='Cortical-structures/' + s)
                    #
                    # new_fp = os.path.join(dump_folder, self.fixed_volume_uid + '_' + s + '_atlas.nii.gz')
                    # shutil.copyfile(fp, new_fp)
            except Exception as e:
                raise ValueError("Applying the registration on the a cortical structures atlas failed.") from e

            bcb_tracts_cutoff = 0.5
            dump_folder = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                                       'Subcortical-structures')
            os.makedirs(dump_folder, exist_ok=True)

            try:
                for s in ResourcesConfiguration.getInstance().neuro_features_subcortical_structures:
                    for i, elem in enumerate(tqdm(ResourcesConfiguration.getInstance().subcortical_structures['MNI'][s]['Singular'].keys())):
                        raw_filename = ResourcesConfiguration.getInstance().subcortical_structures['MNI'][s]['Singular'][elem]
                        raw_tract_ni = nib.load(raw_filename)
                        raw_tract = raw_tract_ni.get_fdata()[:]
                        raw_tract[raw_tract < bcb_tracts_cutoff] = 0
                        raw_tract[raw_tract >= bcb_tracts_cutoff] = 1
                        raw_tract = raw_tract.astype('uint8')
                        dump_filename = os.path.join(self._registration_runner.registration_folder, os.path.basename(raw_filename))
                        os.makedirs(os.path.dirname(dump_filename), exist_ok=True)
                        nib.save(nib.Nifti1Image(raw_tract, affine=raw_tract_ni.affine), dump_filename)

                        fp = self._registration_runner.apply_registration_inverse_transform(
                            moving=dump_filename,
                            fixed=fixed_filepath,
                            interpolation='nearestNeighbor',
                            label='Subcortical-structures/' + os.path.basename(raw_filename).split('.')[0].replace('_mni', ''))
                        new_fp = os.path.join(dump_folder, self.fixed_volume_uid + '_' + s + '_atlas_' + elem + '.nii.gz')
                        shutil.copyfile(fp, new_fp)

                    overall_mask_filename = ResourcesConfiguration.getInstance().subcortical_structures['MNI'][s]['Mask']
                    fp = self._registration_runner.apply_registration_inverse_transform(
                        moving=overall_mask_filename,
                        fixed=fixed_filepath,
                        interpolation='nearestNeighbor',
                        label='Subcortical-structures/' + s)

                    new_fp = os.path.join(dump_folder, self.fixed_volume_uid + '_' + s + '_atlas_overall_mask.nii.gz')
                    shutil.copyfile(fp, new_fp)
            except Exception as e:
                raise ValueError("Applying the registration on the a subcortical structures atlas failed.") from e

            dump_folder = os.path.join(self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                                       'Braingrid-structures')
            os.makedirs(dump_folder, exist_ok=True)

            try:
                for s in ResourcesConfiguration.getInstance().neuro_features_braingrid:
                    self.__warp_braingrid_element(structure_name=s, fixed_filepath=fixed_filepath, dump_folder=dump_folder)
                    # overall_mask_filename = ResourcesConfiguration.getInstance().braingrid_structures['MNI'][s]['Mask']
                    # fp = self._registration_runner.apply_registration_inverse_transform(
                    #     moving=overall_mask_filename,
                    #     fixed=fixed_filepath,
                    #     interpolation='nearestNeighbor',
                    #     label='Braingrid-structures/' + s)
                    #
                    # new_fp = os.path.join(dump_folder, self.fixed_volume_uid + '_' + s + '_atlas.nii.gz')
                    # shutil.copyfile(fp, new_fp)
            except Exception as e:
                raise ValueError("Applying the registration on the a BrainGrid structures atlas failed.") from e
        except Exception as e:
            raise ValueError(f"Applying the registration on the atlas volume failed with: {e}.") from e

    def __apply_registration_atlas_space_parallel(self):
        try:
            fixed_filepath = self._patient_parameters.get_radiological_volume(
                volume_uid=self.moving_volume_uid).usable_input_filepath
            inner_workers = max(1, os.cpu_count() // max(1, ResourcesConfiguration.getInstance().num_workers))

            # --- Cortical structures ---
            dump_folder = os.path.join(
                self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                'Cortical-structures')
            os.makedirs(dump_folder, exist_ok=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as pool:
                cortical_futures = {
                    pool.submit(self.__warp_cortical_structure, s, fixed_filepath, dump_folder): s
                    for s in ResourcesConfiguration.getInstance().neuro_features_cortical_structures
                }
                for future in concurrent.futures.as_completed(cortical_futures):
                    future.result()  # re-raises with structure name available via cortical_futures[future]

            # --- Subcortical structures ---
            bcb_tracts_cutoff = 0.5
            dump_folder = os.path.join(
                self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                'Subcortical-structures')
            os.makedirs(dump_folder, exist_ok=True)
            cfg = ResourcesConfiguration.getInstance()

            with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as pool:
                subcortical_futures = {}
                for s in cfg.neuro_features_subcortical_structures:
                    for elem, raw_fn in cfg.subcortical_structures['MNI'][s]['Singular'].items():
                        f = pool.submit(self.__warp_subcortical_element, s, elem, raw_fn,
                                        fixed_filepath, dump_folder, bcb_tracts_cutoff)
                        subcortical_futures[f] = (s, elem)
                    # overall mask is independent of the per-element loop
                    f = pool.submit(self._registration_runner.apply_registration_inverse_transform,
                                    cfg.subcortical_structures['MNI'][s]['Mask'],
                                    fixed_filepath, 'nearestNeighbor', 'Subcortical-structures/' + s)
                    subcortical_futures[f] = (s, 'overall_mask')

                for future in concurrent.futures.as_completed(subcortical_futures):
                    s, elem = subcortical_futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        raise ValueError(
                            f"Warping subcortical structure {s}/{elem} failed: {e}") from e

            # --- BrainGrid structures ---
            dump_folder = os.path.join(
                self._patient_parameters.get_radiological_volume(volume_uid=self.moving_volume_uid).output_folder,
                'Braingrid-structures')
            os.makedirs(dump_folder, exist_ok=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as pool:
                braingrid_futures = {
                    pool.submit(self.__warp_braingrid_element, s, fixed_filepath, dump_folder): s
                    for s in ResourcesConfiguration.getInstance().neuro_features_braingrid
                }
                for future in concurrent.futures.as_completed(braingrid_futures):
                    future.result()  # re-raises with structure name available via braingrid_futures[future]
        except Exception as e:
            raise ValueError(f"Applying the registration on the atlas volume failed with: {e}.") from e

    def __warp_cortical_structure(self, structure_name: str, fixed_filepath: str,
                                  dump_folder: str) -> None:
        fp = self._registration_runner.apply_registration_inverse_transform(
            moving=ResourcesConfiguration.getInstance().cortical_structures['MNI'][structure_name]['Mask'],
            fixed=fixed_filepath,
            interpolation='nearestNeighbor',
            label='Cortical-structures/' + structure_name)
        shutil.copyfile(fp,
                        os.path.join(dump_folder, self.fixed_volume_uid + '_' + structure_name + '_atlas.nii.gz'))

    def __warp_subcortical_element(self, structure_name: str, elem: str, raw_filename: str,
                                   fixed_filepath: str, dump_folder: str,
                                   bcb_tracts_cutoff: float) -> None:
        raw_tract_ni = nib.load(raw_filename)
        raw_tract = raw_tract_ni.get_fdata()[:]
        raw_tract = np.where(raw_tract >= bcb_tracts_cutoff, 1, 0).astype('uint8')
        dump_filename = os.path.join(self._registration_runner.registration_folder,
                                     os.path.basename(raw_filename))
        os.makedirs(os.path.dirname(dump_filename), exist_ok=True)
        nib.save(nib.Nifti1Image(raw_tract, affine=raw_tract_ni.affine), dump_filename)

        fp = self._registration_runner.apply_registration_inverse_transform(
            moving=dump_filename,
            fixed=fixed_filepath,
            interpolation='nearestNeighbor',
            label='Subcortical-structures/' + os.path.basename(raw_filename).split('.')[0].replace('_mni', ''))
        shutil.copyfile(fp, os.path.join(dump_folder,
                                         self.fixed_volume_uid + '_' + structure_name + '_atlas_' + elem + '.nii.gz'))

    def __warp_braingrid_element(self, structure_name: str, fixed_filepath: str,
                                  dump_folder: str) -> None:
        overall_mask_filename = ResourcesConfiguration.getInstance().braingrid_structures['MNI'][structure_name]['Mask']
        fp = self._registration_runner.apply_registration_inverse_transform(
            moving=overall_mask_filename,
            fixed=fixed_filepath,
            interpolation='nearestNeighbor',
            label='Braingrid-structures/' + structure_name)

        new_fp = os.path.join(dump_folder, self.fixed_volume_uid + '_' + structure_name + '_atlas.nii.gz')
        shutil.copyfile(fp, new_fp)