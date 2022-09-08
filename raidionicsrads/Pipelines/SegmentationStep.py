import os
import shutil
import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.AnnotationStructure import Annotation, AnnotationClassType


class SegmentationStep(AbstractPipelineStep):
    _input_volume_uid = None
    _segmentation_target = None
    _model_name = None
    _patient_parameters = None  # Might be useless to store them here, unless we do an assess step, if possible
    _input_volume_filepath = None

    def __init__(self, step_json: dict):
        super(SegmentationStep, self).__init__(step_json=step_json)
        self.__reset()
        self._model_name = self._step_json["model"]
        self._segmentation_target = self._step_json["target"]

    def __reset(self):
        self._input_volume_uid = None
        self._segmentation_target = None
        self._model_name = None
        self._patient_parameters = None
        self._input_volume_filepath = None

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters
        volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=self._step_json["input"]["timestamp"],
                                                                          sequence=self._step_json["input"]["sequence"])
        if volume_uid != "-1":
            self._input_volume_uid = volume_uid
            self._input_volume_filepath = self._patient_parameters._radiological_volumes[self._input_volume_uid]._usable_input_filepath

    def execute(self):
        if self._input_volume_uid:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                self.__perform_neuro_segmentation()
            else:
                self.__perform_mediastinum_segmentation()
        return self._patient_parameters

    def __perform_neuro_segmentation(self) -> None:
        """

        """
        # @TODO. Should check if there is a tumor annotation linked to the current volume_uid
        if not ResourcesConfiguration.getInstance().runtime_tumor_mask_filepath is None \
                and os.path.exists(ResourcesConfiguration.getInstance().runtime_tumor_mask_filepath):
            return ResourcesConfiguration.getInstance().runtime_tumor_mask_filepath
        else:
            seg_config_filename = ""
            try:
                tmp_dir = os.path.join(ResourcesConfiguration.getInstance().output_folder, "segmentation_tmp")
                os.makedirs(tmp_dir, exist_ok=True)

                seg_config = configparser.ConfigParser()
                seg_config.add_section('System')
                seg_config.set('System', 'gpu_id', ResourcesConfiguration.getInstance().gpu_id)
                seg_config.set('System', 'input_filename', self._input_volume_filepath)
                seg_config.set('System', 'output_folder', tmp_dir)
                seg_config.set('System', 'model_folder', os.path.join(ResourcesConfiguration.getInstance().model_folder, self._model_name))
                seg_config.add_section('Runtime')
                seg_config.set('Runtime', 'reconstruction_method', 'thresholding')
                seg_config.set('Runtime', 'reconstruction_order', 'resample_first')

                existing_brain_annotations = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=self._input_volume_uid,
                                                                                                                         annotation_class=AnnotationClassType.Brain)
                if len(existing_brain_annotations) != 0:
                    seg_config.add_section('Neuro')
                    seg_config.set('Neuro', 'brain_segmentation_filename',
                                   self._patient_parameters._annotation_volumes[existing_brain_annotations[0]]._usable_input_filepath)
                seg_config_filename = os.path.join(tmp_dir, 'seg_config.ini')
                with open(seg_config_filename, 'w') as outfile:
                    seg_config.write(outfile)

                log_level = logging.getLogger().level
                log_str = 'warning'
                if log_level == 10:
                    log_str = 'debug'
                elif log_level == 20:
                    log_str = 'info'
                elif log_level == 40:
                    log_str = 'error'

                from raidionicsseg.fit import run_model
                run_model(seg_config_filename)
            except Exception as e:
                logging.error("Automatic tumor segmentation failed with: {}.".format(traceback.format_exc()))
                if os.path.exists(seg_config_filename):
                    os.remove(seg_config_filename)
                raise ValueError("Impossible to perform automatic tumor segmentation.")

            # Collecting the results and associating them with the parent radiological volume.
            generated_segmentations = []
            for _, _, files in os.walk(tmp_dir):
                for f in files:
                    if 'nii.gz' in f:
                        generated_segmentations.append(f)
                break

            segmentation_targets = self._step_json["target"].split(',')
            for s in generated_segmentations:
                label_name = s.split('_')[1].split('.')[0]
                if label_name in segmentation_targets:
                    seg_filename = os.path.join(tmp_dir, s)
                    final_seg_filename = os.path.join(self._patient_parameters._radiological_volumes[self._input_volume_uid]._output_folder,
                                                      os.path.basename(self._patient_parameters._radiological_volumes[self._input_volume_uid]._raw_input_filepath).split('.')[0] + '_annotation-' + label_name + '.nii.gz')
                    shutil.move(seg_filename, final_seg_filename)
                    non_available_uid = True
                    anno_uid = None
                    while non_available_uid:
                        anno_uid = 'A' + str(np.random.randint(0, 10000))
                        if anno_uid not in self._patient_parameters.get_all_annotations_uids():
                            non_available_uid = False
                    annotation = Annotation(uid=anno_uid, input_filename=final_seg_filename,
                                            output_folder=self._patient_parameters._radiological_volumes[self._input_volume_uid]._output_folder,
                                            radiological_volume_uid=self._input_volume_uid, annotation_class=label_name)
                    self._patient_parameters.include_annotation(anno_uid, annotation)

            shutil.rmtree(tmp_dir)

    def __perform_mediastinum_segmentation(self):
        pass
