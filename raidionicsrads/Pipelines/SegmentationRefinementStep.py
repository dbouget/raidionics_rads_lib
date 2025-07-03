import os
import shutil
import numpy as np
import nibabel as nib
import logging
import configparser
import traceback
from ..Utils.utilities import get_type_from_string, get_type_from_enum_name
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.volume_utilities import prediction_binary_dilation
from ..Utils.io import load_nifti_volume
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.DataStructures.PatientStructure import PatientParameters
from ..Utils.DataStructures.AnnotationStructure import Annotation, AnnotationClassType, BrainTumorType
from ..Processing.brain_processing import perform_brain_overlap_refinement, perform_segmentation_global_consistency_refinement


class SegmentationRefinementStep(AbstractPipelineStep):
    """
    Processing the results from the raidionics_seg_lib backend to improve or clean the predictions
    """
    _input_volume_uid = None  # Internal unique id to the first/main radiological volume.
    _input_annotation_uid = None
    _patient_parameters = None  # Overall patient parameters, updated on-the-fly
    _working_folder = None  # Temporary directory on disk to store inputs/outputs for the segmentation
    _refinement_operation = None  # Type of refinement to operate
    _refinement_args = None  # Generic arguments needed for the specified refinement operation

    def __init__(self, step_json: dict):
        super(SegmentationRefinementStep, self).__init__(step_json=step_json)
        self.__reset()
        self._refinement_operation = self._step_json["operation"]
        self._refinement_args = self._step_json["args"]

    def __reset(self):
        self._input_volume_uid = None
        self._input_annotation_uid = None
        self._patient_parameters = None
        self._working_folder = None
        self._refinement_operation = None
        self._refinement_args = None

    @property
    def refinement_operation(self) -> str:
        return self._refinement_operation

    @refinement_operation.setter
    def refinement_operation(self, operation: str) -> None:
        self._refinement_operation = operation

    def setup(self, patient_parameters: PatientParameters) -> None:
        """
        Sanity check that all requirements are met for running the segmentation refinement step.
        @TODO. Should it be enforced with only a single input (i.e., brain_overlap refinement only for tumor-CE on T1-CE
        , and not on FLAIR changes AND tumor-CE on two different MR scans as input?

        Parameters
        ----------
        patient_parameters: PatientParameters
            Placeholder for the current patient data, which will be updated with the results of this step.
        """
        self._patient_parameters = patient_parameters

        self._working_folder = os.path.join(ResourcesConfiguration.getInstance().output_folder, "seg_refinement_tmp")
        os.makedirs(self._working_folder, exist_ok=True)
        os.makedirs(os.path.join(self._working_folder, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(self._working_folder, 'outputs'), exist_ok=True)

        try:
            for k in list(self._step_json["inputs"].keys()):
                input_json = self._step_json["inputs"][k]
                # Use-case where the radiological volume should be used in its original reference space
                if input_json["space"]["timestamp"] == input_json["timestamp"] and \
                        input_json["space"]["sequence"] == input_json["sequence"]:
                    volume_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                                                      sequence=input_json["sequence"])
                    if volume_uid == "-1":
                        raise ValueError("No radiological volume for {}.".format(input_json))

                    # Assuming the first input is actually the final target. Might need to add another parameter
                    # for specifying the volume the annotation is linked to, if multiple inputs.
                    if not self._input_volume_uid:
                        self._input_volume_uid = volume_uid
                    if self.refinement_operation != "brain_overlap" and ResourcesConfiguration.getInstance().predictions_use_stripped_data:
                        self.skip = True
                        return
                    if self.refinement_operation != "global_context":
                        # Retrieving the annotation to refine
                        if input_json["labels"]:
                            annotation_type = get_type_from_enum_name(AnnotationClassType, input_json["labels"])
                            anno_uids = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid,
                                                                                                                    annotation_class=annotation_type)
                            if len(anno_uids) == 0:
                                raise ValueError("No annotation for {}.".format(input_json))
                            elif len(anno_uids) > 1:
                                #@TODO. Assuming the last one is to refine? In case a structure is segmented multiple times
                                # before some kind of refinement?
                                raise ValueError("Too many annotations for {}.".format(input_json))
                            anno_uid = anno_uids[0]
                            self._input_annotation_uid = anno_uid
                        else:
                            raise ValueError("No annotation to refine was provided in {}.".format(input_json))
                # Use-case where the radiological volume should be used in another reference space
                else:
                    logging.warning(f"Use-case not handled yet!")
            if (self._input_volume_uid is None or
                    (self._input_annotation_uid is None and self.refinement_operation != "global_context")):
                raise ValueError(f"The inputs to refine could not be properly retrieved.")
        except Exception as e:
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("[SegmentationRefinementStep] setup failed with: {}.".format(e))

    def execute(self) -> PatientParameters:
        """
        Executes the current step.

        Returns
        -------
        PatientParameters
            Updated placeholder with the results of the current step.
        """
        if self.skip:
            if self.refinement_operation != "brain_overlap" and ResourcesConfiguration.getInstance().predictions_use_stripped_data:
                logging.info("Skipping brain overlap segmentation refinement, the input scans are skull-stripped")
            return self._patient_parameters

        try:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                self.__perform_neuro_postprocessing()
            else:
                logging.warning("[SegmentationRefinementStep] No execution implemented yet for the task {}".format(
                    ResourcesConfiguration.getInstance().diagnosis_task))
                pass
        except Exception as e:
            raise ValueError("[SegmentationRefinementStep] Step execution failed with: {}.".format(e))
        return self._patient_parameters

    def cleanup(self):
        if self._working_folder is not None and os.path.exists(self._working_folder):
            shutil.rmtree(self._working_folder)

    def __perform_neuro_postprocessing(self) -> None:
        """
        @TODO. If the use_registered_data flag is True, the annotations used in global_context refinement should hence
        also include the annotations from other sequences not featured in T1, for inclusion, not looking in the
        registered_volumes attribute.
        """
        try:
            if self.refinement_operation == "dilation":
                predictions_filepath = self._patient_parameters.get_annotation(annotation_uid=self._input_annotation_uid).usable_input_filepath
                prediction_binary_dilation(predictions_filepath, arg=int(self._refinement_args))
            elif self.refinement_operation == "brain_overlap":
                predictions_filepath = self._patient_parameters.get_annotation(annotation_uid=self._input_annotation_uid).usable_input_filepath
                brain_annotation_uids = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=self._input_volume_uid, annotation_class=AnnotationClassType.Brain)
                if len(brain_annotation_uids) == 0 or len(brain_annotation_uids) > 1:
                    raise ValueError(f"The brain annotation could not be retrieved for performing segmentation refinement.")
                brain_annotation_uid = brain_annotation_uids[0]
                brain_mask_filepath = self._patient_parameters.get_annotation(
                    annotation_uid=brain_annotation_uid).usable_input_filepath
                perform_brain_overlap_refinement(predictions_filepath=predictions_filepath, brain_mask_filepath=brain_mask_filepath)
            elif self.refinement_operation == "global_context":
                annotation_files = {}
                for a in self._patient_parameters.get_all_annotations_radiological_volume(volume_uid=self._input_volume_uid):
                    annotation_files[a.get_annotation_type_str()] = a.usable_input_filepath
                for v in self._patient_parameters.get_all_radiological_volumes_for_timestamp(timestamp=self._step_json["inputs"]["0"]["timestamp"]):
                    if v.unique_id != self._input_volume_uid:
                        linked_annos = self._patient_parameters.get_all_annotations_radiological_volume(v.unique_id)
                        for anno in linked_annos:
                            if not ResourcesConfiguration.getInstance().predictions_use_registered_data:
                                if self._input_volume_uid in list(anno.registered_volumes.keys()):
                                    if anno.get_annotation_type_str() not in list(annotation_files.keys()):
                                        annotation_files[anno.get_annotation_type_str()] = anno.registered_volumes[self._input_volume_uid]["filepath"]
                            else:
                                annotation_files[anno.get_annotation_type_str()] = anno.usable_input_filepath

                tumor_general_type = "contrast-enhancing"
                if self.step_json["inputs"]["0"]["labels"] == "FLAIRChanges":
                    tumor_general_type = "non contrast-enhancing"
                refined_annos = perform_segmentation_global_consistency_refinement(annotation_files=annotation_files,
                                                                   timestamp=self._step_json["inputs"]["0"]["timestamp"],
                                                                                   tumor_general_type=tumor_general_type)
                for ranno in list(refined_annos.keys()):
                    if not self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=self._input_volume_uid,
                                                                                                       annotation_class=ranno):
                        # A new annotation has been created and should be included.
                        non_available_uid = True
                        anno_uid = None
                        while non_available_uid:
                            anno_uid = 'A' + str(np.random.randint(0, 10000))
                            if anno_uid not in self._patient_parameters.get_all_annotations_uids():
                                non_available_uid = False
                        annotation = Annotation(uid=anno_uid, input_filename=refined_annos[ranno],
                                                output_folder=self._patient_parameters.get_radiological_volume(
                                                    volume_uid=self._input_volume_uid).output_folder,
                                                radiological_volume_uid=self._input_volume_uid,
                                                annotation_class=ranno)
                        self._patient_parameters.include_annotation(anno_uid, annotation)
            else:
                raise ValueError("The selected refinement operation is not available, with value {}".format(self.refinement_operation))
        except Exception as e:
            if self._working_folder is not None and os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("Segmentation refinement execution could not proceed with: {}.".format(e))

        if self._working_folder is not None and os.path.exists(self._working_folder):
            shutil.rmtree(self._working_folder)


    def __perform_neuro_postprocessing_old(self) -> None:
        """

        """
        try:
            segmentation_folder = os.path.join(self._working_folder, 'inputs')
            segmentation_files = []
            for _, _, files in os.walk(segmentation_folder):
                for f in files:
                    segmentation_files.append(f)
                break

            for fn in segmentation_files:
                seg_fn = os.path.join(segmentation_folder, fn)
                seg_ni = nib.load(seg_fn)
                seg = seg_ni.get_fdata()[:]

                res = None
                if self.refinement_operation == "dilation":
                    res = prediction_binary_dilation(seg.astype('uint8'),
                                                     voxel_volume=np.prod(seg_ni.header.get_zooms()) * 1e-3,
                                                     arg=int(self._refinement_args))
                elif self.refinement_operation == "brain_overlap":
                    pass
                else:
                    raise ValueError("The selected refinement operation is not available, with value {}".format(self.refinement_operation))

                res_ni = nib.Nifti1Image(res, affine=seg_ni.affine)
                output_fn = os.path.join(self._working_folder, 'outputs', fn)
                nib.save(res_ni, output_fn)

        except Exception as e:
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("Segmentation refinement execution could not proceed with: {}.".format(e))

        try:
            # Collecting the results and associating them with the parent radiological volume.
            generated_segmentations = []
            for _, _, files in os.walk(os.path.join(self._working_folder, 'outputs')):
                for f in files:
                    if 'nii.gz' in f:
                        generated_segmentations.append(f)
                break

            for k in list(self._step_json["inputs"].keys()):
                output_index = ["input" + str(k) in x for x in generated_segmentations].index(True)
                seg_filename = os.path.join(self._working_folder, "outputs", generated_segmentations[output_index])
                annotation_struct = self._patient_parameters.get_input_from_json(self._step_json["inputs"][k])
                shutil.copyfile(src=seg_filename, dst=annotation_struct.raw_input_filepath)

        except Exception as e:
            if os.path.exists(self._working_folder):
                shutil.rmtree(self._working_folder)
            raise ValueError("Segmentation refinement results parsing failed with: {}.".format(e))

        if os.path.exists(self._working_folder):
            shutil.rmtree(self._working_folder)
