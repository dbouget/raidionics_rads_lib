import json
import os.path
import traceback
import logging
import numpy as np
import nibabel as nib
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.ReportingStructures.NeuroReportingStructure import NeuroReportingStructure
from ..Processing.neuro_report_computing import *
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType, BrainTumorType
from ..Utils.utilities import get_type_from_enum_name


class FeaturesComputationStep(AbstractPipelineStep):
    """
    @TODO. Current approach is fine for features computed over a single volume, what about cross-volume features?
    """
    _patient_parameters = None
    _radiological_volume_uid = None
    _report_spaces = None
    _report = None
    _targets = None

    def __init__(self, step_json: dict) -> None:
        self._reset()
        super(FeaturesComputationStep, self).__init__(step_json=step_json)
        self._report_spaces = [x.strip() for x in self._step_json["spaces"]]
        self._targets = [x.strip() for x in self._step_json["targets"]]

    @property
    def report_spaces(self) -> List[str]:
        return self._report_spaces

    @report_spaces.setter
    def report_spaces(self, text: List[str]) -> None:
        self.report_spaces = text

    @property
    def targets(self) -> List[str]:
        return self._targets

    @targets.setter
    def targets(self, targets: List[str]) -> None:
        self._targets = targets

    def _reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        super()._reset()
        self._patient_parameters = None
        self._radiological_volume_uid = None
        self._report_spaces = None
        self._report = None
        self._targets = None

    def setup(self, patient_parameters):
        """
        Verify that the requirements are met for executing the step - should go deeper (copy the initial checks from the execute step)
        """
        if True in [x not in ["Patient", "MNI"] for x in self.report_spaces]:
            raise ValueError(f"Features computation only implemented for Patient and MNI spaces, "
                             f"unknown key provided in \"{self.report_spaces}\".")
        self._patient_parameters = patient_parameters        

    def execute(self):
        """

        """
        try:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                self.__run_neuro_reporting()
            else:
                logging.warning("[FeaturesComputationStep] No execution implemented yet for the task {}".format(
                    ResourcesConfiguration.getInstance().diagnosis_task))
                pass
        except Exception as e:
            raise ValueError("[FeaturesComputationStep] Step execution failed with: {}.".format(e))
        return self._patient_parameters

    def cleanup(self):
        pass

    def __run_neuro_reporting(self):
        """
        # @TODO. What to do to ensure the first radiological volume is correct if multiple.
        # The FLAIR image might not be the base image for non-ce either... need to find a smarter way to handle this...

        """
        non_available_uid = True
        report_uid = None
        while non_available_uid:
            report_uid = 'RADS' + str(np.random.randint(0, 10000))
            if report_uid not in self._patient_parameters.get_all_reportings_uids():
                non_available_uid = False
        report = NeuroReportingStructure(id=report_uid,
                                        output_folder=ResourcesConfiguration.getInstance().output_folder,
                                        timestamp=self.step_json["timestamp"])
        base_radiological_volume = self._patient_parameters.get_radiological_volume_for_timestamp_and_sequence(timestamp=self.step_json["timestamp"], sequence=str(MRISequenceType.T1c)) if self.step_json["tumor_type"] == "contrast-enhancing" else self._patient_parameters.get_radiological_volume_for_timestamp_and_sequence(timestamp=self.step_json["timestamp"], sequence=str(MRISequenceType.FLAIR))
        brain_annotation = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=base_radiological_volume[0].unique_id, annotation_class=AnnotationClassType.Brain, return_objects=True)
        for sp in self.report_spaces:
            brain_nib = None
            if len(brain_annotation) != 0:
                if sp == "Patient":
                    brain_filepath = brain_annotation[0].usable_input_filepath
                else:
                    if sp not in brain_annotation[0].registered_volumes.keys():
                        # Can only be MNI space for now
                        brain_filepath = ResourcesConfiguration.getInstance().mni_atlas_brain_mask_filepath
                        # raise ValueError(f"The {self.report_space} key was not found for {brain_annotation[0].radiological_volume_uid}")
                    else:
                        brain_filepath = brain_annotation[0].registered_volumes[sp]["filepath"]
                brain_nib = nib.load(brain_filepath)

            for t in self.targets:
                # Filling in the tumor type (@TODO. not ideal here, but the report is not yet created at the time the
                # classification is performed... Should be made cleaner in the future.
                if t == "Tumor":
                    tumortype_classif_results_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder,
                                                                "BrainTumorType_classification_results_raw.csv")
                    if os.path.exists(tumortype_classif_results_fn):
                        tumortype_classif_results_df = pd.read_csv(tumortype_classif_results_fn)
                        tumor_type = tumortype_classif_results_df.values[tumortype_classif_results_df[tumortype_classif_results_df.columns[1]].idxmax(), 0]
                        report.tumor_type = get_type_from_enum_name(BrainTumorType, tumor_type)
                elif t == "FLAIRChanges" and "Tumor" not in self.targets:
                    report.tumor_type = BrainTumorType.LGG
                structure_nib = None
                if t in [x.name for x in list(AnnotationClassType)]:
                    annotation_filepath = None
                    struct_annotations = self._patient_parameters.get_all_annotations_for_timestamp_and_structure(
                        timestamp=self.step_json["timestamp"], structure=t)
                    if len(struct_annotations) == 0:
                        logging.warning(f"Skipping features computation for {t} as no segmentation file exists")
                        continue
                    if sp == "Patient":
                        annotation_filepath = struct_annotations[0].usable_input_filepath
                    else:
                        annotation_filepath = struct_annotations[0].registered_volumes[sp]["filepath"]

                    if annotation_filepath is None:
                        logging.error("No structure filepath found on disk.")
                    else:
                        structure_nib = nib.load(annotation_filepath)
                else:
                    # @TODO. Have to manually assemble the combined structure
                    pass

                if structure_nib is None:
                    logging.error(f"No segmentation file found nor assembled for structure: {t}")
                    continue
                else:
                    res, clustered_nib = compute_structure_statistics(input_mask=structure_nib, brain_mask=brain_nib,
                                                    params={"report_space":sp, "structure": t})
                    report.include_statistics(structure=t, statistics=res, space=sp)
                    dest_ccl_fn = os.path.join(ResourcesConfiguration.getInstance().output_folder, "reporting", f'T{self.step_json["timestamp"]}', f'input-{base_radiological_volume[0].get_sequence_type_str()}_space-{sp}_class-{struct_annotations[0].get_annotation_type_name()}_CC-labelled.nii.gz')
                    os.makedirs(os.path.dirname(dest_ccl_fn), exist_ok=True)
                    nib.save(clustered_nib, dest_ccl_fn)
        # Include the acquisition infos here for now
        acquisition_infos = compute_acquisition_infos(self._patient_parameters.get_all_radiological_volumes_for_timestamp(timestamp=self.step_json["timestamp"]))
        report.acquisition_infos = acquisition_infos
        self._patient_parameters.include_reporting(report_uid, report)
        report.to_disk()
