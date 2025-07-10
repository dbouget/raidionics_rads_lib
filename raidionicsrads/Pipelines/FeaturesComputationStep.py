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
    _report_space = None
    _report = None
    _targets = None

    def __init__(self, step_json: dict) -> None:
        super(FeaturesComputationStep, self).__init__(step_json=step_json)
        self.__reset()
        self._report_space = self._step_json["space"]
        self._targets = self._step_json["target"]

    @property
    def report_space(self) -> str:
        return self._report_space

    @report_space.setter
    def report_space(self, text: str) -> None:
        self.report_space = text

    @property
    def targets(self) -> str:
        return self._targets

    @targets.setter
    def targets(self, targets: str) -> None:
        self._targets = targets

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._patient_parameters = None
        self._radiological_volume_uid = None
        self._report_space = None
        self._report = None
        self._targets = None

    def setup(self, patient_parameters):
        """
        Verify that the requirements are met for executing the step
        """
        if self.report_space != "MNI":
            raise ValueError(f"Features computation only implemented for MNI space, "
                             f"unknown key \"{self.report_space}\" provided.")
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
        brain_nib = None
        if len(brain_annotation) != 0:
            if self.report_space == "Patient":
                brain_filepath = brain_annotation[0].usable_input_filepath
            else:
                if self.report_space not in brain_annotation[0].registered_volumes.keys() :
                    raise ValueError(f"The {self.report_space} key was not found for {brain_annotation[0].radiological_volume_uid}")
                brain_filepath = brain_annotation[0].registered_volumes[self.report_space]["filepath"]
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
                if self.report_space == "Patient":
                    annotation_filepath = struct_annotations[0].usable_input_filepath
                else:
                    annotation_filepath = struct_annotations[0].registered_volumes[self.report_space]["filepath"]

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
                res = compute_structure_statistics(input_mask=structure_nib, brain_mask=brain_nib)
                report.include_statistics(structure=t, statistics=res, space=self.report_space)
                if self.report_space != 'Patient':
                    # Including the tumor volume in original patient space, quick fix for now as the only
                    # supported report_space is MNI
                    pat_space_result = NeuroStructureStatistics()
                    patient_anno = nib.load(annotation_filepath).get_fdata()[:]
                    volume = np.count_nonzero(patient_anno) * np.prod(
                        nib.load(annotation_filepath).header.get_zooms()[0:3]) * 1e-3
                    pat_space_result.volume = NeuroVolumeStatistics(volume=volume, brain_percentage=-1.)
                    report.include_statistics(structure=t, statistics=pat_space_result, space="Patient")
        # Include the acquisition infos here (for now?)
        acquisition_infos = compute_acquisition_infos(self._patient_parameters.get_all_radiological_volumes_for_timestamp(timestamp=self.step_json["timestamp"]))
        report.acquisition_infos = acquisition_infos
        self._patient_parameters.include_reporting(report_uid, report)
        report.to_disk()

    def __run_neuro_reporting_old(self):
        """
        @TODO. The self.report_space will not handle properly the Atlas files, should have another flag inside the
        compute_neuro_report method to open the original MNI space files or back-registered files in patient space!
        """
        try:
            uid = self._patient_parameters.get_radiological_volume_uid(timestamp=self.step_json["input"]["timestamp"],
                                                                       sequence=self.step_json["input"]["sequence"])
            if not uid:
                return None
            self._radiological_volume_uid = uid

            # @TODO. Send the proper annotation file, registered to MNI, but for now assuming it's the Tumor annotation
            # @TODO2. What if we want to report for the TumorCE, Tumor Core, Whole tumor, how to make it adjustable?
            anno_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=self._radiological_volume_uid,
                                                                                                   annotation_class=get_type_from_enum_name(AnnotationClassType, self.target))
            if len(anno_uid) == 0:
                return None
            anno_uid = anno_uid[0]
            report_filename_input = self._patient_parameters.get_annotation(annotation_uid=anno_uid).usable_input_filepath
            if self.report_space != 'Patient':
                reg_data = self._patient_parameters.get_annotation(annotation_uid=anno_uid).get_registered_volume_info(destination_space_uid=self.report_space)
                report_filename_input = reg_data["filepath"]

            non_available_uid = True
            report_uid = None
            while non_available_uid:
                report_uid = 'RADS' + str(np.random.randint(0, 10000))
                if report_uid not in self._patient_parameters.get_all_reportings_uids():
                    non_available_uid = False
            report = NeuroReportingStructure(id=report_uid, parent_uid=self._radiological_volume_uid,
                                             output_folder=ResourcesConfiguration.getInstance().output_folder)
            report._tumor_type = self._patient_parameters.get_annotation(annotation_uid=anno_uid).get_annotation_subtype_str()
            updated_report = compute_neuro_report(report_filename_input, report)
            if self.report_space != 'Patient':
                # Including the tumor volume in original patient space, quick fix for now
                patient_anno_fn = self._patient_parameters.get_annotation(annotation_uid=anno_uid).usable_input_filepath
                patient_anno = nib.load(patient_anno_fn).get_fdata()[:]
                volume = np.count_nonzero(patient_anno) * np.prod(nib.load(patient_anno_fn).header.get_zooms()[0:3]) * 1e-3
                updated_report._statistics['Main']['Overall'].original_space_volume = np.round(volume, 2)
            self._patient_parameters.include_reporting(report_uid, updated_report)
            updated_report.to_txt()
            updated_report.to_csv()
            updated_report.to_json()
            updated_report.dump_descriptions()
        except Exception as e:
            raise ValueError("[FeaturesComputationStep] Neuro reporting failed with: {}.".format(e))
