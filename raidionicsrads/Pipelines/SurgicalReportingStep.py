import json
import traceback
import logging
import numpy as np
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.ReportingStructures.NeuroSurgicalReportingStructure import NeuroSurgicalReportingStructure
from ..Processing.neuro_report_computing import compute_surgical_report
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType


class SurgicalReportingStep(AbstractPipelineStep):
    """

    """
    _patient_parameters = None
    _report = None
    _tumor_type = None

    def __init__(self, step_json: dict) -> None:
        super(SurgicalReportingStep, self).__init__(step_json=step_json)
        self.__reset()
        step_keys = list(self._step_json.keys())
        self._tumor_type = self._step_json["tumor_type"] if "tumor_type" in step_keys else None

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._patient_parameters = None
        self._report = None
        self._tumor_type = None

    @property
    def tumor_type(self) -> str:
        return self._tumor_type

    @tumor_type.setter
    def tumor_type(self, value: str) -> None:
        self._tumor_type = value

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters

    def execute(self):
        try:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                self.__run_neuro_surgical_reporting()
            else:
                logging.warning("[SurgicalReportingStep] No execution implemented yet for the task {}".format(
                    ResourcesConfiguration.getInstance().diagnosis_task))
                pass
        except Exception as e:
            raise ValueError(f"[SurgicalReportingStep] Step execution failed with: {e}.")
        return self._patient_parameters

    def cleanup(self):
        pass

    def __run_neuro_surgical_reporting(self):
        """

        """
        try:
            non_available_uid = True
            report_uid = None
            while non_available_uid:
                report_uid = 'SurRep' + str(np.random.randint(0, 10000))
                if report_uid not in self._patient_parameters.get_all_reportings_uids():
                    non_available_uid = False
            report = NeuroSurgicalReportingStructure(id=report_uid,
                                                     output_folder=ResourcesConfiguration.getInstance().output_folder)
            if self.tumor_type.lower() == "contrast-enhancing":
                preop_t1ce_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=0, sequence="T1-CE")
                postop_t1ce_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=1, sequence="T1-CE")
                preop_tumor_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                    volume_uid=preop_t1ce_uid, annotation_class=AnnotationClassType.Tumor)
                postop_tumor_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                    volume_uid=postop_t1ce_uid, annotation_class=AnnotationClassType.TumorCE)
                postop_flairchanges_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                    volume_uid=postop_t1ce_uid, annotation_class=AnnotationClassType.FLAIRChanges)
                postop_cavity_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                    volume_uid=postop_t1ce_uid, annotation_class=AnnotationClassType.Cavity)
                if len(preop_tumor_uid) > 0 and len(postop_tumor_uid) > 0:
                    preop_fn = self._patient_parameters.get_annotation(annotation_uid=preop_tumor_uid[0]).usable_input_filepath
                    postop_fn = self._patient_parameters.get_annotation(annotation_uid=postop_tumor_uid[0]).usable_input_filepath
                    flairchanges_fn = self._patient_parameters.get_annotation(annotation_uid=postop_flairchanges_uid[0]).usable_input_filepath if len(postop_flairchanges_uid) > 0 else None
                    cavity_postop_fn = self._patient_parameters.get_annotation(
                        annotation_uid=postop_cavity_uid[0]).usable_input_filepath if len(
                        postop_cavity_uid) > 0 else None
                    compute_surgical_report(tumor_preop_fn=preop_fn, tumor_postop_fn=postop_fn,
                                            flairchanges_postop_fn=flairchanges_fn, cavity_postop_fn=cavity_postop_fn,
                                            report=report, tumor_type=self.tumor_type)
                else:
                    raise ValueError("Missing either the preoperative or postoperative tumor segmentation.")
            elif self.tumor_type.lower() == "non contrast-enhancing":
                preop_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=0, sequence="FLAIR")
                postop_uid = self._patient_parameters.get_radiological_volume_uid(timestamp=1, sequence="FLAIR")
                preop_tumor_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                    volume_uid=preop_uid, annotation_class=AnnotationClassType.FLAIRChanges)
                postop_tumor_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                    volume_uid=postop_uid, annotation_class=AnnotationClassType.FLAIRChanges)
                postop_cavity_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(
                    volume_uid=postop_uid, annotation_class=AnnotationClassType.Cavity)
                if len(preop_tumor_uid) > 0 and len(postop_tumor_uid) > 0:
                    preop_fn = self._patient_parameters.get_annotation(annotation_uid=preop_tumor_uid[0]).usable_input_filepath
                    postop_fn = self._patient_parameters.get_annotation(annotation_uid=postop_tumor_uid[0]).usable_input_filepath
                    cavity_postop_fn = self._patient_parameters.get_annotation(
                        annotation_uid=postop_cavity_uid[0]).usable_input_filepath if len(
                        postop_cavity_uid) > 0 else None
                    compute_surgical_report(tumor_preop_fn=preop_fn, tumor_postop_fn=postop_fn,
                                            cavity_postop_fn=cavity_postop_fn,
                                            report=report, tumor_type=self.tumor_type)
                else:
                    raise ValueError("Missing either the preoperative or postoperative FLAIR changes segmentation.")
            self._patient_parameters.include_reporting(report_uid, report)
            report.to_disk()
        except Exception as e:
            raise ValueError(f"[SurgicalReportingStep] Neurosurgical reporting failed with: {e}.")
