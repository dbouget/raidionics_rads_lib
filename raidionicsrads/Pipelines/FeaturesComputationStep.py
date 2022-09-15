import json
import numpy as np
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.ReportingStructures.NeuroReportingStructure import NeuroReportingStructure
from ..Processing.neuro_report_computing import compute_neuro_report
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType


class FeaturesComputationStep(AbstractPipelineStep):
    """
    @TODO. Current approach is fine for features computed over a single volume, what about cross-volume features?
    """
    _patient_parameters = None
    _radiological_volume_uid = None
    _report = None

    def __init__(self, step_json: dict) -> None:
        super(FeaturesComputationStep, self).__init__(step_json=step_json)
        self.__reset()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._patient_parameters = None
        self._radiological_volume_uid = None
        self._report = None

    def setup(self, patient_parameters):
        self._patient_parameters = patient_parameters

    def execute(self):
        if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
            self.__run_neuro_reporting()
        else:
            pass
        return self._patient_parameters

    def __run_neuro_reporting(self):
        # @TODO. Should the report be computed in original patient space?
        # Or twice including also the reference space, for easier benchmarking/comparison?
        uid = self._patient_parameters.get_radiological_volume_uid(timestamp=self._step_json["input"]["timestamp"],
                                                                   sequence=self._step_json["input"]["sequence"])
        if not uid:
            return None
        self._radiological_volume_uid = uid

        # @TODO. Send the proper annotation file, registered to MNI, but for now assuming it's the Tumor annotation
        anno_uid = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=self._radiological_volume_uid,
                                                                                               annotation_class=AnnotationClassType.Tumor)
        if len(anno_uid) == 0:
            return None
        anno_uid = anno_uid[0]
        reg_data = self._patient_parameters.get_annotation(annotation_uid=anno_uid).get_registered_volume_info(destination_space_uid="MNI")

        non_available_uid = True
        report_uid = None
        while non_available_uid:
            report_uid = 'RADS' + str(np.random.randint(0, 10000))
            if report_uid not in self._patient_parameters.get_all_reportings_uids():
                non_available_uid = False
        report = NeuroReportingStructure(id=report_uid, parent_uid=self._radiological_volume_uid,
                                         output_folder=ResourcesConfiguration.getInstance().output_folder)
        report._tumor_type = "HGGlioma"  # @TODO. Have to find a way to store this info somewhere
        updated_report = compute_neuro_report(reg_data["filepath"], report)
        self._patient_parameters.include_reporting(report_uid, updated_report)
        updated_report.to_txt()
        updated_report.to_csv()
        updated_report.to_json()
        updated_report.dump_descriptions()
