import os
import logging
import numpy as np
import nibabel as nib
from .AbstractPipelineStep import AbstractPipelineStep
from ..Utils.configuration_parser import ResourcesConfiguration
from ..Utils.ReportingStructures.NeuroReportingStructure import NeuroReportingStructure
from ..Utils.ReportingStructures.NeuroTimepointReportingStructure import NeuroTimepointReportingStructure
from ..Processing.compute_composition import load_composition, compute_composition
from ..Utils.DataStructures.AnnotationStructure import AnnotationClassType
from ..Utils.utilities import get_type_from_enum_name


class TimepointReportingStep(AbstractPipelineStep):
    """
    Relations between the segmented structures of one timestamp (e.g., necrosis fraction of a tumour, thickness of the
    enhancing rim, overlaps between compartments meant to be disjoint), following a composition schema which tells
    which structures are the compartments of the lesion (see raidionicsrads/Compositions).
    Meant to run after the features computation of the same timestamp, whose per-structure measurements are
    referenced in the report. The masks are taken on the grid of the base volume of the composition (e.g., the T1-CE
    scan), either directly or from their registration into it, and only in patient space for now.
    Step json: {"task": "Timepoint reporting", "timestamp": 0, "composition": "enhancing_tumor" (or the schema as a
    dict), "space": "Patient" (optional), "description": "..."}. Without composition, the tumor_type key decides it
    (contrast-enhancing or not).
    @TODO. Relations needing extra structures (e.g., ventricle contact, midline shift) once such masks are available.
    """
    _patient_parameters = None
    _timestamp = None  # Timestamp to report on
    _composition = None  # Composition schema, loaded at setup
    _composition_key = None  # Composition as given in the step (name or dict)
    _report_space = None  # Space of the masks used, only Patient for now

    def __init__(self, step_json: dict) -> None:
        self._reset()
        super(TimepointReportingStep, self).__init__(step_json=step_json)
        step_keys = list(self._step_json.keys())
        if "timestamp" not in step_keys:
            raise ValueError("[TimepointReportingStep] A timestamp is required in the step.")
        self._timestamp = int(self._step_json["timestamp"])
        self._report_space = "Patient"
        if "space" in step_keys:
            self._report_space = self._step_json["space"]
        if "composition" in step_keys:
            self._composition_key = self._step_json["composition"]
        elif "tumor_type" in step_keys and self._step_json["tumor_type"] is not None:
            if self._step_json["tumor_type"].lower() == "contrast-enhancing":
                self._composition_key = "enhancing_tumor"
            else:
                self._composition_key = "non_enhancing_tumor"

    def _reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        super()._reset()
        self._patient_parameters = None
        self._timestamp = None
        self._composition = None
        self._composition_key = None
        self._report_space = None

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def composition(self) -> dict:
        return self._composition

    @property
    def report_space(self) -> str:
        return self._report_space

    def setup(self, patient_parameters):
        """
        Verify that the requirements are met for executing the step
        """
        if self.report_space != "Patient":
            raise ValueError(f"[TimepointReportingStep] Only implemented in Patient space, unknown key \"{self.report_space}\" provided.")
        if self._composition_key is None:
            raise ValueError("[TimepointReportingStep] A composition (or tumor_type) is required in the step.")
        self._composition = load_composition(composition=self._composition_key)
        self._patient_parameters = patient_parameters

    def execute(self):
        if self.skip:
            logging.info("[TimepointReportingStep] Step skipped.")
            return self._patient_parameters
        try:
            if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis':
                self.__run_neuro_timepoint_reporting()
            else:
                logging.warning("[TimepointReportingStep] No execution implemented yet for the task {}".format(
                    ResourcesConfiguration.getInstance().diagnosis_task))
        except Exception as e:
            raise ValueError(f"[TimepointReportingStep] Step execution failed with: {e}.")
        return self._patient_parameters

    def cleanup(self):
        pass

    def __run_neuro_timepoint_reporting(self):
        base_volumes = self._patient_parameters.get_radiological_volume_for_timestamp_and_sequence(timestamp=self.timestamp,
                                                                                                  sequence=self.composition["base_sequence"])
        if base_volumes is None or len(base_volumes) == 0:
            raise ValueError(f"No {self.composition['base_sequence']} volume at timestamp {self.timestamp}.")
        base_volume = base_volumes[0]
        base_nib = nib.load(base_volume.usable_input_filepath)
        spacing = base_nib.header.get_zooms()[0:3]

        features_report = self.__get_features_report()
        notes = []
        masks = {}
        mask_filepaths = {}
        statistics_filenames = {}
        structures = {}
        for role in self.composition["compartments"].keys():
            structures[role] = self.composition["compartments"][role]
        for role in self.composition["references"].keys():
            structures[role] = self.composition["references"][role]
        for role in structures.keys():
            mask_fp = self.__get_mask_filepath(structure=structures[role], base_volume_uid=base_volume.unique_id)
            if mask_fp is None:
                continue
            mask_nib = nib.load(mask_fp)
            if mask_nib.shape[0:3] != base_nib.shape[0:3] or not np.allclose(mask_nib.affine, base_nib.affine, atol=1e-3):
                notes.append(f"{role} ({structures[role]}) mask not on the grid of the {self.composition['base_sequence']} "
                             f"volume, left out")
                continue
            mask_array = np.asarray(mask_nib.dataobj)
            if len(mask_array.shape) == 4:
                mask_array = mask_array[..., 0]
            masks[role] = mask_array
            mask_filepaths[role] = mask_fp
            if features_report is not None and structures[role] in features_report.statistics.keys() \
                    and self.report_space in features_report.statistics[structures[role]].keys():
                statistics_filenames[role] = os.path.join("statistics", f"{structures[role]}_{self.report_space}_statistics.json")

        composition_statistics = compute_composition(masks=masks, spacing=spacing, composition=self.composition,
                                                     mask_filepaths=mask_filepaths, statistics_filenames=statistics_filenames)
        for note in notes:
            composition_statistics.notes.append(note)
        if features_report is None:
            composition_statistics.notes.append(f"no features computation report for timestamp {self.timestamp}, "
                                                f"per-structure measurements not referenced")

        report_uid = self._patient_parameters.generate_reporting_uid(prefix="TpRep")
        report = NeuroTimepointReportingStructure(id=report_uid, output_folder=ResourcesConfiguration.getInstance().output_folder,
                                             timestamp=self.timestamp, space=self.report_space)
        report.base_volume_uid = base_volume.unique_id
        if features_report is not None:
            report.features_report_uid = features_report.unique_id
        report.composition = composition_statistics
        self._patient_parameters.include_reporting(report_uid, report)
        report.to_disk()

    def __get_features_report(self):
        """
        Latest report of the features computation for the timestamp, None if there is none.
        """
        found = None
        for uid in self._patient_parameters.get_all_reportings_uids():
            report = self._patient_parameters.reportings[uid]
            if isinstance(report, NeuroReportingStructure) and str(report.timestamp) == str(self.timestamp):
                found = report
        return found

    def __get_mask_filepath(self, structure: str, base_volume_uid: str) -> str | None:
        """
        Mask of the structure on the grid of the base volume: drawn on the base volume, or registered into it.
        """
        annotation_class = get_type_from_enum_name(AnnotationClassType, structure)
        if annotation_class == -1 or annotation_class is None:
            logging.warning(f"[TimepointReportingStep] Unknown structure {structure} in the composition.")
            return None
        annotations = self._patient_parameters.get_all_annotations_uids_class_radiological_volume(volume_uid=base_volume_uid,
                                                                                                  annotation_class=annotation_class,
                                                                                                  include_coregistrations=True,
                                                                                                  return_objects=True)
        for annotation in annotations:
            if annotation.radiological_volume_uid == base_volume_uid:
                return annotation.usable_input_filepath
        for annotation in annotations:
            if base_volume_uid in annotation.registered_volumes.keys():
                return annotation.registered_volumes[base_volume_uid]["filepath"]
        return None
