import os
import numpy as np
import re
from typing import List
from ..configuration_parser import ResourcesConfiguration
from .RadiologicalVolumeStructure import RadiologicalVolume
from .AnnotationStructure import Annotation, AnnotationClassType
from .RegistrationStructure import Registration


class PatientParameters:
    _unique_id = None  # Internal unique identifier for the patient
    _input_filepath = None  #
    _timestamps = {}  # All timestamps for the current patient.
    _radiological_volumes = {}  # All radiological volume instances loaded for the current patient.
    _annotation_volumes = {}  # All Annotation instances loaded for the current patient.
    _atlas_volumes = {}  # All Atlas instances loaded for the current patient.
    _registrations = {}  # All registration transforms.

    def __init__(self, id: str, patient_filepath: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._input_filepath = patient_filepath

        if not patient_filepath or not os.path.exists(patient_filepath):
            # Error case
            return

        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._input_filepath = None
        self._timestamps = {}
        self._radiological_volumes = {}
        self._annotation_volumes = {}
        self._atlas_volumes = {}
        self._registrations = {}

    def __init_from_scratch(self):
        timestamp_folders = []
        for _, dirs, _ in os.walk(self._input_filepath):
            for d in dirs:
                timestamp_folders.append(d)
            break

        ts_folders_dict = {}
        for i in timestamp_folders:
            ts_folders_dict[int(re.search(r'\d+', i).group())] = i

        ordered_ts_folders = dict(sorted(ts_folders_dict.items(), key=lambda item: item[0], reverse=False))

        for i, ts in enumerate(list(ordered_ts_folders.keys())):
            ts_folder = os.path.join(self._input_filepath, ordered_ts_folders[ts])
            patient_files = []

            timestamp_uid = "T" + str(i)
            timestamp_instance = TimestampParameters(id=timestamp_uid, timestamp_filepath=ts_folder)
            self._timestamps[timestamp_uid] = timestamp_instance

            for _, _, files in os.walk(ts_folder):
                for f in files:
                    if '.'.join(f.split('.')[1:]) in ResourcesConfiguration.getInstance().get_accepted_image_formats():
                        patient_files.append(f)
                break

            for f in patient_files:
                # Generating a unique id for the volume
                base_data_uid = os.path.basename(f).strip().split('.')[0]
                non_available_uid = True
                while non_available_uid:
                    data_uid = 'V' + str(np.random.randint(0, 10000)) + '_' + base_data_uid
                    if data_uid not in list(self._radiological_volumes.keys()):
                        non_available_uid = False
                self._radiological_volumes[data_uid] = RadiologicalVolume(uid=data_uid,
                                                                          input_filename=os.path.join(ts_folder, f),
                                                                          timestamp_uid=timestamp_uid)

    def include_annotation(self, anno_uid, annotation):
        self._annotation_volumes[anno_uid] = annotation

    def include_registration(self, reg_uid, registration):
        self._registrations[reg_uid] = registration

    def get_radiological_volume_uid(self, timestamp: int, sequence: str) -> str:
        for v in self._radiological_volumes.keys():
            if self._radiological_volumes[v]._timestamp_id == "T" + str(timestamp) and str(self._radiological_volumes[v]._sequence_type) == sequence:
                return v
        return "-1"

    def get_all_annotations_uids(self) -> List[str]:
        return list(self._annotation_volumes.keys())

    def get_all_annotations_uids_radiological_volume(self, volume_uid: str) -> List[str]:
        res = []
        for v in self._annotation_volumes.keys():
            if self._annotation_volumes[v]._radiological_volume_uid == volume_uid:
                res.append(v)
        return res

    def get_all_annotations_uids_class_radiological_volume(self, volume_uid: str,
                                                           annotation_class: AnnotationClassType) -> List[str]:
        res = []
        for v in self._annotation_volumes.keys():
            if self._annotation_volumes[v]._radiological_volume_uid == volume_uid and \
                    self._annotation_volumes[v]._annotation_type == annotation_class:
                res.append(v)
        return res


class TimestampParameters:
    _unique_id = None  # Internal unique identifier for the patient
    _input_filepath = None  #

    def __init__(self, id: str, timestamp_filepath: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._input_filepath = timestamp_filepath

        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._input_filepath = None

    def __init_from_scratch(self):
        pass
