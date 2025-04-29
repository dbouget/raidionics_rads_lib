import os
import numpy as np
import re
import glob
import pandas as pd
import logging
import nibabel as nib
from typing import List
from ..configuration_parser import ResourcesConfiguration
from ..utilities import input_file_category_disambiguation, get_type_from_enum_name
from .RadiologicalVolumeStructure import RadiologicalVolume
from .AnnotationStructure import Annotation, AnnotationClassType
from .RegistrationStructure import Registration


class PatientParameters:
    _unique_id = None  # Internal unique identifier for the patient
    _input_filepath = None  # Folder path containing all the data for the current patient.
    _timestamps = {}  # All timestamps for the current patient.
    _radiological_volumes = {}  # All radiological volume instances loaded for the current patient.
    _annotation_volumes = {}  # All Annotation instances loaded for the current patient.
    _atlas_volumes = {}  # All Atlas instances loaded for the current patient.
    _registrations = {}  # All registration transforms.
    _reportings = {}  # All clinical reports (if applicable).

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
        self._reportings = {}

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def input_filepath(self) -> str:
        return self._input_filepath

    @property
    def radiological_volumes(self) -> dict:
        return self._radiological_volumes

    @property
    def annotation_volumes(self) -> dict:
        return self._annotation_volumes

    @property
    def registrations(self) -> dict:
        return self._registrations

    @property
    def reportings(self) -> dict:
        return self._reportings

    def __init_from_scratch(self):
        """
        Iterating through the patient folder to identify the radiological volumes for each timestamp.

        In case of stripped inputs (i.e., skull-stripped or lung-stripped), the corresponding mask should be created
        for each input
        """
        try:
            timestamp_folders = []
            for _, dirs, _ in os.walk(self._input_filepath):
                for d in dirs:
                    timestamp_folders.append(d)
                break

            ts_folders_dict = {}
            for i in timestamp_folders:
                if re.search(r'\d+', i):  # Skipping folders without an integer inside, otherwise assuming timestamps from 0 onwards
                    ts_folders_dict[int(re.search(r'\d+', i).group())] = i

            ordered_ts_folders = dict(sorted(ts_folders_dict.items(), key=lambda item: item[0], reverse=False))

            for i, ts in enumerate(list(ordered_ts_folders.keys())):
                ts_folder = os.path.join(self._input_filepath, ordered_ts_folders[ts])
                if ResourcesConfiguration.getInstance().caller == 'raidionics':  # Specifics to cater to Raidionics
                    ts_folder = os.path.join(ts_folder, 'raw')
                patient_files = []

                timestamp_uid = "T" + str(i)
                timestamp_instance = TimestampParameters(id=timestamp_uid, timestamp_filepath=ts_folder)
                self._timestamps[timestamp_uid] = timestamp_instance

                for _, _, files in os.walk(ts_folder):
                    for f in files:
                        if '.'.join(f.split('.')[1:]) in ResourcesConfiguration.getInstance().get_accepted_image_formats():
                            patient_files.append(f)
                    break

                annotation_files = []
                for f in patient_files:
                    file_content_type = input_file_category_disambiguation(os.path.join(ts_folder, f))
                    # Generating a unique id for the radiological volume
                    if file_content_type == "Volume":
                        base_data_uid = os.path.basename(f).strip().split('.')[0]
                        non_available_uid = True
                        while non_available_uid:
                            data_uid = 'V' + str(np.random.randint(0, 10000)) + '_' + base_data_uid
                            if data_uid not in list(self._radiological_volumes.keys()):
                                non_available_uid = False
                        self._radiological_volumes[data_uid] = RadiologicalVolume(uid=data_uid,
                                                                                  input_filename=os.path.join(ts_folder, f),
                                                                                  timestamp_uid=timestamp_uid)
                    elif file_content_type == "Annotation":
                        annotation_files.append(f)

                # Iterating over the annotation files in a second time, when all the parent objects have been created
                for f in annotation_files:
                    # Collecting the base name of the radiological volume, often before a label or annotation tag
                    base_name = os.path.basename(f).strip().split('.')[0].split('label')[0][:-1]
                    if ResourcesConfiguration.getInstance().caller == 'raidionics':
                        base_name = os.path.basename(f).strip().split('.')[0].split('annotation')[0][:-1]
                    parent_link = [base_name in x for x in list(self._radiological_volumes.keys())]
                    if True in parent_link:
                        parent_uid = list(self._radiological_volumes.keys())[parent_link.index(True)]
                        non_available_uid = True
                        while non_available_uid:
                            data_uid = 'A' + str(np.random.randint(0, 10000)) + '_' + base_name
                            if data_uid not in list(self.annotation_volumes.keys()):
                                non_available_uid = False
                        if ResourcesConfiguration.getInstance().caller == 'raidionics':
                            class_name = os.path.basename(f).strip().split('.')[0].split('annotation')[1][1:]
                        else:
                            class_name = os.path.basename(f).strip().split('.')[0].split('label')[1][1:]
                        self.annotation_volumes[data_uid] = Annotation(uid=data_uid,
                                                                        input_filename=os.path.join(ts_folder, f),
                                                                        output_folder=self._radiological_volumes[parent_uid].output_folder,
                                                                        radiological_volume_uid=parent_uid,
                                                                        annotation_class=class_name)
                    else:
                        # Case where the annotation does not match any radiological volume, has to be left aside
                        pass
                registration_folders = [os.path.join(ts_folder, d) for d in os.listdir(ts_folder) if os.path.isdir(os.path.join(ts_folder, d))]
                for rf in registration_folders:
                    registered_radiological_volumes = []
                    registered_labels = []
                    for _, _, files in os.walk(rf):
                        for f in files:
                            if "label" in f or "annotation" in f:
                                registered_labels.append(f)
                            else:
                                registered_radiological_volumes.append(f)
                    for rr in registered_radiological_volumes:
                        fixed_volume = self.get_radiological_volume_by_base_filename(base_fn=os.path.basename(rf[:-1]).replace("_space", ""))
                        reg_volume = self.get_radiological_volume_by_base_filename(base_fn=rr.split('_reg')[0])
                        reg_volume.include_registered_volume(filepath=os.path.join(rf, rr), registration_uid=None,
                                                             destination_space_uid=fixed_volume.unique_id)

            sequences_filename = os.path.join(self._input_filepath, 'mri_sequences.csv')
            if os.path.exists(sequences_filename):
                df = pd.read_csv(sequences_filename)
                volume_basenames = list(df['File'].values)
                for vn in volume_basenames:
                    volume_object = self.get_radiological_volume_by_base_filename(vn)
                    if volume_object:
                        volume_object.set_sequence_type(df.loc[df['File'] == vn]['MRI sequence'].values[0])
                    else:
                        logging.warning("[PatientStructure] Filename {} not matching any radiological volume volume.".format(vn))

            # Setting up masks (i.e., brain or lungs) if stripped inputs are used.
            if ResourcesConfiguration.getInstance().predictions_use_stripped_data:
                target_type = AnnotationClassType.Brain if ResourcesConfiguration.getInstance().diagnosis_task == 'neuro_diagnosis' else AnnotationClassType.Lungs
                for uid in self.get_all_radiological_volume_uids():
                    volume = self.get_radiological_volume(uid)
                    volume_nib = nib.load(volume.raw_input_filepath)
                    img_data = volume_nib.get_fdata()[:]
                    mask = np.zeros(img_data.shape)
                    mask[img_data != 0] = 1
                    mask_fn = os.path.join(volume.output_folder,
                                           os.path.basename(volume.raw_input_filepath).split('.')[0] + '_label_' + str(target_type) + '.nii.gz')

                    nib.save(nib.Nifti1Image(mask, affine=volume_nib.affine), mask_fn)
                    non_available_uid = True
                    anno_uid = None
                    while non_available_uid:
                        anno_uid = 'A' + str(np.random.randint(0, 10000))
                        if anno_uid not in self.get_all_annotations_uids():
                            non_available_uid = False
                    self.annotation_volumes[anno_uid] = Annotation(uid=data_uid, input_filename=mask_fn,
                                                                    output_folder=volume.output_folder,
                                                                    radiological_volume_uid=uid,
                                                                    annotation_class=target_type)
        except Exception as e:
            raise ValueError("Patient structure setup from disk folder failed with: {}".format(e))

    def include_annotation(self, anno_uid, annotation):
        self.annotation_volumes[anno_uid] = annotation

    def include_registration(self, reg_uid, registration):
        self.registrations[reg_uid] = registration

    def include_reporting(self, report_uid, report):
        self.reportings[report_uid] = report

    def get_input_from_json(self, input_json: dict):
        """
        Automatic identifies and returns the proper structure instance based on the content of the input json dict.
        """
        # Use-case where the input is in its original reference space
        if input_json["space"]["timestamp"] == input_json["timestamp"] and \
                input_json["space"]["sequence"] == input_json["sequence"]:
            volume_uid = self.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                          sequence=input_json["sequence"])
            # Use-case where the input is actually an annotation and not a raw radiological volume
            if input_json["labels"]:
                annotation_type = get_type_from_enum_name(AnnotationClassType, input_json["labels"])
                anno_uids = self.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid,
                                                                                    annotation_class=annotation_type)
                if len(anno_uids) != 0:
                    return self.get_annotation(anno_uids[0])
                else:
                    raise ValueError("No annotation file existing for the specified json input with:\n {}.\n".format(input_json))
            else:
                return self.get_radiological_volume(volume_uid)
        else:  # The input is in a registered space
            volume_uid = self.get_radiological_volume_uid(timestamp=input_json["timestamp"],
                                                          sequence=input_json["sequence"])
            if volume_uid == "-1":
                raise ValueError("No radiological volume for {}.".format(input_json))

            ref_space_uid = self.get_radiological_volume_uid(timestamp=input_json["space"]["timestamp"],
                                                             sequence=input_json["space"]["sequence"])
            if ref_space_uid == "-1" and input_json["space"]["timestamp"] != "-1":
                raise ValueError("No radiological volume for {}.".format(input_json))
            else:  # @TODO. The reference space is an atlas, have to make an extra-pass for this.
                pass
            # Use-case where the input is actually an annotation and not a raw radiological volume
            if input_json["labels"]:
                annotation_type = get_type_from_enum_name(AnnotationClassType, input_json["labels"])
                if annotation_type == -1:
                    raise ValueError("No radiological volume for {}.".format(input_json))

                anno_uids = self.get_all_annotations_uids_class_radiological_volume(volume_uid=volume_uid,
                                                                                    annotation_class=annotation_type)
                if len(anno_uids) == 0:
                    raise ValueError("No radiological volume for {}.".format(input_json))
                anno_uid = anno_uids[0]
                volume = self.get_annotation(annotation_uid=anno_uid).get_registered_volume_info(ref_space_uid)
                input_fp = volume["filepath"]
                if not os.path.exists(input_fp):
                    raise ValueError("No radiological volume for {}.".format(input_json))
                else:
                    return volume
                # Use-case where the provided inputs are already co-registered
            elif ResourcesConfiguration.getInstance().predictions_use_registered_data:
                volume = self.get_radiological_volume(volume_uid=volume_uid)
                input_fp = volume.usable_input_filepath
                if not os.path.exists(input_fp):
                    raise ValueError("No radiological volume for {}.".format(input_json))
                else:
                    return volume
            else:
                volume = self.get_radiological_volume(volume_uid=volume_uid).get_registered_volume_info(ref_space_uid)
                reg_fp = volume["filepath"]
                if not os.path.exists(reg_fp):
                    raise ValueError("No radiological volume for {}.".format(input_json))
                else:
                    return volume

    def get_all_radiological_volume_uids(self) -> List[str]:
        return list(self.radiological_volumes.keys())

    def get_all_radiological_volumes_for_timestamp(self, timestamp: int) -> List[RadiologicalVolume]:
        res_list = []
        for v in self.radiological_volumes.keys():
            if self.radiological_volumes[v]._timestamp_id == "T" + str(timestamp):
                res_list.append(self.radiological_volumes[v])
        return res_list

    def get_all_radiological_volumes_uids_for_timestamp(self, timestamp: int) -> List[str]:
        res_list = []
        for v in self.radiological_volumes.keys():
            if self.radiological_volumes[v]._timestamp_id == "T" + str(timestamp):
                res_list.append(v)
        return res_list

    def get_radiological_volume_uid(self, timestamp: int, sequence: str) -> str:
        for v in self.radiological_volumes.keys():
            if self.radiological_volumes[v]._timestamp_id == "T" + str(timestamp) and str(self.radiological_volumes[v]._sequence_type) == sequence:
                return v
        return "-1"

    def get_radiological_volume(self, volume_uid: str) -> RadiologicalVolume:
        return self.radiological_volumes[volume_uid]

    def get_radiological_volume_by_base_filename(self, base_fn: str):
        result = None
        for im in self.radiological_volumes:
            if os.path.basename(self.radiological_volumes[im].usable_input_filepath).split('.')[0] == base_fn:
                return self.radiological_volumes[im]
        return result

    def get_all_annotations_uids(self) -> List[str]:
        return list(self.annotation_volumes.keys())

    def get_annotation(self, annotation_uid: str) -> Annotation:
        return self.annotation_volumes[annotation_uid]

    def get_all_annotations_radiological_volume(self, volume_uid: str) -> List[Annotation]:
        res = []
        for v in self.annotation_volumes.keys():
            if self.annotation_volumes[v]._radiological_volume_uid == volume_uid:
                res.append(self.annotation_volumes[v])
        return res

    def get_all_annotations_for_timestamp(self, timestamp: int) -> List[Annotation]:
        res_list = []
        volumes = self.get_all_radiological_volumes_uids_for_timestamp(timestamp=timestamp)
        for a in self.annotation_volumes:
            if a.radiological_volume_uid in volumes:
                res_list.append(a)
        return res_list

    def get_all_annotations_for_timestamp_and_structure(self, timestamp: int,
                                                        structure: str) -> List[Annotation]:
        res_list = []
        volumes = self.get_all_radiological_volumes_uids_for_timestamp(timestamp=timestamp)
        for a in list(self.annotation_volumes.keys()):
            if self.annotation_volumes[a].radiological_volume_uid in volumes and self.annotation_volumes[a].get_annotation_type_name() == structure:
                res_list.append(self.annotation_volumes[a])
        return res_list

    def get_all_annotations_uids_radiological_volume(self, volume_uid: str) -> List[str]:
        res = []
        for v in self.annotation_volumes.keys():
            if self.annotation_volumes[v]._radiological_volume_uid == volume_uid:
                res.append(v)
        return res

    def get_all_registered_annotations_uids_radiological_volume(self, volume_uid: str) -> List[str]:
        res = []
        for v in self.annotation_volumes.keys():
            for r in self.annotation_volumes[v].registered_volumes.keys():
                if r == volume_uid:
                    res.append(v)
        return res

    def get_all_annotations_uids_class_radiological_volume(self, volume_uid: str,
                                                           annotation_class: AnnotationClassType,
                                                           include_coregistrations: bool = False) -> List[str]:
        res = []
        for v in self.annotation_volumes.keys():
            if self.annotation_volumes[v]._radiological_volume_uid == volume_uid and \
                    self.annotation_volumes[v]._annotation_type == annotation_class:
                res.append(v)
            if include_coregistrations:
                if volume_uid in self.annotation_volumes[v].registered_volumes.keys() and \
                        self.annotation_volumes[v]._annotation_type == annotation_class:
                    res.append(v)
        return res

    def get_all_annotations_fns_class_radiological_volume(self, volume_uid: str,
                                                           annotation_class: AnnotationClassType,
                                                           include_coregistrations: bool = False) -> List[str]:
        """
        @TODO. What if the volume_uid is an atlas?
        """
        res = []
        for v in self.annotation_volumes.keys():
            if self.annotation_volumes[v]._radiological_volume_uid == volume_uid and \
                    self.annotation_volumes[v]._annotation_type == annotation_class:
                res.append(self.annotation_volumes[v].usable_input_filepath)
            if include_coregistrations:
                if volume_uid in self.annotation_volumes[v].registered_volumes.keys() and \
                        self.annotation_volumes[v]._annotation_type == annotation_class:
                    res.append(self.annotation_volumes[v].registered_volumes[volume_uid]["filepath"])
        return res

    def get_registration_by_uids(self, fixed_uid: str, moving_uid: str) -> Registration:
        registration = None
        for r in list(self.registrations.keys()):
            if self.registrations[r].fixed_uid == fixed_uid and self.registrations[r].moving_uid == moving_uid:
                return self.registrations[r]
        return registration

    def get_registration_by_json(self, fixed: dict, moving: dict) -> Registration:
        fixed_ts = fixed["timestamp"]
        fixed_seq = fixed["sequence"]

        moving_ts = moving["timestamp"]
        moving_seq = moving["sequence"]

        fixed_uid = None
        moving_uid = None
        if fixed_ts == -1:
            fixed_uid = 'MNI'
        else:
            fixed_uid = self.get_radiological_volume_uid(fixed_ts, fixed_seq)

        if moving_ts == -1:
            moving_uid = 'MNI'
        else:
            moving_uid = self.get_radiological_volume_uid(moving_ts, moving_seq)

        return self.get_registration_by_uids(fixed_uid=fixed_uid, moving_uid=moving_uid)

    def get_all_reportings_uids(self) -> List[str]:
        return list(self.reportings.keys())


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

    @property
    def unique_id(self) -> str:
        return self._unique_id

    def __init_from_scratch(self):
        pass
