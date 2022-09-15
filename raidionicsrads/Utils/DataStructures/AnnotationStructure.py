import os
from typing import List
from aenum import Enum, unique
from ..utilities import get_type_from_string, input_file_type_conversion
from ..configuration_parser import ResourcesConfiguration


@unique
class AnnotationClassType(Enum):
    """

    """
    _init_ = 'value string'

    Brain = 0, 'Brain'
    Tumor = 1, 'Tumor'

    Lungs = 100, 'Lungs'
    Airways = 101, 'Airways'

    def __str__(self):
        return self.string


class Annotation:
    """
    Class defining how an annotation should be handled.
    """
    _unique_id = None  # Internal unique identifier for the annotation instance
    _raw_input_filepath = None  # Original volume filepath on the user's machine
    _usable_input_filepath = None  #
    _output_folder = None  #
    _radiological_volume_uid = None
    _annotation_type = None
    _registered_volumes = {}

    def __init__(self, uid: str, input_filename: str, output_folder: str, radiological_volume_uid: str,
                 annotation_class: str) -> None:
        self.__reset()
        self._unique_id = uid
        self._raw_input_filepath = input_filename
        self._output_folder = output_folder
        self._radiological_volume_uid = radiological_volume_uid
        self._annotation_type = get_type_from_string(AnnotationClassType, annotation_class)
        self.__init_from_scratch()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._raw_input_filepath = None
        self._usable_input_filepath = None
        self._output_folder = None
        self._radiological_volume_uid = None
        self._annotation_type = None
        self._registered_volumes = {}

    def get_unique_id(self) -> str:
        return self._unique_id

    def get_output_folder(self) -> str:
        return self._output_folder

    def get_usable_input_filepath(self) -> str:
        return self._usable_input_filepath

    def get_parent_radiological_volume_uid(self) -> str:
        return self._radiological_volume_uid

    def get_annotation_type_enum(self) -> Enum:
        return self._annotation_type

    def get_annotation_type_str(self) -> str:
        return str(self._annotation_type)

    def set_annotation_type(self, type: str) -> None:
        """
        Update the annotation class type by providing its string version, which will be matched to the proper
        element inside AnnotationClassType.

        Parameters
        ----------
        type: str
            New annotation type to associate with the current instance.
        """
        if isinstance(type, str):
            ctype = get_type_from_string(AnnotationClassType, type)
            if ctype != -1:
                self._sequence_type = ctype
        elif isinstance(type, AnnotationClassType):
            self._sequence_type = type

    def include_registered_volume(self, filepath: str, registration_uid: str, destination_space_uid: str) -> None:
        if destination_space_uid in list(self._registered_volumes.keys()):
            raise ValueError("[AnnotationStructure] Trying to insert a registered volume with an already existing "
                             "destination space key: {}.".format(destination_space_uid))
        self._registered_volumes[destination_space_uid] = {"filepath": filepath, "registration_uid": registration_uid}

    def get_registered_volume_info(self, destination_space_uid: str):
        return self._registered_volumes[destination_space_uid]

    def get_registered_volume_destination_uids(self) -> List[str]:
        return list(self._registered_volumes.keys())

    def __init_from_scratch(self):
        """
        Mostly in case the annotation was provided by the user in a non-nifti format.
        """
        self._usable_input_filepath = input_file_type_conversion(input_filename=self._raw_input_filepath,
                                                                 output_folder=self._output_folder)
