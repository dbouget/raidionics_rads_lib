from abc import ABC, abstractmethod


class AbstractReportingStructure(ABC):
    """
    Common part of the application-oriented reports (e.g., neuro, neurosurgical, mediastinum), built from the generic
    statistics (see Utils/DataStructures/StatisticsStructure.py) and dumped on disk in the different formats.
    """
    _unique_id = None  # Internal unique identifier for the report
    _output_folder = None
    _timestamp = None  # Timestamp attached to the report, if any

    def __init__(self, id: str, output_folder: str, timestamp: str = None):
        self._unique_id = id
        self._output_folder = output_folder
        self._timestamp = timestamp

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @property
    def timestamp(self) -> str:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: str) -> None:
        self._timestamp = value

    def to_disk(self) -> None:
        self.to_txt()
        self.to_csv()
        self.to_json()

    @abstractmethod
    def to_txt(self) -> None:
        pass

    @abstractmethod
    def to_json(self) -> None:
        pass

    @abstractmethod
    def to_csv(self) -> None:
        pass
