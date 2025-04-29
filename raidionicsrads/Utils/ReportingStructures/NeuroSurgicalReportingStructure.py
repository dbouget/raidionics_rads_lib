from aenum import Enum, unique
import logging
import traceback
import os
import numpy as np
import operator
import json
import pandas as pd
import collections
from ..configuration_parser import ResourcesConfiguration
from ..io import generate_cortical_structures_labels_for_slicer, generate_subcortical_structures_labels_for_slicer


@unique
class ResectionCategoryType(Enum):
    """

    """
    _init_ = 'value string'

    SupR = 0, 'Supramaximal resection of CE tumor'
    ComR = 1, 'Complete resection of CE tumor'
    NeaR = 2, 'Near total resection of CE tumor'
    SubR = 3, 'Subtotal resection of CE tumor'
    ParR = 4, 'Partial resection of CE tumor'

    def __str__(self):
        return self.string


class SurgicalStatistics:
    """
    Specific class for holding the surgical report features
    """
    _tumor_volume_preop = None
    _tumor_volume_postop = None
    _extent_of_resection = None
    _resection_category = None
    _flairchanges_volume_postop = None
    _cavity_volume_postop = None

    def __init__(self, volume_pre: float = None, volume_post: float = None, eor: float = None, category: str = None,
                 flairchanges_volume_postop: float = None, _cavity_volume_postop: float = None):
        self._tumor_volume_preop = volume_pre
        self._tumor_volume_postop = volume_post
        self._extent_of_resection = eor
        self._resection_category = category
        self._flairchanges_volume_postop = flairchanges_volume_postop
        self._cavity_volume_postop = _cavity_volume_postop

    @property
    def tumor_volume_preop(self) -> float:
        return self._tumor_volume_preop

    @tumor_volume_preop.setter
    def tumor_volume_preop(self, value: float) -> None:
        self._tumor_volume_preop = value

    @property
    def tumor_volume_postop(self) -> float:
        return self._tumor_volume_postop

    @tumor_volume_postop.setter
    def tumor_volume_postop(self, value: float) -> None:
        self._tumor_volume_postop = value

    @property
    def extent_of_resection(self) -> float:
        return self._extent_of_resection

    @extent_of_resection.setter
    def extent_of_resection(self, value: float) -> None:
        self._extent_of_resection = value

    @property
    def resection_category(self) -> str:
        return self._resection_category

    @resection_category.setter
    def resection_category(self, value: str) -> None:
        self._resection_category = value

    @property
    def flairchanges_volume_postop(self) -> float:
        return self._flairchanges_volume_postop

    @flairchanges_volume_postop.setter
    def flairchanges_volume_postop(self, value: float) -> None:
        self._flairchanges_volume_postop = value

    @property
    def cavity_volume_postop(self) -> float:
        return self._cavity_volume_postop

    @cavity_volume_postop.setter
    def cavity_volume_postop(self, value: float) -> None:
        self._cavity_volume_postop = value


class NeuroSurgicalReportingStructure:
    """
    Reporting at a single timestamp with characteristics/features for the tumor.
    """
    _unique_id = None  # Internal unique identifier for the report
    _output_folder = None
    _statistics = None

    def __init__(self, id: str, output_folder: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._output_folder = output_folder
        self._statistics = SurgicalStatistics()

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._output_folder = None
        self._statistics = None

    @property
    def statistics(self) -> SurgicalStatistics:
        return self._statistics

    @statistics.setter
    def statistics(self, value: SurgicalStatistics) -> None:
        self._statistics = value

    def setup(self) -> None:
        pass

    def to_disk(self) -> None:
        output_folder = os.path.join(self._output_folder, "reporting")
        os.makedirs(output_folder, exist_ok=True)
        self.to_txt()
        self.to_json()
        self.to_csv()

    def to_txt(self) -> None:
        """

        Exporting the computed tumor characteristics and standardized report in filename.

        Parameters
        ----------

        Returns
        -------
        None
        """
        try:
            filename = os.path.join(self._output_folder, "reporting", "neuro_surgical_report.txt")
            logging.info(f"Exporting neurosurgical reporting to text in {filename}.")
            pfile = open(filename, 'w')
            pfile.write(f'########### Raidionics standardized surgical reporting ###########\n')
            pfile.write(' Preoperative tumor volume: {:.2f} ml\n'.format(self.statistics.tumor_volume_preop))
            pfile.write(' Postoperative tumor volume: {:.2f} ml\n'.format(self.statistics.tumor_volume_postop))
            pfile.write(' Extent of resection: {:.2f} %\n'.format(self.statistics.extent_of_resection))
            pfile.write(' Resection category: {}\n'.format(self.statistics.resection_category))
            if self.statistics.flairchanges_volume_postop is not None:
                pfile.write(' Postoperative flair changes volume: {:.2f} ml\n'.format(self.statistics.flairchanges_volume_postop))
            if self.statistics.cavity_volume_postop is not None:
                pfile.write(' Postoperative cavity volume: {:.2f} ml\n'.format(self.statistics.cavity_volume_postop))
            pfile.close()
        except Exception as e:
            raise RuntimeError(f"Neurosurgical reporting writing on disk as text failed with {e}")
        return

    def to_json(self) -> None:
        try:
            param_json = {}
            filename = os.path.join(self._output_folder, "reporting", "neuro_surgical_report.json")
            logging.info("Exporting surgical reporting to json in {}.".format(filename))
            param_json["preop_volume"] = self.statistics.tumor_volume_preop
            param_json["postop_volume"] = self.statistics.tumor_volume_postop
            param_json["eor"] = self.statistics.extent_of_resection
            param_json["resection_category"] = str(self.statistics.resection_category)
            param_json["flairchanges_postop_volume"] = self.statistics.flairchanges_volume_postop
            param_json["cavity_postop_volume"] = self.statistics.cavity_volume_postop
            with open(filename, 'w', newline='\n') as outfile:
                json.dump(param_json, outfile, indent=4, sort_keys=True)
        except Exception as e:
            raise RuntimeError(f"Neurosurgical reporting writing on disk as json failed with {e}")

        return

    def to_csv(self) -> None:
        try:
            filename = os.path.join(self._output_folder, "reporting", "neuro_surgical_report.csv")
            logging.info(f"Exporting neurosurgical reporting to csv in {filename}.")
            column_names = []
            all_values = []

            column_names.extend(["Volume tumor preop (ml)", "Volume tumor postop (ml)"])
            all_values.extend([self.statistics.tumor_volume_preop, self.statistics.tumor_volume_postop])
            column_names.extend(["Extent of resection (%)", "Resection category"])
            all_values.extend([self.statistics.extent_of_resection, self.statistics.resection_category])
            column_names.extend(["Volume FLAIR changes postop (ml)", "Volume cavity postop (ml)"])
            all_values.extend([self.statistics.flairchanges_volume_postop, self.statistics.cavity_volume_postop])
            values_df = pd.DataFrame(np.asarray([all_values]), columns=column_names)
            values_df.to_csv(filename, index=False)
        except Exception as e:
            raise RuntimeError(f"Neurosurgical reporting writing on disk as csv failed with {e}")

