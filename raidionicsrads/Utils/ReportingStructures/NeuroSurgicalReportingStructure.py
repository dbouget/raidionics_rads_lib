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
    Resection assessment according to the RANO 2.0 guidelines.
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
    @TODO. Should be organized neatly, based on main MR scan sequence used or preop/postop?
    """
    _tumor_volume_preop = None
    _tumor_volume_postop = None
    _extent_of_resection = None
    _resection_category = None
    _flairchanges_volume_preop = None
    _flairchanges_volume_postop = None
    _extent_of_resection_flair = None
    _cavity_volume_postop = None
    _brain_volume_preop = None
    _brain_volume_postop = None
    _brain_volume_change = None
    _necrosis_volume_preop = None
    _necrosis_volume_postop = None
    _necrosis_volume_change = None
    _tumor_to_brain_ratio_preop = None
    _tumor_to_brain_ratio_postop = None

    def __init__(self, volume_pre: float = None, volume_post: float = None, eor: float = None,
                 eor_flair: float = None, category: str = None, flairchanges_volume_preop: float = None,
                 flairchanges_volume_postop: float = None, cavity_volume_postop: float = None,
                 brain_volume_preop: float = None, brain_volume_postop: float = None,
                 brain_volume_change: float = None, tumor_to_brain_ratio_preop: float = None,
                 tumor_to_brain_ratio_postop: float = None, necrosis_volume_preop: float = None,
                 necrosis_volume_postop: float = None, necrosis_volume_change: float = None):
        self._tumor_volume_preop = volume_pre
        self._tumor_volume_postop = volume_post
        self._extent_of_resection = eor
        self._resection_category = category
        self._flairchanges_volume_preop = flairchanges_volume_preop
        self._flairchanges_volume_postop = flairchanges_volume_postop
        self._extent_of_resection_flair = eor_flair
        self._cavity_volume_postop = cavity_volume_postop
        self._brain_volume_preop = brain_volume_preop
        self._brain_volume_postop = brain_volume_postop
        self._brain_volume_change = brain_volume_change
        self._tumor_to_brain_ratio_preop = tumor_to_brain_ratio_preop
        self._tumor_to_brain_ratio_postop = tumor_to_brain_ratio_postop
        self._necrosis_volume_preop = necrosis_volume_preop
        self._necrosis_volume_postop = necrosis_volume_postop
        self._necrosis_volume_change = necrosis_volume_change

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
    def extent_of_resection_flair(self) -> float:
        return self._extent_of_resection_flair

    @extent_of_resection_flair.setter
    def extent_of_resection_flair(self, value: float) -> None:
        self._extent_of_resection_flair = value

    @property
    def resection_category(self) -> str:
        return self._resection_category

    @resection_category.setter
    def resection_category(self, value: str) -> None:
        self._resection_category = value

    @property
    def flairchanges_volume_preop(self) -> float:
        return self._flairchanges_volume_preop

    @flairchanges_volume_preop.setter
    def flairchanges_volume_preop(self, value: float) -> None:
        self._flairchanges_volume_preop = value

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

    @property
    def brain_volume_preop(self) -> float:
        return self._brain_volume_preop

    @brain_volume_preop.setter
    def brain_volume_preop(self, value: float) -> None:
        self._brain_volume_preop = value

    @property
    def brain_volume_postop(self) -> float:
        return self._brain_volume_postop

    @brain_volume_postop.setter
    def brain_volume_postop(self, value: float) -> None:
        self._brain_volume_postop = value

    @property
    def brain_volume_change(self) -> float:
        return self._brain_volume_change

    @brain_volume_change.setter
    def brain_volume_change(self, value: float) -> None:
        self._brain_volume_change = value

    @property
    def necrosis_volume_preop(self) -> float:
        return self._necrosis_volume_preop

    @necrosis_volume_preop.setter
    def necrosis_volume_preop(self, value: float) -> None:
        self._necrosis_volume_preop = value

    @property
    def necrosis_volume_postop(self) -> float:
        return self._necrosis_volume_postop

    @necrosis_volume_postop.setter
    def necrosis_volume_postop(self, value: float) -> None:
        self._necrosis_volume_postop = value

    @property
    def necrosis_volume_change(self) -> float:
        return self._necrosis_volume_change

    @necrosis_volume_change.setter
    def necrosis_volume_change(self, value: float) -> None:
        self._necrosis_volume_change = value

    @property
    def tumor_to_brain_ratio_preop(self) -> float:
        return self._tumor_to_brain_ratio_preop

    @tumor_to_brain_ratio_preop.setter
    def tumor_to_brain_ratio_preop(self, value: float) -> None:
        self._tumor_to_brain_ratio_preop = value

    @property
    def tumor_to_brain_ratio_postop(self) -> float:
        return self._tumor_to_brain_ratio_postop

    @tumor_to_brain_ratio_postop.setter
    def tumor_to_brain_ratio_postop(self, value: float) -> None:
        self._tumor_to_brain_ratio_postop = value


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
            pfile.write(' Preoperative brain volume: {:.2f} ml\n'.format(self.statistics.brain_volume_preop))
            pfile.write(' Postoperative brain volume: {:.2f} ml\n'.format(self.statistics.brain_volume_postop))
            pfile.write(' Brain volume change: {:.2f} ml\n'.format(self.statistics.brain_volume_change))
            pfile.write(' Preoperative tumor volume: {:.2f} ml\n'.format(self.statistics.tumor_volume_preop))
            pfile.write(' Postoperative tumor volume: {:.2f} ml\n'.format(self.statistics.tumor_volume_postop))
            pfile.write(' Extent of resection: {:.2f} %\n'.format(self.statistics.extent_of_resection))
            pfile.write(' Resection category: {}\n'.format(self.statistics.resection_category))
            if self.statistics.necrosis_volume_preop is not None:
                pfile.write(' Preoperative necrosis volume: {:.2f} ml\n'.format(self.statistics.necrosis_volume_preop))
            if self.statistics.necrosis_volume_postop is not None:
                pfile.write(' Postoperative necrosis volume: {:.2f} ml\n'.format(self.statistics.necrosis_volume_postop))
            if self.statistics.necrosis_volume_change is not None:
                pfile.write(' Necrosis volume change: {:.2f} ml\n'.format(self.statistics.necrosis_volume_change))
            if self.statistics.flairchanges_volume_preop is not None:
                pfile.write(' Preoperative flair changes volume: {:.2f} ml\n'.format(self.statistics.flairchanges_volume_preop))
            if self.statistics.flairchanges_volume_postop is not None:
                pfile.write(' Postoperative flair changes volume: {:.2f} ml\n'.format(self.statistics.flairchanges_volume_postop))
            if self.statistics.extent_of_resection_flair is not None:
                pfile.write(' Flair changes volume change: {:.2f} ml\n'.format(self.statistics.extent_of_resection_flair))
            if self.statistics.cavity_volume_postop is not None:
                pfile.write(' Postoperative cavity volume: {:.2f} ml\n'.format(self.statistics.cavity_volume_postop))
            pfile.write(' Tumor to brain ratio preop: {:.2f} %\n'.format(self.statistics.tumor_to_brain_ratio_preop))
            pfile.write(' Tumor to brain ratio postop: {:.2f} %\n'.format(self.statistics.tumor_to_brain_ratio_postop))
            pfile.close()
        except Exception as e:
            raise RuntimeError(f"Neurosurgical reporting writing on disk as text failed with {e}")
        return

    def to_json(self) -> None:
        try:
            param_json = {}
            filename = os.path.join(self._output_folder, "reporting", "neuro_surgical_report.json")
            logging.info("Exporting surgical reporting to json in {}.".format(filename))
            param_json["tumor_preop_volume"] = self.statistics.tumor_volume_preop
            param_json["tumor_postop_volume"] = self.statistics.tumor_volume_postop
            param_json["eor"] = self.statistics.extent_of_resection
            param_json["resection_category"] = str(self.statistics.resection_category)
            param_json["flairchanges_preop_volume"] = self.statistics.flairchanges_volume_preop
            param_json["flairchanges_postop_volume"] = self.statistics.flairchanges_volume_postop
            param_json["eor_flairchanges"] = self.statistics.extent_of_resection_flair
            param_json["cavity_postop_volume"] = self.statistics.cavity_volume_postop
            param_json["brain_preop_volume"] = self.statistics.brain_volume_preop
            param_json["brain_postop_volume"] = self.statistics.brain_volume_postop
            param_json["brain_volume_change"] = self.statistics.brain_volume_change
            param_json["tumor_to_brain_ratio_preop"] = self.statistics.tumor_to_brain_ratio_preop
            param_json["tumor_to_brain_ratio_postop"] = self.statistics.tumor_to_brain_ratio_postop
            param_json["necrosis_preop_volume"] = self.statistics.necrosis_volume_preop
            param_json["necrosis_postop_volume"] = self.statistics.necrosis_volume_postop
            param_json["necrosis_volume_change"] = self.statistics.necrosis_volume_change
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

            column_names.extend(["Volume brain preop (ml)", "Volume brain postop (ml)"])
            all_values.extend([self.statistics.brain_volume_preop, self.statistics.brain_volume_postop])
            column_names.extend(["Volume tumor preop (ml)", "Volume tumor postop (ml)"])
            all_values.extend([self.statistics.tumor_volume_preop, self.statistics.tumor_volume_postop])
            column_names.extend(["Volume necrosis preop (ml)", "Volume necrosis postop (ml)"])
            all_values.extend([self.statistics.necrosis_volume_preop, self.statistics.necrosis_volume_postop])
            column_names.extend(["Extent of resection (%)", "Resection category"])
            all_values.extend([self.statistics.extent_of_resection, self.statistics.resection_category])
            column_names.extend(["Volume FLAIR changes preop (ml)", "Volume FLAIR changes postop (ml)",
                                 "Volume cavity postop (ml)"])
            all_values.extend([self.statistics.flairchanges_volume_preop, self.statistics.flairchanges_volume_postop,
                               self.statistics.cavity_volume_postop])
            column_names.extend(["FLAIRchanges volume change (%)", "Brain volume change (%)", "Necrosis volume change (%)"])
            all_values.extend([self.statistics.extent_of_resection_flair, self.statistics.brain_volume_change,
                               self.statistics.necrosis_volume_change])
            column_names.extend(["Tumor to brain ratio preop (%)", "Tumor to brain ratio postop (%)"])
            all_values.extend([self.statistics.tumor_to_brain_ratio_preop, self.statistics.tumor_to_brain_ratio_postop])
            values_df = pd.DataFrame(np.asarray([all_values]), columns=column_names)
            values_df.to_csv(filename, index=False)
        except Exception as e:
            raise RuntimeError(f"Neurosurgical reporting writing on disk as csv failed with {e}")

