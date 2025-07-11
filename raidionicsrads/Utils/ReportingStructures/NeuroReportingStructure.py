import logging
import shutil
import traceback
import os
import numpy as np
import operator
import json
import pandas as pd
from typing import Dict
import collections
from ..configuration_parser import ResourcesConfiguration
from ..io import generate_cortical_structures_labels_for_slicer, generate_subcortical_structures_labels_for_slicer, generate_braingrid_structures_labels_for_slicer


class NeuroRadiologicalVolumeStatistics:
    _dimension_x = None
    _dimension_y = None
    _dimension_z = None
    _spacing_x = None
    _spacing_y = None
    _spacing_z = None

    def __init__(self, dim_x: int = None, dim_y: int = None, dim_z: int = None, spac_x: float = None,
                 spac_y: float = None, spac_z: float = None):
        self._dimension_x = dim_x
        self._dimension_y = dim_y
        self._dimension_z = dim_z
        self._spacing_x = spac_x
        self._spacing_y = spac_y
        self._spacing_z = spac_z

    @property
    def dimension_x(self) -> int:
        return self._dimension_x

    @dimension_x.setter
    def dimension_x(self, value: int) -> None:
        self._dimension_x = value

    @property
    def dimension_y(self) -> int:
        return self._dimension_y

    @dimension_y.setter
    def dimension_y(self, value: int) -> None:
        self._dimension_y = value

    @property
    def dimension_z(self) -> int:
        return self._dimension_z

    @dimension_z.setter
    def dimension_z(self, value: int) -> None:
        self._dimension_z = value

    @property
    def spacing_x(self) -> float:
        return self._spacing_x

    @spacing_x.setter
    def spacing_x(self, value: float) -> None:
        self._spacing_x = value

    @property
    def spacing_y(self) -> float:
        return self._spacing_y

    @spacing_y.setter
    def spacing_y(self, value: float) -> None:
        self._spacing_y = value

    @property
    def spacing_z(self) -> float:
        return self._spacing_z

    @spacing_z.setter
    def spacing_z(self, value: float) -> None:
        self._spacing_z = value


class NeuroAcquisitionInfo:
    _t1c_stats = None
    _t1w_stats = None
    _t2f_stats = None
    _t2w_stats = None

    def __init__(self, t1c_stats: NeuroRadiologicalVolumeStatistics = None,
                 t1w_stats: NeuroRadiologicalVolumeStatistics = None,
                 t2f_stats: NeuroRadiologicalVolumeStatistics = None,
                 t2w_stats: NeuroRadiologicalVolumeStatistics = None):
        self._t1c_stats = t1c_stats
        self._t1w_stats = t1w_stats
        self._t2f_stats = t2f_stats
        self._t2w_stats = t2w_stats

    @property
    def t1c_stats(self) -> NeuroRadiologicalVolumeStatistics:
        return self._t1c_stats

    @t1c_stats.setter
    def t1c_stats(self, value: NeuroRadiologicalVolumeStatistics) -> None:
        self._t1c_stats = value

    @property
    def t1w_stats(self) -> NeuroRadiologicalVolumeStatistics:
        return self._t1w_stats

    @t1w_stats.setter
    def t1w_stats(self, value: NeuroRadiologicalVolumeStatistics) -> None:
        self._t1w_stats = value

    @property
    def t2f_stats(self) -> NeuroRadiologicalVolumeStatistics:
        return self._t2f_stats

    @t2f_stats.setter
    def t2f_stats(self, value: NeuroRadiologicalVolumeStatistics) -> None:
        self._t2f_stats = value

    @property
    def t2w_stats(self) -> NeuroRadiologicalVolumeStatistics:
        return self._t2w_stats

    @t2w_stats.setter
    def t2w_stats(self, value: NeuroRadiologicalVolumeStatistics) -> None:
        self._t2w_stats = value


class NeuroVolumeStatistics:
    _volume = -1.
    _brain_percentage = -1.

    def __init__(self, volume: float = -1., brain_percentage: float = -1.):
        self._volume = volume
        self._brain_percentage = brain_percentage

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = value

    @property
    def brain_percentage(self) -> float:
        return self._brain_percentage

    @brain_percentage.setter
    def brain_percentage(self, value: float) -> None:
        self._brain_percentage = value

class NeuroDiameterStatistics:
    _long_axis_diameter = -1.
    _short_axis_diameter = -1.
    _feret_diameter = -1.
    _equivalent_diameter_area = -1.

    def __init__(self, long_axis_diameter: float = -1., short_axis_diameter: float = -1.,
                 feret_diameter: float = -1., equivalent_diameter_area: float = -1.):
        self._long_axis_diameter = long_axis_diameter
        self._short_axis_diameter = short_axis_diameter
        self._feret_diameter = feret_diameter
        self._equivalent_diameter_area = equivalent_diameter_area

    @property
    def long_axis_diameter(self) -> float:
        return self._long_axis_diameter

    @long_axis_diameter.setter
    def long_axis_diameter(self, value: float) -> None:
        self._long_axis_diameter = value

    @property
    def short_axis_diameter(self) -> float:
        return self._short_axis_diameter

    @short_axis_diameter.setter
    def short_axis_diameter(self, value: float) -> None:
        self._short_axis_diameter = value

    @property
    def feret_diameter(self) -> float:
        return self._feret_diameter

    @feret_diameter.setter
    def feret_diameter(self, value: float) -> None:
        self._feret_diameter = value

    @property
    def equivalent_diameter_area(self) -> float:
        return self._equivalent_diameter_area

    @equivalent_diameter_area.setter
    def equivalent_diameter_area(self, value: float) -> None:
        self._equivalent_diameter_area = value


class NeuroMultifocalityStatistics:
    _multifocality = None
    _nb_parts = -1
    _max_distance = -1.
    def __init__(self, status: bool = None, parts: int = -1, distance: float = -1.):
        self._multifocality = status
        self._nb_parts = parts
        self._max_distance = distance

    @property
    def multifocality(self) -> bool:
        return self._multifocality

    @multifocality.setter
    def multifocality(self, value: bool) -> None:
        self._multifocality = value

    @property
    def nb_parts(self) -> int:
        return self._nb_parts

    @nb_parts.setter
    def nb_parts(self, value: int) -> None:
        self._nb_parts = value

    @property
    def max_distance(self) -> float:
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value: float) -> None:
        self._max_distance = value

class NeuroLocationStatistics:
    _left_laterality_percentage = -1.
    _right_laterality_percentage = -1.
    _laterality_midline_crossing = None

    def __init__(self, left: float = -1., right: float = -1., crossing: bool = None):
        self._left_laterality_percentage = left
        self._right_laterality_percentage = right
        self._laterality_midline_crossing = crossing

    @property
    def left_laterality_percentage(self) -> float:
        return self._left_laterality_percentage

    @left_laterality_percentage.setter
    def left_laterality_percentage(self, value: float) -> None:
        self._left_laterality_percentage = value

    @property
    def right_laterality_percentage(self) -> float:
        return self._right_laterality_percentage

    @right_laterality_percentage.setter
    def right_laterality_percentage(self, value: float) -> None:
        self._right_laterality_percentage = value

    @property
    def laterality_midline_crossing(self) -> bool:
        return self._laterality_midline_crossing

    @laterality_midline_crossing.setter
    def laterality_midline_crossing(self, value: bool) -> None:
        self._laterality_midline_crossing = value


class NeuroResectabilityStatistics:
    _expected_resectable_tumor_volume = -1.
    _expected_residual_tumor_volume = -1.
    _resectability_index = -1.

    def __init__(self, resectable: float = -1., residual: float = -1., index: float = -1.):
        self._expected_resectable_tumor_volume = resectable
        self._expected_residual_tumor_volume = residual
        self._resectability_index = index

    @property
    def expected_resectable_tumor_volume(self) -> float:
        return self._expected_resectable_tumor_volume

    @expected_resectable_tumor_volume.setter
    def expected_resectable_tumor_volume(self, value: float) -> None:
        self._expected_resectable_tumor_volume = value

    @property
    def expected_residual_tumor_volume(self) -> float:
        return self._expected_residual_tumor_volume

    @expected_residual_tumor_volume.setter
    def expected_residual_tumor_volume(self, value: float) -> None:
        self._expected_residual_tumor_volume = value

    @property
    def resectability_index(self) -> float:
        return self._resectability_index

    @resectability_index.setter
    def resectability_index(self, value: float) -> None:
        self._resectability_index = value


class NeuroCorticalStatistics:
    _cortical_structures_overlap = None
    _cortical_structures_distance = None

    def __init__(self, overlap: {} = None, distance: {} = None):
        self._cortical_structures_overlap = overlap
        self._cortical_structures_distance = distance

    @property
    def cortical_structures_overlap(self) -> {}:
        return self._cortical_structures_overlap

    @cortical_structures_overlap.setter
    def cortical_structures_overlap(self, value: {}) -> None:
        self._cortical_structures_overlap = value

    @property
    def cortical_structures_distance(self) -> {}:
        return self._cortical_structures_distance

    @cortical_structures_distance.setter
    def cortical_structures_distance(self, value: {}) -> None:
        self._cortical_structures_distance = value


class NeuroSubCorticalStatistics:
    _subcortical_structures_overlap = None
    _subcortical_structures_distance = None

    def __init__(self, overlap: {} = None, distance: {} = None):
        self._subcortical_structures_overlap = overlap
        self._subcortical_structures_distance = distance

    @property
    def subcortical_structures_overlap(self) -> {}:
        return self._subcortical_structures_overlap

    @subcortical_structures_overlap.setter
    def subcortical_structures_overlap(self, value: {}) -> None:
        self._subcortical_structures_overlap = value

    @property
    def subcortical_structures_distance(self) -> {}:
        return self._subcortical_structures_distance

    @subcortical_structures_distance.setter
    def subcortical_structures_distance(self, value: {}) -> None:
        self._subcortical_structures_distance = value

class NeuroInfiltrationStatistics:
    _overlap = None
    _count = None

    def __init__(self, overlap: {} = None, count: int = None):
        self._overlap = overlap
        self._count = count

    @property
    def overlap(self) -> {}:
        return self._overlap

    @overlap.setter
    def overlap(self, value: {}) -> None:
        self._overlap = value

    @property
    def count(self) -> {}:
        return self._count

    @count.setter
    def count(self, value: {}) -> None:
        self._count = value


class NeuroStructureStatistics:
    """
    Specific class for holding the computed characteristics/features.
    """
    _volume = None
    _diameters = None
    _multifocality = None
    _location = None
    _resectability = None
    _cortical = None
    _subcortical = None
    _infiltration = None

    def __init__(self):
        self._volume = NeuroVolumeStatistics()
        self._diameters = NeuroDiameterStatistics()
        self._multifocality = NeuroMultifocalityStatistics()
        self._location = NeuroLocationStatistics()
        self._resectability = NeuroResectabilityStatistics()
        self._cortical = {}
        self._subcortical = {}
        self._infiltration = {}

    @property
    def volume(self) -> NeuroVolumeStatistics:
        return self._volume

    @volume.setter
    def volume(self, value: NeuroVolumeStatistics) -> None:
        self._volume = value

    @property
    def diameters(self) -> NeuroDiameterStatistics:
        return self._diameters

    @diameters.setter
    def diameters(self, value: NeuroDiameterStatistics) -> None:
        self._diameters = value

    @property
    def multifocality(self) -> NeuroLocationStatistics:
        return self._multifocality

    @multifocality.setter
    def multifocality(self, value: NeuroMultifocalityStatistics) -> None:
        self._multifocality = value

    @property
    def location(self) -> NeuroLocationStatistics:
        return self._location

    @location.setter
    def location(self, value: NeuroLocationStatistics) -> None:
        self._location = value

    @property
    def resectability(self) -> NeuroResectabilityStatistics:
        return self._resectability

    @resectability.setter
    def resectability(self, value: NeuroResectabilityStatistics) -> None:
        self._resectability = value

    @property
    def cortical(self) -> Dict[str, NeuroCorticalStatistics]:
        return self._cortical

    @cortical.setter
    def cortical(self, value: Dict[str, NeuroCorticalStatistics]) -> None:
        self._cortical = value

    @property
    def subcortical(self) -> Dict[str, NeuroSubCorticalStatistics]:
        return self._subcortical

    @subcortical.setter
    def subcortical(self, value: Dict[str, NeuroSubCorticalStatistics]) -> None:
        self._subcortical = value

    @property
    def infiltration(self) -> Dict[str, NeuroInfiltrationStatistics]:
        return self._infiltration

    @infiltration.setter
    def infiltration(self, value: Dict[str, NeuroInfiltrationStatistics]) -> None:
        self._infiltration = value

class NeuroReportingStructure:
    """
    Reporting at a single timestamp with characteristics/features for the different structures.
    """
    _unique_id = None  # Internal unique identifier for the report
    _timestamp = None  # Timestamp attached to the report
    _output_folder = None
    _tumor_type = None  # Type of CNS tumor identified
    _acquisition_infos = None
    _statistics = {}

    def __init__(self, id: str, output_folder: str, timestamp: str):
        """
        """
        self.__reset()
        self._unique_id = id
        self._output_folder = output_folder
        self._timestamp = timestamp

    def __reset(self):
        """
        All objects share class or static variables.
        An instance or non-static variables are different for different objects (every object has a copy).
        """
        self._unique_id = None
        self._output_folder = None
        self._tumor_type = None
        self._statistics = {}
        self._acquisition_infos = None

    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @property
    def statistics(self) -> dict:
        return self._statistics

    @statistics.setter
    def statistics(self, value: dict) -> None:
        self._statistics = value

    def include_statistics(self, structure: str, space: str, statistics: NeuroStructureStatistics):
        if structure not in list(self.statistics.keys()):
            self.statistics[structure] = {}
        self.statistics[structure][space] = statistics

    @property
    def timestamp(self) -> str:
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: str) -> None:
        self._timestamp = value

    @property
    def tumor_type(self) -> str:
        return self._tumor_type

    @tumor_type.setter
    def tumor_type(self, value: str) -> None:
        self._tumor_type = value

    @property
    def acquisition_infos(self) -> NeuroAcquisitionInfo:
        return self._acquisition_infos

    @acquisition_infos.setter
    def acquisition_infos(self, value: NeuroAcquisitionInfo) -> None:
        self._acquisition_infos = value

    def to_disk(self) -> None:
        self.to_txt()
        self.to_csv()
        self.to_json()
        self.dump_descriptions()
        self.dump_acquisition_infos()

    def to_txt(self) -> None:
        """

        Exporting the computed standardized characteristics report in filename.

        Parameters
        ----------

        Returns
        -------
        None
        """
        try:
            filename = os.path.join(self._output_folder, "reporting",
                                    "T" + str(self.timestamp), "neuro_clinical_report.txt")
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            logging.info(f"Exporting standardized report for timestamp timestamp {self.timestamp} to text in {filename}.")
            pfile = open(filename, 'w')
            pfile.write('########### Raidionics standardized report for timestamp {} in MNI space ###########\n'.format(self.timestamp))
            pfile.write('Tumor type: {}\n'.format(self.tumor_type))

            for s in self.statistics.keys():
                pfile.write(f'\n Features for {s} category.\n')
                pfile.write(' Volumes: \n')
                pfile.write(f'  * Volume: {round(self.statistics[s]["MNI"].volume.volume, 2)} (ml)\n')
                pfile.write(f'  * Brain percentage: {round(self.statistics[s]["MNI"].volume.brain_percentage * 100., 2)} (%)\n')

                pfile.write(' Diameters: \n')
                pfile.write(f'  * Long-axis diameter: {round(self.statistics[s]["MNI"].diameters.long_axis_diameter, 2)} (mm)\n')
                pfile.write(f'  * Short-axis diameter: {round(self.statistics[s]["MNI"].diameters.short_axis_diameter, 2)} (mm)\n')
                pfile.write(f'  * Feret diameter maximum: {round(self.statistics[s]["MNI"].diameters.feret_diameter, 2)} (mm)\n')
                pfile.write(f'  * Equivalent diameter area: {round(self.statistics[s]["MNI"].diameters.equivalent_diameter_area, 2)} (mm)\n\n')

                pfile.write(' Multifocality: \n')
                pfile.write(f'  * Status: {self.statistics[s]["MNI"].multifocality.multifocality}\n')
                pfile.write(f'  * Number parts: {self.statistics[s]["MNI"].multifocality.nb_parts}\n')
                pfile.write(f'  * Largest distance between components: {round(self.statistics[s]["MNI"].multifocality.max_distance, 2)} (mm)\n\n')

                pfile.write(' Location: \n')
                pfile.write(f'  * Left hemisphere: {self.statistics[s]["MNI"].location.left_laterality_percentage}\n')
                pfile.write(f'  * Right hemisphere: {self.statistics[s]["MNI"].location.right_laterality_percentage}\n')
                pfile.write(f'  * Midline crossing: {self.statistics[s]["MNI"].location.laterality_midline_crossing}\n\n')

                # @TODO. Should have an if tumor type is GBM
                if "Tumor" in s:
                    pfile.write(' Resectability: \n\n')
                    pfile.write(f'  * Index: {self.statistics[s]["MNI"].resectability.resectability_index}\n')
                    pfile.write(f'  * Resectable volume: {self.statistics[s]["MNI"].resectability.expected_resectable_tumor_volume} ml\n')
                    pfile.write(f'  * Residual volume: {self.statistics[s]["MNI"].resectability.expected_residual_tumor_volume} ml\n\n')

                if len(ResourcesConfiguration.getInstance().neuro_features_cortical_structures) != 0:
                    pfile.write(' Cortical structures profile\n')
                    for t in self.statistics[s]["MNI"].cortical.keys():
                        pfile.write('  * {} atlas\n'.format(t))
                        structures_ordered = collections.OrderedDict(
                            sorted(self.statistics[s]["MNI"].cortical[t].cortical_structures_overlap.items(),
                                   key=operator.itemgetter(1), reverse=True))
                        for r in structures_ordered.keys():
                            if structures_ordered[r] != 0:
                                struct_name = ' '.join(r.lower().replace('main', '').split('_')[:])
                                pfile.write('    - {}: {}%\n'.format(struct_name, structures_ordered[r]))

                if len(ResourcesConfiguration.getInstance().neuro_features_subcortical_structures) != 0:
                    pfile.write('\n Subcortical structures profile\n')
                    for t in self.statistics[s]["MNI"].subcortical.keys():
                        pfile.write('  * {} atlas\n'.format(t))
                        tracts_ordered = collections.OrderedDict(
                            sorted(self.statistics[s]["MNI"].subcortical[t].subcortical_structures_overlap.items(),
                                   key=operator.itemgetter(1), reverse=True))
                        for r in tracts_ordered.keys():
                            if tracts_ordered[r] != 0:
                                tract_name = ' '.join(
                                    r.lower().replace('main', '').replace('mni', '').split('.')[0].split('_'))
                                pfile.write('    - {}: {}% overlap\n'.format(tract_name, np.round(tracts_ordered[r], 2)))

                        pfile.write('\n')
                        tracts_ordered = collections.OrderedDict(
                            sorted(self.statistics[s]["MNI"].subcortical[t].subcortical_structures_distance.items(),
                                   key=operator.itemgetter(1), reverse=False))
                        for r in tracts_ordered.keys():
                            if tracts_ordered[r] != -1.:
                                tract_name = ' '.join(
                                    r.lower().replace('main', '').replace('mni', '').split('.')[0].split('_'))
                                pfile.write('    - {}: {}mm away\n'.format(tract_name, np.round(tracts_ordered[r], 2)))

                if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                    pfile.write('\n Infiltration profile\n')
                    for t in self.statistics[s]["MNI"].infiltration.keys():
                        pfile.write('  * {} atlas\n'.format(t))
                        pfile.write('  Total infiltrated regions: {}\n'.format(self.statistics[s]["MNI"].infiltration[t].count))
                        voxels_ordered = collections.OrderedDict(
                            sorted(self.statistics[s]["MNI"].infiltration[t].overlap.items(),
                                   key=operator.itemgetter(1), reverse=True))
                        for r in voxels_ordered.keys():
                            if voxels_ordered[r] != 0:
                                voxel_name = ' '.join(r.lower().replace('main', '').split('_')[:])
                                pfile.write('    - {}: {}%\n'.format(voxel_name, voxels_ordered[r]))
            pfile.close()
        except Exception as e:
            raise RuntimeError(f"Exporting standardized report on disk as text failed with {e}")
        return

    def to_json(self) -> None:
        try:
            filename = os.path.join(self._output_folder, "reporting",
                                    "T" + str(self.timestamp), "neuro_clinical_report.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            logging.info(f"Exporting standardized report for timestamp timestamp {self.timestamp} to json in {filename}.")
            param_json = {}
            for i, s in enumerate(self.statistics.keys()):
                param_json[s] = {}
                param_json[s]["Patient"] = {}
                param_json[s]["Patient"]["Type"] = str(self.tumor_type)
                param_json[s]["Patient"]["Volume (ml)"] = float(self.statistics[s]["Patient"].volume.volume)

                param_json[s]["MNI"] = {}
                param_json[s]["MNI"]["Volume (ml)"] = float(self.statistics[s]["MNI"].volume.volume)
                param_json[s]["MNI"]["Brain percentage (%)"] = float(self.statistics[s]["MNI"].volume.brain_percentage)

                param_json[s]["MNI"]["Diameters"] = {}
                param_json[s]["MNI"]["Diameters"]["Long-axis diameter (mm)"] = float(self.statistics[s]["MNI"].diameters.long_axis_diameter)
                param_json[s]["MNI"]["Diameters"]["Short-axis diameter (mm)"] = float(self.statistics[s]["MNI"].diameters.short_axis_diameter)
                param_json[s]["MNI"]["Diameters"]["Feret diameter (mm)"] = float(self.statistics[s]["MNI"].diameters.feret_diameter)
                param_json[s]["MNI"]["Diameters"]["Equivalent diameter area (mm)"] = float(self.statistics[s]["MNI"].diameters.equivalent_diameter_area)

                param_json[s]["MNI"]["Multifocality"] = {}
                param_json[s]["MNI"]["Multifocality"]["Status"] = self.statistics[s]["MNI"].multifocality.multifocality
                param_json[s]["MNI"]["Multifocality"]["Elements"] = self.statistics[s]["MNI"].multifocality.nb_parts
                param_json[s]["MNI"]["Multifocality"]["Max distance (mm)"] = float(self.statistics[s]["MNI"].multifocality.max_distance)

                param_json[s]["MNI"]["Location"] = {}
                param_json[s]["MNI"]["Location"]["Left laterality (%)"] = float(self.statistics[s]["MNI"].location.left_laterality_percentage)
                param_json[s]["MNI"]["Location"]["Right laterality (%)"] = float(self.statistics[s]["MNI"].location.right_laterality_percentage)
                param_json[s]["MNI"]["Location"]["Midline crossing"] = self.statistics[s]["MNI"].location.laterality_midline_crossing


                # @TODO. Should be only for glioblastoma, but no tumor type classification yet
                if "Tumor" in s:
                    param_json[s]["MNI"]["Resectability"] = {}
                    param_json[s]["MNI"]["Resectability"]["Index"] = self.statistics[s]["MNI"].resectability.resectability_index
                    param_json[s]["MNI"]["Resectability"]["Resectable volume (ml)"] = float(self.statistics[s]["MNI"].resectability.expected_resectable_tumor_volume)
                    param_json[s]["MNI"]["Resectability"]["Expected residual volume (ml)"] = float(self.statistics[s]["MNI"].resectability.expected_residual_tumor_volume)

                param_json[s]["MNI"]["Cortical Profile"] = {}
                for t in self.statistics[s]["MNI"].cortical.keys():
                    param_json[s]["MNI"]["Cortical Profile"][t] = {}
                    param_json[s]["MNI"]["Cortical Profile"][t]["Overlap"] = {}
                    for r in self.statistics[s]["MNI"].cortical[t].cortical_structures_overlap.keys():
                        param_json[s]["MNI"]["Cortical Profile"][t]["Overlap"][r] = float(self.statistics[s]["MNI"].cortical[t].cortical_structures_overlap[r])

                param_json[s]["MNI"]["Subcortical Profile"] = {}
                for t in self.statistics[s]["MNI"].subcortical.keys():
                    param_json[s]["MNI"]["Subcortical Profile"][t] = {}
                    param_json[s]["MNI"]["Subcortical Profile"][t]["Overlap"] = {}
                    for r in self.statistics[s]["MNI"].subcortical[t].subcortical_structures_overlap.keys():
                        param_json[s]["MNI"]["Subcortical Profile"][t]["Overlap"][r] = float(self.statistics[s]["MNI"].subcortical[t].subcortical_structures_overlap[r])
                    param_json[s]["MNI"]["Subcortical Profile"][t]["Distance"] = {}
                    for r in self.statistics[s]["MNI"].subcortical[t].subcortical_structures_distance.keys():
                        param_json[s]["MNI"]["Subcortical Profile"][t]["Distance"][r] = float(self.statistics[s]["MNI"].subcortical[t].subcortical_structures_distance[r])

                if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                    param_json[s]["MNI"]["Infiltration"] = {}
                    for t in self.statistics[s]["MNI"].infiltration.keys():
                        param_json[s]["MNI"]["Infiltration"][t] = {}
                        param_json[s]["MNI"]["Infiltration"][t]["Count"] = self.statistics[s]["MNI"].infiltration[t].count
                        param_json[s]["MNI"]["Infiltration"][t]["Overlap"] = {}
                        for r in self.statistics[s]["MNI"].infiltration[t].overlap.keys():
                            param_json[s]["MNI"]["Infiltration"][t]["Overlap"][r] = float(self.statistics[s]["MNI"].infiltration[t].overlap[r])

            with open(filename, 'w', newline='\n') as outfile:
                json.dump(param_json, outfile, indent=4, sort_keys=True)
        except Exception as e:
            raise RuntimeError(f"Standardized report dump on disk for T{self.timestamp} as json failed with {e}.\n")

        return

    def to_csv(self) -> None:
        """
        Exporting the standardized report to a csv file on disk.
        """
        try:
            filename = os.path.join(self._output_folder, "reporting",
                                    "T" + str(self.timestamp), "neuro_clinical_report.csv")
            logging.info(f"Exporting standardized report for T{self.timestamp} to csv in {filename}.")
            column_names = []
            all_values = []
            for i, s in enumerate(self.statistics.keys()):
                structure_values = []
                if i == 0:
                    column_names.extend(["Structure", "Type"])
                structure_values.extend([s, str(self.tumor_type)])
                if i == 0:
                    column_names.extend(["Volume patient space (ml)"])
                structure_values.extend([self.statistics[s]["Patient"].volume.volume])

                if i == 0:
                    column_names.extend(["Volume MNI space (ml)", "Brain percentage (%)"])
                structure_values.extend([self.statistics[s]["MNI"].volume.volume,
                                         self.statistics[s]["MNI"].volume.brain_percentage])

                if i == 0:
                    column_names.extend(["Long-axis diameter (mm)", "Short-axis diameter (mm)",
                                         "Feret diameter maximum (mm)", "Equivalent diameter area (mm)"])
                structure_values.extend([self.statistics[s]["MNI"].diameters.long_axis_diameter,
                                         self.statistics[s]["MNI"].diameters.short_axis_diameter,
                                         self.statistics[s]["MNI"].diameters.feret_diameter,
                                         self.statistics[s]["MNI"].diameters.equivalent_diameter_area])

                if i == 0:
                    column_names.extend(["Multifocality", "Number parts", "Multifocal distance (mm)"])
                structure_values.extend([self.statistics[s]["MNI"].multifocality.multifocality,
                                         self.statistics[s]["MNI"].multifocality.nb_parts,
                                         self.statistics[s]["MNI"].multifocality.max_distance])

                if i == 0:
                    column_names.extend(['Left laterality (%)', 'Right laterality (%)', 'Midline crossing'])
                structure_values.extend([self.statistics[s]["MNI"].location.left_laterality_percentage,
                                         self.statistics[s]["MNI"].location.right_laterality_percentage,
                                         self.statistics[s]["MNI"].location.laterality_midline_crossing])

                # @TODO. Should be only for glioblastoma, but no tumor type classification yet
                if "Tumor" in s:
                    if i == 0:
                        column_names.extend(['ResectionIndex', 'ExpectedResectableVolume (ml)',
                                             'ExpectedResidualVolume (ml)'])
                    structure_values.extend([self.statistics[s]["MNI"].resectability.resectability_index,
                                             self.statistics[s]["MNI"].resectability.expected_resectable_tumor_volume,
                                             self.statistics[s]["MNI"].resectability.expected_residual_tumor_volume])

                for t in self.statistics[s]["MNI"].cortical.keys():
                    for r in self.statistics[s]["MNI"].cortical[t].cortical_structures_overlap.keys():
                        if i == 0:
                            column_names.extend([t + '_' + r.split('.')[0].lower().strip() + '_overlap'])
                        structure_values.extend([self.statistics[s]["MNI"].cortical[t].cortical_structures_overlap[r]])

                for t in self.statistics[s]["MNI"].subcortical.keys():
                    for r in self.statistics[s]["MNI"].subcortical[t].subcortical_structures_overlap.keys():
                        if i == 0:
                            if t == "MNI":
                                column_names.extend([t + '_' + r.split('.')[0][:-4] + '_overlap'])
                            else:
                                column_names.extend([t + '_' + r.split('.')[0] + '_overlap'])
                        structure_values.extend([self.statistics[s]["MNI"].subcortical[t].subcortical_structures_overlap[r]])
                    for r in self.statistics[s]["MNI"].subcortical[t].subcortical_structures_distance.keys():
                        if i == 0:
                            if t == "MNI":
                                column_names.extend([t + '_' + r.split('.')[0][:-4] + '_distance'])
                            else:
                                column_names.extend([t + '_' + r.split('.')[0] + '_distance'])
                        structure_values.extend([self.statistics[s]["MNI"].subcortical[t].subcortical_structures_distance[r]])

                if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                    for t in self.statistics[s]["MNI"].infiltration.keys():
                        if i == 0:
                            column_names.extend([f'{t} - Infiltration count'])
                        structure_values.extend([self.statistics[s]["MNI"].infiltration[t].count])
                        for r in self.statistics[s]["MNI"].infiltration[t].overlap.keys():
                            if i == 0:
                                column_names.extend([t + '_' + r.split('.')[0] + '_overlap'])
                            structure_values.extend([self.statistics[s]["MNI"].infiltration[t].overlap[r]])
                all_values.append(structure_values)

            values_df = pd.DataFrame(np.asarray(all_values), columns=column_names)
            values_df.to_csv(filename, index=False)
        except Exception as e:
            raise RuntimeError(f"Standardized report dump on disk for T{self.timestamp} as csv failed with {e}")

    def dump_descriptions(self) -> None:
        """
        Text files with a readable and comprehension description of the different labels in each atlas file are saved
        on disk for later user (i.e., manual inspection or re-use in Raidionics).
        In addition, and for viewing purposes in Raidionics, the actual atlas annotation files also need to be saved
        on disk again.
        """
        try:
            atlas_desc_dir = os.path.join(self._output_folder, 'atlas_descriptions')
            os.makedirs(atlas_desc_dir, exist_ok=True)
            atlases = ResourcesConfiguration.getInstance().neuro_features_cortical_structures
            for a in atlases:
                df = generate_cortical_structures_labels_for_slicer(atlas_name=a)
                output_filename = os.path.join(atlas_desc_dir, a + '_description.csv')
                df.to_csv(output_filename)
                shutil.copyfile(src=ResourcesConfiguration.getInstance().cortical_structures['MNI'][a]['Mask'],
                                dst=os.path.join(atlas_desc_dir, 'MNI_' + a + '_structures.nii.gz'))
            atlases = ResourcesConfiguration.getInstance().neuro_features_subcortical_structures
            for a in atlases:
                df = generate_subcortical_structures_labels_for_slicer(atlas_name=a)
                output_filename = os.path.join(atlas_desc_dir, a + '_description.csv')
                df.to_csv(output_filename)
                shutil.copyfile(src=ResourcesConfiguration.getInstance().subcortical_structures['MNI'][a]['Mask'],
                                dst=os.path.join(atlas_desc_dir, 'MNI_' + a + '_structures.nii.gz'))
            atlases = ResourcesConfiguration.getInstance().neuro_features_braingrid
            for a in atlases:
                df = generate_braingrid_structures_labels_for_slicer(atlas_name=a)
                output_filename = os.path.join(atlas_desc_dir, a + '_description.csv')
                df.to_csv(output_filename)
                shutil.copyfile(src=ResourcesConfiguration.getInstance().braingrid_structures['MNI'][a]['Mask'],
                                dst=os.path.join(atlas_desc_dir, 'MNI_' + a + '_structures.nii.gz'))
        except Exception as e:
            raise RuntimeError(f"Neuro-parameters atlas descriptions dump failed with {e}")

    def dump_acquisition_infos(self):
        try:
            filename = os.path.join(self._output_folder, "reporting",
                                    "T" + str(self.timestamp), "radiological_acquisition_infos.csv")
            column_names = ["Sequence", "DimX", "DimY", "DimZ", "SpacX", "SpacY", "SpacZ"]

            t1c_values = ["t1c", None, None, None, None, None, None]
            t1w_values = ["t1w", None, None, None, None, None, None]
            t2f_values = ["t2f", None, None, None, None, None, None]
            t2w_values = ["t2w", None, None, None, None, None, None]
            if self.acquisition_infos.t1c_stats is not None:
                t1c_values[1] = self.acquisition_infos.t1c_stats.dimension_x
                t1c_values[2] = self.acquisition_infos.t1c_stats.dimension_y
                t1c_values[3] = self.acquisition_infos.t1c_stats.dimension_z
                t1c_values[4] = self.acquisition_infos.t1c_stats.spacing_x
                t1c_values[5] = self.acquisition_infos.t1c_stats.spacing_y
                t1c_values[6] = self.acquisition_infos.t1c_stats.spacing_z
            if self.acquisition_infos.t1w_stats is not None:
                t1w_values[1] = self.acquisition_infos.t1w_stats.dimension_x
                t1w_values[2] = self.acquisition_infos.t1w_stats.dimension_y
                t1w_values[3] = self.acquisition_infos.t1w_stats.dimension_z
                t1w_values[4] = self.acquisition_infos.t1w_stats.spacing_x
                t1w_values[5] = self.acquisition_infos.t1w_stats.spacing_y
                t1w_values[6] = self.acquisition_infos.t1w_stats.spacing_z
            if self.acquisition_infos.t2f_stats is not None:
                t2f_values[1] = self.acquisition_infos.t2f_stats.dimension_x
                t2f_values[2] = self.acquisition_infos.t2f_stats.dimension_y
                t2f_values[3] = self.acquisition_infos.t2f_stats.dimension_z
                t2f_values[4] = self.acquisition_infos.t2f_stats.spacing_x
                t2f_values[5] = self.acquisition_infos.t2f_stats.spacing_y
                t2f_values[6] = self.acquisition_infos.t2f_stats.spacing_z
            if self.acquisition_infos.t2w_stats is not None:
                t2w_values[1] = self.acquisition_infos.t2w_stats.dimension_x
                t2w_values[2] = self.acquisition_infos.t2w_stats.dimension_y
                t2w_values[3] = self.acquisition_infos.t2w_stats.dimension_z
                t2w_values[4] = self.acquisition_infos.t2w_stats.spacing_x
                t2w_values[5] = self.acquisition_infos.t2w_stats.spacing_y
                t2w_values[6] = self.acquisition_infos.t2w_stats.spacing_z
            all_values = [t1c_values, t1w_values, t2f_values, t2w_values]
            values_df = pd.DataFrame(np.asarray(all_values), columns=column_names)
            values_df.to_csv(filename, index=False)
        except Exception as e:
            logging.error(f"Dumping acquisition infos on disk failed with: {e}")