import logging
import shutil
import traceback
import os
import numpy as np
import operator
import json
import pandas as pd
from typing import Dict, Tuple, List, TextIO
import collections
from ..configuration_parser import ResourcesConfiguration
from ..DataStructures.StatisticsStructure import VolumeStatistics, DiameterStatistics, MultifocalityStatistics, RadiologicalVolumeStatistics, AtlasOverlapStatistics
from ..io import generate_cortical_structures_labels_for_slicer, generate_subcortical_structures_labels_for_slicer, generate_braingrid_structures_labels_for_slicer


class NeuroAcquisitionInfo:
    _t1c_stats = None
    _t1w_stats = None
    _t2f_stats = None
    _t2w_stats = None

    def __init__(self, t1c_stats: RadiologicalVolumeStatistics = None,
                 t1w_stats: RadiologicalVolumeStatistics = None,
                 t2f_stats: RadiologicalVolumeStatistics = None,
                 t2w_stats: RadiologicalVolumeStatistics = None):
        self._t1c_stats = t1c_stats
        self._t1w_stats = t1w_stats
        self._t2f_stats = t2f_stats
        self._t2w_stats = t2w_stats

    @property
    def t1c_stats(self) -> RadiologicalVolumeStatistics:
        return self._t1c_stats

    @t1c_stats.setter
    def t1c_stats(self, value: RadiologicalVolumeStatistics) -> None:
        self._t1c_stats = value

    @property
    def t1w_stats(self) -> RadiologicalVolumeStatistics:
        return self._t1w_stats

    @t1w_stats.setter
    def t1w_stats(self, value: RadiologicalVolumeStatistics) -> None:
        self._t1w_stats = value

    @property
    def t2f_stats(self) -> RadiologicalVolumeStatistics:
        return self._t2f_stats

    @t2f_stats.setter
    def t2f_stats(self, value: RadiologicalVolumeStatistics) -> None:
        self._t2f_stats = value

    @property
    def t2w_stats(self) -> RadiologicalVolumeStatistics:
        return self._t2w_stats

    @t2w_stats.setter
    def t2w_stats(self, value: RadiologicalVolumeStatistics) -> None:
        self._t2w_stats = value

class NeuroLocationStatistics:
    _left_laterality_percentage = -1.
    _right_laterality_percentage = -1.
    _laterality_midline_crossing = None
    _depth_minimum = -1.
    _depth_maximum = -1.
    _depth_mean = -1.

    def __init__(self, left: float = -1., right: float = -1., crossing: bool = None,
                 depth_min: float = -1., depth_max: float = -1., depth_mean: float = -1.):
        self._left_laterality_percentage = left
        self._right_laterality_percentage = right
        self._laterality_midline_crossing = crossing
        self._depth_minimum = depth_min
        self._depth_maximum = depth_max
        self._depth_mean = depth_mean

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

    @property
    def depth_minimum(self) -> float:
        return self._depth_minimum

    @depth_minimum.setter
    def depth_minimum(self, value: float) -> None:
        self._depth_minimum = value

    @property
    def depth_maximum(self) -> float:
        return self._depth_maximum

    @depth_maximum.setter
    def depth_maximum(self, value: float) -> None:
        self._depth_maximum = value

    @property
    def depth_mean(self) -> float:
        return self._depth_mean

    @depth_mean.setter
    def depth_mean(self, value: float) -> None:
        self._depth_mean = value

    def to_dict(self) -> dict:
        key = "location"
        statistics = {key: {}}
        statistics[key]["left_laterality"] = self.left_laterality_percentage
        statistics[key]["right_laterality"] = self.right_laterality_percentage
        statistics[key]["midline_crossing"] = self.laterality_midline_crossing
        statistics[key]["depth_minimum_mm"] = self.depth_minimum
        statistics[key]["depth_mean_mm"] = self.depth_mean
        statistics[key]["depth_maximum_mm"] = self.depth_maximum

        return statistics

    def to_json(self, filename: str) -> None:
        with open(filename, 'w', newline='\n') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)

    def to_csv_inputs(self) -> Tuple[List[str], List]:
        column_names = ['Left laterality (%)', 'Right laterality (%)', 'Midline crossing',
                        'Depth minimum (mm)', 'Depth mean (mm)', 'Depth maximum (mm)']
        values = [self.left_laterality_percentage, self.right_laterality_percentage, self.laterality_midline_crossing,
                  self.depth_minimum, self.depth_mean, self.depth_maximum]
        return column_names, values

    def to_txt(self, pfile: TextIO) -> None:
        pfile.write(' Location: \n')
        pfile.write(f'  * Left hemisphere: {self.left_laterality_percentage}\n')
        pfile.write(f'  * Right hemisphere: {self.right_laterality_percentage}\n')
        pfile.write(f'  * Midline crossing: {self.laterality_midline_crossing}\n')
        pfile.write(f'  * Depth min: {self.depth_minimum} (mm)\n')
        pfile.write(f'  * Depth mean: {self.depth_mean} (mm)\n')
        pfile.write(f'  * Depth max: {self.depth_maximum} (mm)\n\n')
    
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

    def to_dict(self) -> dict:
        key = "resectability"
        statistics = {key: {}}
        statistics[key]["index"] = self.resectability_index
        statistics[key]["resectable_volume_ml"] = self.expected_resectable_tumor_volume
        statistics[key]["residual_volume_ml"] = self.expected_residual_tumor_volume

        return statistics

    def to_json(self, filename: str) -> None:
        with open(filename, 'w', newline='\n') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)

    def to_csv_inputs(self) -> Tuple[List[str], List]:
        column_names = ['Index', 'Resectable volume (ml)', 'Expected residual volume (ml)']
        values = [self.resectability_index, self.expected_resectable_tumor_volume, self.expected_residual_tumor_volume]
        return column_names, values

    def to_txt(self, pfile: TextIO) -> None:
        pfile.write(' Resectability: \n\n')
        pfile.write(f'  * Index: {self.resectability_index}\n')
        pfile.write(f'  * Resectable volume: {self.expected_resectable_tumor_volume} ml\n')
        pfile.write(f'  * Residual volume: {self.expected_residual_tumor_volume} ml\n\n')

class NeuroInfiltrationStatistics:
    _overlap = None
    _count = None

    def __init__(self, overlap: dict = None, count: int = None):
        self._overlap = overlap
        self._count = count

    @property
    def overlap(self) -> dict:
        return self._overlap

    @overlap.setter
    def overlap(self, value: dict) -> None:
        self._overlap = value

    @property
    def count(self) -> dict:
        return self._count

    @count.setter
    def count(self, value: dict) -> None:
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
        self._volume = VolumeStatistics()
        self._diameters = DiameterStatistics()
        self._multifocality = MultifocalityStatistics()
        self._location = NeuroLocationStatistics()
        self._resectability = NeuroResectabilityStatistics()
        self._cortical = {}
        self._subcortical = {}
        self._infiltration = {}

    @property
    def volume(self) -> VolumeStatistics:
        return self._volume

    @volume.setter
    def volume(self, value: VolumeStatistics) -> None:
        self._volume = value

    @property
    def diameters(self) -> DiameterStatistics:
        return self._diameters

    @diameters.setter
    def diameters(self, value: DiameterStatistics) -> None:
        self._diameters = value

    @property
    def multifocality(self) -> MultifocalityStatistics:
        return self._multifocality

    @multifocality.setter
    def multifocality(self, value: MultifocalityStatistics) -> None:
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
    def cortical(self) -> Dict[str, AtlasOverlapStatistics]:
        return self._cortical

    @cortical.setter
    def cortical(self, value: Dict[str, AtlasOverlapStatistics]) -> None:
        self._cortical = value

    @property
    def subcortical(self) -> Dict[str, AtlasOverlapStatistics]:
        return self._subcortical

    @subcortical.setter
    def subcortical(self, value: Dict[str, AtlasOverlapStatistics]) -> None:
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
        self.to_json()
        self.to_txt()
        self.to_csv()
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
            pfile.write('Tumor type: {}\n'.format(self.tumor_type))

            for s in self.statistics.keys():
                pfile.write(f'\n Features for {s} category.\n')
                for sp in self.statistics[s].keys():
                    pfile.write(f'########### Raidionics standardized report for timestamp {self.timestamp} in {sp} space ###########\n')
                    self.statistics[s][sp].volume.to_txt(pfile)
                    self.statistics[s][sp].diameters.to_txt(pfile)
                    self.statistics[s][sp].multifocality.to_txt(pfile)
                    self.statistics[s][sp].location.to_txt(pfile)

                    if "Tumor" in s: # @TODO. Should have an if tumor type is GBM
                        self.statistics[s][sp].resectability.to_txt(pfile)

                    if len(ResourcesConfiguration.getInstance().neuro_features_cortical_structures) != 0:
                        pfile.write(' Cortical structures profile\n')
                        for t in self.statistics[s][sp].cortical.keys():
                            self.statistics[s][sp].cortical[t].to_txt(pfile)

                    if len(ResourcesConfiguration.getInstance().neuro_features_subcortical_structures) != 0:
                        pfile.write('\n Subcortical structures profile\n')
                        for t in self.statistics[s][sp].subcortical.keys():
                            self.statistics[s][sp].subcortical[t].to_txt(pfile)
                            # pfile.write('  * {} atlas\n'.format(t))
                            # tracts_ordered = collections.OrderedDict(
                            #     sorted(self.statistics[s][sp].subcortical[t].overlap.items(),
                            #         key=operator.itemgetter(1), reverse=True))
                            # for r in tracts_ordered.keys():
                            #     if tracts_ordered[r] != 0:
                            #         tract_name = ' '.join(
                            #             r.lower().replace('main', '').replace('mni', '').split('.')[0].split('_'))
                            #         pfile.write('    - {}: {}% overlap\n'.format(tract_name, np.round(tracts_ordered[r], 2)))

                            # pfile.write('\n')
                            # tracts_ordered = collections.OrderedDict(
                            #     sorted(self.statistics[s][sp].subcortical[t].distance.items(),
                            #         key=operator.itemgetter(1), reverse=False))
                            # for r in tracts_ordered.keys():
                            #     if tracts_ordered[r] != -1.:
                            #         tract_name = ' '.join(
                            #             r.lower().replace('main', '').replace('mni', '').split('.')[0].split('_'))
                            #         pfile.write('    - {}: {}mm away\n'.format(tract_name, np.round(tracts_ordered[r], 2)))

                    if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                        pfile.write('\n Infiltration profile\n')
                        for t in self.statistics[s][sp].infiltration.keys():
                            pfile.write('  * {} atlas\n'.format(t))
                            pfile.write('  Total infiltrated regions: {}\n'.format(self.statistics[s][sp].infiltration[t].count))
                            voxels_ordered = collections.OrderedDict(
                                sorted(self.statistics[s][sp].infiltration[t].overlap.items(),
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
            for s in self.statistics.keys():
                param_json[s] = {}
                for sp in self.statistics[s].keys():
                    param_json[s][sp] = {}
                    if sp == "Patient":
                        param_json[s][sp]["Type"] = str(self.tumor_type)

                    param_json[s][sp] = {}
                    param_json[s][sp].update(self.statistics[s][sp].volume.to_dict())
                    param_json[s][sp].update(self.statistics[s][sp].diameters.to_dict())
                    param_json[s][sp].update(self.statistics[s][sp].multifocality.to_dict())
                    param_json[s][sp].update(self.statistics[s][sp].location.to_dict())

                    # @TODO. Should be only for glioblastoma, but no tumor type classification yet
                    if "Tumor" in s:
                        param_json[s][sp].update(self.statistics[s][sp].resectability.to_dict())

                    param_json[s][sp]["Cortical Profile"] = {}
                    for t in self.statistics[s][sp].cortical.keys():
                        param_json[s][sp]["Cortical Profile"][t] = {}
                        param_json[s][sp]["Cortical Profile"][t].update(self.statistics[s][sp].cortical[t].to_dict())

                    param_json[s][sp]["Subcortical Profile"] = {}
                    for t in self.statistics[s][sp].subcortical.keys():
                        param_json[s][sp]["Subcortical Profile"][t] = {}
                        param_json[s][sp]["Subcortical Profile"][t].update(self.statistics[s][sp].subcortical[t].to_dict())

                    if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                        param_json[s][sp]["Infiltration"] = {}
                        for t in self.statistics[s][sp].infiltration.keys():
                            param_json[s][sp]["Infiltration"][t] = {}
                            param_json[s][sp]["Infiltration"][t]["Count"] = self.statistics[s][sp].infiltration[t].count
                            param_json[s][sp]["Infiltration"][t]["Overlap"] = {}
                            for r in self.statistics[s][sp].infiltration[t].overlap.keys():
                                param_json[s][sp]["Infiltration"][t]["Overlap"][r] = float(self.statistics[s][sp].infiltration[t].overlap[r])

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
            i = 0
            for s in self.statistics.keys():
                for sp in self.statistics[s].keys():
                    structure_values = []
                    if i == 0:
                        column_names.extend(["Structure", "Type", "Space"])
                    structure_values.extend([s, str(self.tumor_type), sp])

                    volume_names, volume_values = self.statistics[s][sp].volume.to_csv_inputs()
                    if i == 0:
                        column_names.extend(volume_names)
                    structure_values.extend(volume_values)

                    diameters_names, diameters_values = self.statistics[s][sp].diameters.to_csv_inputs()
                    if i == 0:
                        column_names.extend(diameters_names)
                    structure_values.extend(diameters_values)

                    multifocality_names, multifocality_values = self.statistics[s][sp].multifocality.to_csv_inputs()
                    if i == 0:
                        column_names.extend(multifocality_names)
                    structure_values.extend(multifocality_values)

                    location_names, location_values = self.statistics[s][sp].location.to_csv_inputs()
                    if i == 0:
                        column_names.extend(location_names)
                    structure_values.extend(location_values)

                    resectability_names, resectability_values = self.statistics[s][sp].resectability.to_csv_inputs()
                    if i == 0:
                        column_names.extend(resectability_names)
                    structure_values.extend(resectability_values)

                    for t in self.statistics[s][sp].cortical.keys():
                        atlas_names, atlas_values = self.statistics[s][sp].cortical[t].to_csv_inputs()
                        if i == 0:
                            column_names.extend(atlas_names)
                        structure_values.extend(atlas_values)

                    for t in self.statistics[s][sp].subcortical.keys():
                        atlas_names, atlas_values = self.statistics[s][sp].subcortical[t].to_csv_inputs()
                        if i == 0:
                            column_names.extend(atlas_names)
                        structure_values.extend(atlas_values)

                    if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                        for t in self.statistics[s][sp].infiltration.keys():
                            if i == 0:
                                column_names.extend([f'{t} - Infiltration count'])
                            structure_values.extend([self.statistics[s][sp].infiltration[t].count])
                            for r in self.statistics[s][sp].infiltration[t].overlap.keys():
                                if i == 0:
                                    column_names.extend([t + '_' + r.split('.')[0] + '_overlap'])
                                structure_values.extend([self.statistics[s][sp].infiltration[t].overlap[r]])
                    all_values.append(structure_values)
                    i = i + 1

            values_df = pd.DataFrame(np.asarray(all_values), columns=column_names) if len(np.asarray(all_values)) > 1 else pd.DataFrame()
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