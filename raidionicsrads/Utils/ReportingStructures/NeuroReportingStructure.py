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


class NeuroMultifocalityStatistics:
    _multifocality = None
    _nb_parts = None
    _max_distance = None
    def __init__(self, status: bool = None, parts: int = None, distance: float = None):
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
    _left_laterality_percentage = None
    _right_laterality_percentage = None
    _laterality_midline_crossing = None

    def __init__(self, left: float = None, right: float = None, crossing: bool = None):
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
    _expected_resectable_tumor_volume = None
    _expected_residual_tumor_volume = None
    _resectability_index = None

    def __init__(self, resectable: float = None, residual: float = None, index: float = None):
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
    _multifocality = None
    _location = None
    _resectability = None
    _cortical = None
    _subcortical = None
    _infiltration = None

    def __init__(self):
        self._volume = None
        self._multifocality = NeuroMultifocalityStatistics()
        self._location = NeuroLocationStatistics()
        self._resectability = NeuroResectabilityStatistics()
        self._cortical = {}
        self._subcortical = {}
        self._infiltration = {}

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = value

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

    def to_disk(self) -> None:
        self.to_txt()
        # self.to_csv()
        # self.to_json()
        self.dump_descriptions()

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
                                    "T" + str(self._timestamp), "neuro_clinical_report.txt")
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            logging.info(f"Exporting standardized report for timestamp timestamp {self._timestamp} to text in {filename}.")
            pfile = open(filename, 'w')
            pfile.write('########### Raidionics standardized report for timestamp {} in MNI space ###########\n'.format(self._timestamp))
            pfile.write('Tumor type: {}\n'.format(self._tumor_type))

            pfile.write('\nVolumes: \n')
            for s in self.statistics.keys():
                pfile.write('  * {}: {}ml\n'.format(s, self.statistics[s]["MNI"].volume))
            for s in self.statistics.keys():
                pfile.write(f'\n Features for {s} category.\n')
                pfile.write(' Multifocality: \n')
                pfile.write(f'  * Status: {self.statistics[s]["MNI"].multifocality.multifocality}\n')
                pfile.write(f'  * Number parts: {self.statistics[s]["MNI"].multifocality.nb_parts}\n')
                pfile.write(f'  * Largest distance between components: {np.round(self.statistics[s]["MNI"].multifocality.max_distance, 2)} (mm)\n\n')

                pfile.write(' Location: \n')
                pfile.write(f'  * Left hemisphere: {self.statistics[s]["MNI"].location.left_laterality_percentage}\n')
                pfile.write(f'  * Right hemisphere: {self.statistics[s]["MNI"].location.right_laterality_percentage}\n')
                pfile.write(f'  * Midline crossing: {self.statistics[s]["MNI"].location.laterality_midline_crossing}\n\n')

                # @TODO. Should have an if tumor type is GBM
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
        # @TODO. to modify
        try:
            filename = os.path.join(self._output_folder, "neuro_clinical_report.json")
            logging.info("Exporting neuro-parameters to json in {}.".format(filename))
            param_json = {}
            param_json['Overall'] = {}
            # param_json['Overall']['Presence'] = self._tumor_presence_state
            # if not self._tumor_presence_state:
            #     with open(filename, 'w') as outfile:
            #         json.dump(param_json, outfile)
            #     return

            param_json['Overall']['Type'] = self._tumor_type
            param_json['Overall']['Multifocality'] = self._tumor_multifocal
            param_json['Overall']['Tumor parts nb'] = self._tumor_parts
            param_json['Overall']['Multifocal distance (mm)'] = np.round(self._tumor_multifocal_distance, 2)

            param_json['Main'] = {}
            # param_json['Main']['CenterOfMass'] = {}
            # param_json['Main']['CenterOfMass']['Laterality'] = self.statistics['Main']['CoM'].laterality
            # param_json['Main']['CenterOfMass']['Laterality_perc'] = self.statistics['Main']['CoM'].laterality_percentage
            # param_json['Main']['CenterOfMass']['Lobe'] = []
            # for l in self.statistics['Main']['CoM'].mni_space_cortical_structures_overlap.keys():
            #     param_json['Main']['CenterOfMass']['Lobe'].append([l, self.statistics['Main']['CoM'].mni_space_cortical_structures_overlap[l]])

            param_json['Main']['Total'] = {}
            param_json['Main']['Total']['Volume original (ml)'] = self._statistics['Main']['Overall'].original_space_volume
            param_json['Main']['Total']['Volume in MNI (ml)'] = self._statistics['Main']['Overall'].mni_space_tumor_volume
            param_json['Main']['Total']['Left laterality (%)'] = self._statistics['Main']['Overall'].left_laterality_percentage
            param_json['Main']['Total']['Right laterality (%)'] = self._statistics['Main']['Overall'].right_laterality_percentage
            param_json['Main']['Total']['Midline crossing'] = self._statistics['Main']['Overall'].laterality_midline_crossing

            if self._tumor_type == 'Glioblastoma':
                param_json['Main']['Total']['ExpectedResidualVolume (ml)'] = np.round(self._statistics['Main']['Overall'].mni_space_expected_residual_tumor_volume, 2)
                param_json['Main']['Total']['ResectionIndex'] = np.round(self._statistics['Main']['Overall'].mni_space_resectability_index, 3)

            param_json['Main']['Total']['CorticalStructures'] = {}
            for t in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap.keys():
                param_json['Main']['Total']['CorticalStructures'][t] = {}
                for r in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t].keys():
                    param_json['Main']['Total']['CorticalStructures'][t][r] = self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t][r]

            param_json['Main']['Total']['SubcorticalStructures'] = {}

            for t in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap.keys():
                param_json['Main']['Total']['SubcorticalStructures'][t] = {}
                param_json['Main']['Total']['SubcorticalStructures'][t]['Overlap'] = {}
                param_json['Main']['Total']['SubcorticalStructures'][t]['Distance'] = {}
                for ov in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t].keys():
                    param_json['Main']['Total']['SubcorticalStructures'][t]['Overlap'][ov] = self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t][ov]
                for di in self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[t].keys():
                    param_json['Main']['Total']['SubcorticalStructures'][t]['Distance'][di] = self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[t][di]

            if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                param_json['Main']['Total']['BrainGrid'] = {}
                param_json['Main']['Total']['BrainGrid']['Infiltration count'] = self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_count
                for t in self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_overlap.keys():
                    param_json['Main']['Total']['BrainGrid'][t] = {}
                    for r in self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_overlap[t].keys():
                        param_json['Main']['Total']['BrainGrid'][t][r] = \
                        self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_overlap[t][r]

            # Parameters for each tumor element
            # if self.tumor_multifocal:
            #     for p in range(self.tumor_parts):
            #         tumor_component = str(p+1)
            #         param_json[tumor_component] = {}
            #         param_json[tumor_component]['CenterOfMass'] = {}
            #         param_json[tumor_component]['CenterOfMass']['Laterality'] = self.statistics[tumor_component]['CoM'].laterality
            #         param_json[tumor_component]['CenterOfMass']['Laterality_perc'] = self.statistics[tumor_component]['CoM'].laterality_percentage
            #         param_json[tumor_component]['CenterOfMass']['Lobe'] = []
            #         for l in self.statistics[tumor_component]['CoM'].mni_space_cortical_structures_overlap.keys():
            #             param_json[tumor_component]['CenterOfMass']['Lobe'].append([l, self.statistics[tumor_component]['CoM'].mni_space_cortical_structures_overlap[l]])
            #
            #         param_json[tumor_component]['Total'] = {}
            #         param_json[tumor_component]['Total']['Volume'] = self.statistics[tumor_component]['Overall'].mni_space_tumor_volume
            #         param_json[tumor_component]['Total']['Laterality'] = self.statistics[tumor_component]['Overall'].laterality
            #         param_json[tumor_component]['Total']['Laterality_perc'] = np.round(self.statistics[tumor_component]['Overall'].laterality_percentage * 100., 2)
            #         param_json[tumor_component]['Total']['Resectability'] = np.round(self.statistics[tumor_component]['Overall'].mni_space_resectability_score * 100., 2)
            #
            #         param_json[tumor_component]['Total']['Lobe'] = []
            #
            #         for l in self.statistics[tumor_component]['Overall'].mni_space_cortical_structures_overlap.keys():
            #             param_json[tumor_component]['Total']['Lobe'].append([l, self.statistics[tumor_component]['Overall'].mni_space_cortical_structures_overlap[l]])
            #
            #         param_json[tumor_component]['Total']['Tract'] = {}
            #         param_json[tumor_component]['Total']['Tract']['Distance'] = []
            #
            #         for l in self.statistics[tumor_component]['Overall'].mni_space_tracts_distance.keys():
            #             if self.statistics[tumor_component]['Overall'].mni_space_tracts_distance[l] != -1.:
            #                 param_json[tumor_component]['Total']['Tract']['Distance'].append([l, np.round(self.statistics[tumor_component]['Overall'].mni_space_tracts_distance[l], 2)])
            #
            #         param_json[tumor_component]['Total']['Tract']['Overlap'] = []
            #         for l in self.statistics[tumor_component]['Overall'].mni_space_tracts_overlap.keys():
            #             if self.statistics[tumor_component]['Overall'].mni_space_tracts_overlap[l] > 0:
            #                 param_json[tumor_component]['Total']['Tract']['Overlap'].append([l, np.round(self.statistics[tumor_component]['Overall'].mni_space_tracts_overlap[l] * 100., 2)])

            with open(filename, 'w', newline='\n') as outfile:
                json.dump(param_json, outfile, indent=4, sort_keys=True)
        except Exception as e:
            raise RuntimeError("Neuro-parameters neuro report dump on disk as json failed with {}".format(e))

        return

    def to_csv(self) -> None:
        """
        Exporting the neuro report to a csv file on disk.
        """
        try:
            filename = os.path.join(self._output_folder, "neuro_clinical_report.csv")
            logging.info("Exporting neuro-parameters to csv in {}.".format(filename))

            # if not self._tumor_presence_state:
            #     return
            values = [self._tumor_multifocal, self._tumor_parts, np.round(self._tumor_multifocal_distance, 2)]
            column_names = ['Multifocality', 'Tumor parts nb', 'Multifocal distance (mm)']

            values.extend([self._statistics['Main']['Overall'].original_space_volume, self._statistics['Main']['Overall'].mni_space_tumor_volume])
            column_names.extend(['Volume original (ml)', 'Volume in MNI (ml)'])

            values.extend([self._statistics['Main']['Overall'].left_laterality_percentage,
                           self._statistics['Main']['Overall'].right_laterality_percentage,
                           self._statistics['Main']['Overall'].laterality_midline_crossing])
            column_names.extend(['Left laterality (%)', 'Right laterality (%)', 'Midline crossing'])

            if self._tumor_type == 'Glioblastoma':
                values.extend([np.round(self._statistics['Main']['Overall'].mni_space_expected_residual_tumor_volume, 2),
                               np.round(self._statistics['Main']['Overall'].mni_space_resectability_index, 3)])
                column_names.extend(['ExpectedResidualVolume (ml)', 'ResectionIndex'])

            for t in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap.keys():
                for r in self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t].keys():
                    values.extend([self._statistics['Main']['Overall'].mni_space_cortical_structures_overlap[t][r]])
                    column_names.extend([t + '_' + r.split('.')[0].lower().strip() + '_overlap'])

            for t in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap.keys():
                for r in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t].keys():
                    values.extend([self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t][r]])
                    if t == "MNI":
                        column_names.extend([t + '_' + r.split('.')[0][:-4] + '_overlap'])
                    else:
                        column_names.extend([t + '_' + r.split('.')[0] + '_overlap'])

            for t in self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance.keys():
                for r in self._statistics['Main']['Overall'].mni_space_subcortical_structures_overlap[t].keys():
                    values.extend([self._statistics['Main']['Overall'].mni_space_subcortical_structures_distance[t][r]])
                    if t == "MNI":
                        column_names.extend([t + '_' + r.split('.')[0][:-4] + '_distance'])
                    else:
                        column_names.extend([t + '_' + r.split('.')[0] + '_distance'])

            if len(ResourcesConfiguration.getInstance().neuro_features_braingrid) != 0:
                values.extend([self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_count])
                column_names.extend(['Total infiltrated voxels (BrainGrid)'])
                for t in self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_overlap.keys():
                    for r in self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_overlap[t].keys():
                        values.extend([self._statistics['Main']['Overall'].mni_space_braingrid_infiltration_overlap[t][r]])
                        column_names.extend([t + '_' + r.split('.')[0] + '_overlap'])

            values_df = pd.DataFrame(np.asarray(values).reshape((1, len(values))), columns=column_names)
            values_df.to_csv(filename, index=False)
        except Exception as e:
            raise RuntimeError("Neuro-parameters neuro report dump on disk as csv failed with {}".format(e))

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
            raise RuntimeError("Neuro-parameters atlas descriptions dump failed with {}".format(e))

