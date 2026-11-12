import os
import json
import logging
import pandas as pd
from ..DataStructures.StatisticsStructure import CompositionStatistics
from .AbstractReportingStructure import AbstractReportingStructure


class NeuroTimepointReportingStructure(AbstractReportingStructure):
    """
    Relations between the segmented structures of one timestamp (compartments of a lesion and reference structures),
    on top of the per-structure measurements of the features computation, which are referenced and not duplicated.
    Written as reporting/T<n>/timepoint_report.json (meant to be read by other programs, versioned with
    schema_version), .txt and .csv.
    """
    schema_version = 1  # Version of the timepoint_report.json content, to increase when it changes in a non-additive way
    _space = None  # Space of the masks the relations were computed on
    _base_volume_uid = None  # Radiological volume whose grid the masks share
    _features_report_uid = None  # Report of the features computation for the same timestamp, if any
    _composition = None  # CompositionStatistics

    def __init__(self, id: str, output_folder: str, timestamp: int, space: str = "Patient"):
        super(NeuroTimepointReportingStructure, self).__init__(id=id, output_folder=output_folder, timestamp=timestamp)
        self._space = space
        self._base_volume_uid = None
        self._features_report_uid = None
        self._composition = None

    @property
    def space(self) -> str:
        return self._space

    @property
    def base_volume_uid(self) -> str:
        return self._base_volume_uid

    @base_volume_uid.setter
    def base_volume_uid(self, value: str) -> None:
        self._base_volume_uid = value

    @property
    def features_report_uid(self) -> str:
        return self._features_report_uid

    @features_report_uid.setter
    def features_report_uid(self, value: str) -> None:
        self._features_report_uid = value

    @property
    def composition(self) -> CompositionStatistics:
        return self._composition

    @composition.setter
    def composition(self, value: CompositionStatistics) -> None:
        self._composition = value

    def get_reporting_folder(self) -> str:
        return os.path.join(self.output_folder, "reporting", "T" + str(self.timestamp))

    def to_dict(self) -> dict:
        report = {}
        report["schema_version"] = self.schema_version
        report["report_uid"] = self.unique_id
        report["timestamp"] = self.timestamp
        report["space"] = self.space
        report["base_volume_uid"] = self.base_volume_uid
        report["features_report_uid"] = self.features_report_uid
        composition = self.composition.to_dict()
        for key in composition.keys():
            report[key] = composition[key]
        return report

    def to_json(self) -> None:
        try:
            filename = os.path.join(self.get_reporting_folder(), "timepoint_report.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            logging.info(f"Exporting timepoint report for T{self.timestamp} to json in {filename}.")
            with open(filename, 'w', newline='\n') as outfile:
                json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)
        except Exception as e:
            raise RuntimeError(f"Timepoint report dump on disk for T{self.timestamp} as json failed with {e}")

    def to_txt(self) -> None:
        try:
            filename = os.path.join(self.get_reporting_folder(), "timepoint_report.txt")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            logging.info(f"Exporting timepoint report for T{self.timestamp} to text in {filename}.")
            pfile = open(filename, 'w')
            pfile.write(f'########### Raidionics timepoint report for timestamp {self.timestamp} in {self.space} space ###########\n')
            pfile.write(f'Composition: {self.composition.composition_name} (version {self.composition.composition_version})\n')
            pfile.write('\n Compartments:\n')
            for role in self.composition.compartments.keys():
                compartment = self.composition.compartments[role]
                derived = ""
                if len(compartment.derived_from) != 0:
                    derived = f" (derived from {', '.join(compartment.derived_from)})"
                pfile.write(f'  * {role} ({compartment.structure}): {round(compartment.volume, 2)} ml{derived}\n')
            for role in self.composition.references.keys():
                reference = self.composition.references[role]
                pfile.write(f'  * reference {role} ({reference.structure}): {round(reference.volume, 2)} ml\n')
            if len(self.composition.ratios) != 0:
                pfile.write('\n Ratios:\n')
                for name in self.composition.ratios.keys():
                    pfile.write(f'  * {name}: {round(self.composition.ratios[name], 4)}\n')
            if len(self.composition.rim) != 0:
                pfile.write('\n Rim thickness:\n')
                pfile.write(f'  * mean {round(self.composition.rim["mean_mm"], 2)} mm, min {round(self.composition.rim["min_mm"], 2)} mm, '
                            f'max {round(self.composition.rim["max_mm"], 2)} mm\n')
            if len(self.composition.consistency) != 0:
                pfile.write('\n Inconsistencies:\n')
                for role_a, role_b, overlap in self.composition.consistency:
                    pfile.write(f'  * {role_a} and {role_b} overlap by {round(overlap, 2)} ml\n')
            if len(self.composition.notes) != 0:
                pfile.write('\n Notes:\n')
                for note in self.composition.notes:
                    pfile.write(f'  * {note}\n')
            pfile.close()
        except Exception as e:
            raise RuntimeError(f"Timepoint report dump on disk for T{self.timestamp} as text failed with {e}")

    def to_csv(self) -> None:
        """
        One row with the volumes, ratios and rim thickness, for spreadsheet use.
        """
        try:
            filename = os.path.join(self.get_reporting_folder(), "timepoint_report.csv")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            logging.info(f"Exporting timepoint report for T{self.timestamp} to csv in {filename}.")
            column_names = ["Timestamp", "Composition"]
            values = [self.timestamp, self.composition.composition_name]
            for role in self.composition.compartments.keys():
                column_names.append(f"{role} volume (ml)")
                values.append(self.composition.compartments[role].volume)
            for role in self.composition.references.keys():
                column_names.append(f"{role} volume (ml)")
                values.append(self.composition.references[role].volume)
            for name in self.composition.ratios.keys():
                column_names.append(name)
                values.append(self.composition.ratios[name])
            for key in ["mean_mm", "min_mm", "max_mm"]:
                column_names.append(f"rim {key}")
                if key in self.composition.rim.keys():
                    values.append(self.composition.rim[key])
                else:
                    values.append(None)
            column_names.append("Inconsistencies")
            values.append(len(self.composition.consistency))
            values_df = pd.DataFrame([values], columns=column_names)
            values_df.to_csv(filename, index=False)
        except Exception as e:
            raise RuntimeError(f"Timepoint report dump on disk for T{self.timestamp} as csv failed with {e}")
