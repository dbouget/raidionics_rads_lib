import json
from typing import Dict, List, Tuple, TextIO
import operator
import collections


class RadiologicalVolumeStatistics:
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


class VolumeStatistics:
    """
    Volume of the structure, and its proportion of a reference structure (e.g., brain or lungs) when available.
    @TODO. add volume in mm³?
    """
    _volume = -1.
    _reference_percentage = -1.

    def __init__(self, volume: float = -1., reference_percentage: float = -1.):
        self._volume = volume
        self._reference_percentage = reference_percentage

    @property
    def volume(self) -> float:
        """Volume in milliliters."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = value

    @property
    def reference_percentage(self) -> float:
        """Ratio of the structure volume over the reference structure volume, between 0 and 1."""
        return self._reference_percentage

    @reference_percentage.setter
    def reference_percentage(self, value: float) -> None:
        self._reference_percentage = value

    def to_dict(self) -> dict:
        statistics = {"volume": {}}
        statistics["volume"]["volume_ml"] = self.volume
        statistics["volume"]["reference_percentage"] = self.reference_percentage

        return statistics

    def to_json(self, filename: str) -> None:
        with open(filename, 'w', newline='\n') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)

    def to_csv_inputs(self) -> Tuple[List[str], List]:
        column_names = ["Volume (ml)", "Reference percentage (%)"]
        values = [self.volume, self.reference_percentage]
        return column_names, values

    def to_txt(self, pfile: TextIO) -> None:
        pfile.write(' Volumes: \n')
        pfile.write(f'  * Volume: {self.volume} (ml)\n')
        pfile.write(f'  * Reference percentage: {self.reference_percentage} (%)\n')

class DiameterStatistics:
    """
    Diameters of the largest focus of the structure, in millimeters (see compute_diameters).
    """
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

    def to_dict(self) -> dict:
        statistics = {"diameters": {}}
        statistics["diameters"]["long_axis_mm"] = self.long_axis_diameter
        statistics["diameters"]["short_axis_mm"] = self.short_axis_diameter
        statistics["diameters"]["feret_diameter_mm"] = self.feret_diameter
        statistics["diameters"]["equivalent_diameter_mm"] = self.equivalent_diameter_area

        return statistics
    
    def to_json(self, filename: str) -> None:
        with open(filename, 'w', newline='\n') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)

    def to_csv_inputs(self) -> Tuple[List[str], List]:
        column_names = ["Long-axis diameter (mm)", "Short-axis diameter (mm)",
                        "Feret diameter maximum (mm)", "Equivalent diameter area (mm)"]
        values = [self.long_axis_diameter, self.short_axis_diameter, self.feret_diameter, self.equivalent_diameter_area]
        return column_names, values

    def to_txt(self, pfile: TextIO) -> None:
        pfile.write(' Diameters: \n')
        pfile.write(f'  * Long-axis diameter: {self.long_axis_diameter} (mm)\n')
        pfile.write(f'  * Short-axis diameter: {self.short_axis_diameter} (mm)\n')
        pfile.write(f'  * Feret diameter maximum: {self.feret_diameter} (mm)\n')
        pfile.write(f'  * Equivalent diameter area: {self.equivalent_diameter_area} (mm)\n\n')

class FocusStatistics:
    """
    One focus (connected component) of a structure, ranked by volume (1 = largest). The centroid is in world
    coordinates (mm) and the diameters follow compute_diameters. A focus is measurable (RANO) when both its long and
    short axis diameters are at least 10 mm.
    A focus groups the connected components closer than the multifocality distance threshold (components gives their
    number), and is measured over all of them. The distance to the main focus (rank 1) is the HD95 reported by
    compute_multifocality, and the minimum distance the smallest gap between the two (0 for the main focus itself).
    """
    def __init__(self, rank: int = -1, volume: float = -1., centroid: list = None, diameters: DiameterStatistics = None,
                 measurement_slice: int = -1, distance_to_main: float = -1., min_distance_to_main: float = -1.,
                 components: int = 1):
        self.rank = rank
        self.volume = volume
        self.centroid = centroid
        self.diameters = diameters
        if self.diameters is None:
            self.diameters = DiameterStatistics()
        self.measurement_slice = measurement_slice
        self.distance_to_main = distance_to_main
        self.min_distance_to_main = min_distance_to_main
        self.components = components

    @property
    def is_measurable(self) -> bool:
        return self.diameters.long_axis_diameter >= 10. and self.diameters.short_axis_diameter >= 10.

    def to_dict(self) -> dict:
        focus = {}
        focus["rank"] = int(self.rank)
        focus["volume_ml"] = float(self.volume)
        focus["centroid_mm"] = None
        if self.centroid is not None:
            focus["centroid_mm"] = [float(x) for x in self.centroid]
        focus["long_axis_mm"] = float(self.diameters.long_axis_diameter)
        focus["short_axis_mm"] = float(self.diameters.short_axis_diameter)
        focus["feret_diameter_mm"] = float(self.diameters.feret_diameter)
        focus["equivalent_diameter_mm"] = float(self.diameters.equivalent_diameter_area)
        focus["measurement_slice"] = int(self.measurement_slice)
        focus["is_measurable"] = self.is_measurable
        focus["distance_to_main_mm"] = float(self.distance_to_main)
        focus["min_distance_to_main_mm"] = float(self.min_distance_to_main)
        focus["components"] = int(self.components)
        return focus

    def to_csv_inputs(self) -> Tuple[List[str], List]:
        column_names = ["Rank", "Volume (ml)", "Centroid (mm)", "Long-axis (mm)", "Short-axis (mm)"]
        values = [self.rank, self.volume, ",".join(self.centroid), 
                  self.diameters.long_axis_diameter, self.diameters.short_axis_diameter]
        return column_names, values


class MultifocalityStatistics:
    """
    Number of disconnected parts (foci) of the structure, and the largest distance between them in millimeters.
    The foci themselves (FocusStatistics, ranked by volume) are filled when computed, with the parameters used to
    define them (e.g., {"connectivity": 1, "min_volume_ml": 0.1}).
    """
    _multifocality = None
    _nb_parts = -1
    _max_distance = -1.
    _foci = None
    _foci_parameters = None

    def __init__(self, status: bool = None, parts: int = -1, distance: float = -1., foci: List = []):
        self._multifocality = status
        self._nb_parts = parts
        self._max_distance = distance
        self._foci = foci
        self._foci_parameters = {}

    @property
    def foci(self) -> List[FocusStatistics]:
        return self._foci

    @foci.setter
    def foci(self, value: List[FocusStatistics]) -> None:
        self._foci = value

    @property
    def foci_parameters(self) -> dict:
        return self._foci_parameters

    @foci_parameters.setter
    def foci_parameters(self, value: dict) -> None:
        self._foci_parameters = value

    @property
    def sum_of_products(self) -> float:
        """RANO sum of the products of perpendicular diameters (mm2), over the measurable foci."""
        total = 0.
        for f in self._foci:
            if f.is_measurable:
                total = total + f.diameters.long_axis_diameter * f.diameters.short_axis_diameter
        return total

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

    def to_dict(self) -> dict:
        statistics = {"multifocality": {}}
        statistics["multifocality"]["status"] = self.multifocality
        statistics["multifocality"]["parts"] = self.nb_parts
        statistics["multifocality"]["max_distance_mm"] = self.max_distance
        statistics["multifocality"]["foci_parameters"] = self.foci_parameters
        statistics["multifocality"]["sum_of_products_mm2"] = self.sum_of_products
        statistics["multifocality"]["foci"] = []
        for f in self.foci:
            statistics["multifocality"]["foci"].append(f.to_dict())

        return statistics

    def to_json(self, filename: str) -> None:
        with open(filename, 'w', newline='\n') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)

    def to_csv_inputs(self) -> Tuple[List[str], List]:
        column_names = ["Multifocality", "Number parts", "Multifocal distance (mm)"]
        values = [self.multifocality, self.nb_parts, self.max_distance]
        return column_names, values

    def to_txt(self, pfile: TextIO) -> None:
        pfile.write(' Multifocality: \n')
        pfile.write(f'  * Status: {self.multifocality}\n')
        pfile.write(f'  * Number parts: {self.nb_parts}\n')
        pfile.write(f'  * Largest distance between components: {self.max_distance} (mm)\n\n')

class AtlasOverlapStatistics:
    """
    Profile of the structure with regard to the labels of an atlas: the overlap with each label (in %), the distance
    to each label (in mm), and the number of labels overlapped.
    """
    _atlas = None
    _overlap = None
    _distance = None
    _count = None

    def __init__(self, atlas: str = None, overlap: dict = None, distance: dict = None, count: int = None):
        self._atlas = atlas
        self._overlap = overlap
        self._distance = distance
        self._count = count

    @property
    def atlas(self) -> str:
        return self._atlas

    @atlas.setter
    def atlas(self, value: str) -> None:
        self._atlas = value

    @property
    def overlap(self) -> dict:
        return self._overlap

    @overlap.setter
    def overlap(self, value: dict) -> None:
        self._overlap = value

    @property
    def distance(self) -> dict:
        return self._distance

    @distance.setter
    def distance(self, value: dict) -> None:
        self._distance = value

    @property
    def count(self) -> int:
        return self._count

    @count.setter
    def count(self, value: int) -> None:
        self._count = value

    def to_dict(self) -> dict:
        statistics = {"Overlap": {}, "Distance": {}, "Count": self.count}
        sorted_overlap = collections.OrderedDict(sorted(self.overlap.items(),
                                                        key=operator.itemgetter(1), reverse=True))
        sorted_distance = collections.OrderedDict(sorted(self.distance.items(),
                                                         key=operator.itemgetter(1), reverse=False))
        for k in sorted_overlap.keys():
            statistics["Overlap"][k] = sorted_overlap[k]
        for k in sorted_distance.keys():
            statistics["Distance"][k] = sorted_distance[k]
        return statistics

    def to_json(self, filename: str) -> None:
        with open(filename, 'w', newline='\n') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)

    def to_csv_inputs(self) -> Tuple[List[str], List]:
        column_names = []
        values = []
        for k in self.overlap.keys():
            if self.atlas == "MNI":
                column_names.extend([self.atlas + '_' + k.split('.')[0][:-4] + '_overlap'])
            else:
                column_names.extend([self.atlas + '_' + k.split('.')[0] + '_overlap'])
            values.extend([self.overlap[k]])
        for k in self.distance.keys():
            if self.atlas == "MNI":
                column_names.extend([self.atlas + '_' + k.split('.')[0][:-4] + '_distance'])
            else:
                column_names.extend([self.atlas + '_' + k.split('.')[0] + '_distance'])
            values.extend([self.distance[k]])
        return column_names, values

    def to_txt(self, pfile: TextIO) -> None:
        pfile.write('  * {} atlas\n'.format(self.atlas))
        pfile.write('  Total infiltrated regions: {}\n'.format(self.count))
        overlap_ordered = collections.OrderedDict(
            sorted(self.overlap.items(),
                key=operator.itemgetter(1), reverse=True))
        for r in overlap_ordered.keys():
            if overlap_ordered[r] > 0:
                struct_name = ' '.join(r.lower().replace('main', '').replace('main', '').split('_')[:])
                pfile.write('    - {}: {}%\n'.format(struct_name, overlap_ordered[r]))

        pfile.write('\n')
        distance_ordered = collections.OrderedDict(
            sorted(self.distance.items(),
                key=operator.itemgetter(1), reverse=False))
        for r in distance_ordered.keys():
            if distance_ordered[r] > 0:
                struct_name = ' '.join(r.lower().replace('main', '').replace('mni', '').split('_')[:])
                pfile.write('    - {}: {}mm away\n'.format(struct_name, distance_ordered[r]))

class CompartmentStatistics:
    """
    One compartment of a lesion at a timestamp (e.g., the enhancing part of a tumour), identified by its role in the
    composition (see raidionicsrads/Compositions) and the structure (AnnotationClassType name) it comes from.
    A derived compartment has no mask of its own, it is the union of other compartments (e.g., the core from the
    enhancing part and the necrosis). The volume is computed from the mask, on the grid shared by all compartments.
    """
    _role = None  # Role in the composition, e.g. core, enhancing, necrosis
    _structure = None  # AnnotationClassType name, e.g. TumorCE
    _volume = -1.  # Volume in ml
    _derived_from = None  # Roles the compartment was derived from, empty when coming from its own mask
    _mask_filepath = None  # Mask used, None for a derived compartment
    _statistics_filename = None  # Per-structure statistics json of the features computation, if any

    def __init__(self, role: str = None, structure: str = None, volume: float = -1., derived_from: list = None,
                 mask_filepath: str = None, statistics_filename: str = None):
        self._role = role
        self._structure = structure
        self._volume = volume
        self._derived_from = derived_from
        if self._derived_from is None:
            self._derived_from = []
        self._mask_filepath = mask_filepath
        self._statistics_filename = statistics_filename

    @property
    def role(self) -> str:
        return self._role

    @role.setter
    def role(self, value: str) -> None:
        self._role = value

    @property
    def structure(self) -> str:
        return self._structure

    @structure.setter
    def structure(self, value: str) -> None:
        self._structure = value

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = value

    @property
    def derived_from(self) -> list:
        return self._derived_from

    @derived_from.setter
    def derived_from(self, value: list) -> None:
        self._derived_from = value

    @property
    def mask_filepath(self) -> str:
        return self._mask_filepath

    @mask_filepath.setter
    def mask_filepath(self, value: str) -> None:
        self._mask_filepath = value

    @property
    def statistics_filename(self) -> str:
        return self._statistics_filename

    @statistics_filename.setter
    def statistics_filename(self, value: str) -> None:
        self._statistics_filename = value

    def to_dict(self) -> dict:
        compartment = {}
        compartment["role"] = self.role
        compartment["structure"] = self.structure
        compartment["volume_ml"] = float(self.volume)
        compartment["derived_from"] = self.derived_from
        compartment["mask_filepath"] = self.mask_filepath
        compartment["statistics_filename"] = self.statistics_filename
        return compartment


class CompositionStatistics:
    """
    Relations between the compartments of a lesion (and its reference structures, e.g. the brain) at one timestamp:
    * ratios: {name: value}, between compartment volumes (e.g., necrosis_fraction = necrosis / core).
    * rim: thickness (mm) of the outer compartment around the inner one (e.g., enhancing rim around the necrosis),
    as {"mean_mm", "min_mm", "max_mm", "std_mm"}, empty when not applicable.
    * consistency: overlaps found between compartments meant to be disjoint, as [[role_a, role_b, overlap in ml]].
    * notes: what could not be computed, and why.
    """
    _composition_name = None  # Name of the composition schema used
    _composition_version = None  # Version of the composition schema used
    _compartments = None  # {role: CompartmentStatistics}
    _references = None  # {role: CompartmentStatistics}, e.g. the brain
    _ratios = None
    _rim = None
    _consistency = None
    _notes = None

    def __init__(self, composition_name: str = None, composition_version: int = None):
        self._composition_name = composition_name
        self._composition_version = composition_version
        self._compartments = {}
        self._references = {}
        self._ratios = {}
        self._rim = {}
        self._consistency = []
        self._notes = []

    @property
    def composition_name(self) -> str:
        return self._composition_name

    @property
    def composition_version(self) -> int:
        return self._composition_version

    @property
    def compartments(self) -> Dict[str, CompartmentStatistics]:
        return self._compartments

    @property
    def references(self) -> Dict[str, CompartmentStatistics]:
        return self._references

    @property
    def ratios(self) -> dict:
        return self._ratios

    @property
    def rim(self) -> dict:
        return self._rim

    @rim.setter
    def rim(self, value: dict) -> None:
        self._rim = value

    @property
    def consistency(self) -> list:
        return self._consistency

    @property
    def notes(self) -> list:
        return self._notes

    def to_dict(self) -> dict:
        composition = {}
        composition["composition"] = {}
        composition["composition"]["name"] = self.composition_name
        composition["composition"]["version"] = self.composition_version
        composition["compartments"] = {}
        for role in self.compartments.keys():
            composition["compartments"][role] = self.compartments[role].to_dict()
        composition["references"] = {}
        for role in self.references.keys():
            composition["references"][role] = self.references[role].to_dict()
        composition["ratios"] = {}
        for name in self.ratios.keys():
            composition["ratios"][name] = float(self.ratios[name])
        composition["rim"] = {}
        for key in self.rim.keys():
            composition["rim"][key] = float(self.rim[key])
        composition["consistency"] = []
        for role_a, role_b, overlap in self.consistency:
            composition["consistency"].append({"roles": [role_a, role_b], "overlap_ml": float(overlap)})
        composition["notes"] = self.notes
        return composition
