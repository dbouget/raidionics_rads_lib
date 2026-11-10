import json
from typing import Dict, List


"""
Generic statistics computed over a segmented structure, independent of the organ (e.g., brain or lungs) and of the
application using them. The application-oriented reports (see Utils/ReportingStructures) are built from these.
A default value of -1 (or None for the booleans) means not computed.
"""


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


class AtlasOverlapStatistics:
    """
    Profile of the structure with regard to the labels of an atlas: the overlap with each label (in %), the distance
    to each label (in mm), and the number of labels overlapped. The distances and count are only filled when computed
    for the atlas (e.g., distances to the subcortical tracts, count of infiltrated voxels from BrainGrid).
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


class StructureStatistics:
    """
    Statistics of one structure (e.g., Tumor, Brain, a lymph node) in one space (e.g., Patient, MNI).
    The atlas profiles are grouped by category (e.g., cortical, subcortical) and then by atlas name:
    profiles["cortical"]["Harvard-Oxford"] = AtlasOverlapStatistics(...)
    """
    _structure = None
    _space = None
    _volume = None
    _diameters = None
    _multifocality = None
    _profiles = None

    def __init__(self, structure: str = None, space: str = None):
        self._structure = structure
        self._space = space
        self._volume = VolumeStatistics()
        self._diameters = DiameterStatistics()
        self._multifocality = MultifocalityStatistics()
        self._profiles = {}

    @property
    def structure(self) -> str:
        return self._structure

    @structure.setter
    def structure(self, value: str) -> None:
        self._structure = value

    @property
    def space(self) -> str:
        return self._space

    @space.setter
    def space(self, value: str) -> None:
        self._space = value

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
    def profiles(self) -> Dict[str, Dict[str, AtlasOverlapStatistics]]:
        return self._profiles

    @profiles.setter
    def profiles(self, value: Dict[str, Dict[str, AtlasOverlapStatistics]]) -> None:
        self._profiles = value

    def to_dict(self) -> dict:
        """
        Content of the per-structure statistics json, a value of -1 meaning not computed.
        """
        statistics = {}
        statistics["structure"] = self.structure
        statistics["space"] = self.space
        statistics["volume"] = {}
        statistics["volume"]["volume_ml"] = float(self.volume.volume)
        statistics["volume"]["reference_percentage"] = float(self.volume.reference_percentage)
        statistics["diameters"] = {}
        statistics["diameters"]["long_axis_mm"] = float(self.diameters.long_axis_diameter)
        statistics["diameters"]["short_axis_mm"] = float(self.diameters.short_axis_diameter)
        statistics["diameters"]["feret_diameter_mm"] = float(self.diameters.feret_diameter)
        statistics["diameters"]["equivalent_diameter_mm"] = float(self.diameters.equivalent_diameter_area)
        statistics["multifocality"] = {}
        statistics["multifocality"]["status"] = self.multifocality.multifocality
        statistics["multifocality"]["parts"] = int(self.multifocality.nb_parts)
        statistics["multifocality"]["max_distance_mm"] = float(self.multifocality.max_distance)
        statistics["multifocality"]["foci_parameters"] = self.multifocality.foci_parameters
        statistics["multifocality"]["sum_of_products_mm2"] = float(self.multifocality.sum_of_products)
        statistics["multifocality"]["foci"] = []
        for f in self.multifocality.foci:
            statistics["multifocality"]["foci"].append(f.to_dict())
        statistics["profiles"] = {}
        for category in self.profiles.keys():
            statistics["profiles"][category] = {}
            for atlas in self.profiles[category].keys():
                profile = self.profiles[category][atlas]
                statistics["profiles"][category][atlas] = {}
                statistics["profiles"][category][atlas]["overlap"] = profile.overlap
                statistics["profiles"][category][atlas]["distance"] = profile.distance
                statistics["profiles"][category][atlas]["count"] = profile.count
        return statistics

    def to_json(self, filename: str) -> None:
        with open(filename, 'w', newline='\n') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, sort_keys=True)

    def get_profiles(self, category: str) -> Dict[str, AtlasOverlapStatistics]:
        """
        Atlas profiles of the given category, as {atlas name: AtlasOverlapStatistics}, created empty when missing so
        that profiles can be added directly (e.g., statistics.get_profiles("cortical")["MNI"] = ...).
        """
        if category not in self._profiles.keys():
            self._profiles[category] = {}
        return self._profiles[category]
