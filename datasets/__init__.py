from .blender import BlenderDataset
from .llff import LLFFDataset
from .phototourism import PhototourismDataset
from .behave import BehaveDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'phototourism': PhototourismDataset,
                'behave': BehaveDataset}