"""Dataset loaders for the FAIR-EO change detection benchmark, built on AiTLAS."""

from .bandon import BandonDataset
from .change_detection import ChangeDetectionDataset
from .clcd import CLCDDataset
from .dsifn import DSIFNDataset
from .egy_bcd import EGYBCDDataset
from .levir_cdplus import LEVIRCDPlusDataset
from .msbc import MSBCDataset
from .msoscd import MSOSCDDataset
from .ombria import OMBRIADataset
from .season_varying_cdd import SeasonVaryingCDDDataset
from .sysu_cd import SYSUCDDataset

__all__ = [
    "BandonDataset",
    "ChangeDetectionDataset",
    "CLCDDataset",
    "DSIFNDataset",
    "EGYBCDDataset",
    "LEVIRCDPlusDataset",
    "MSBCDataset",
    "MSOSCDDataset",
    "OMBRIADataset",
    "SeasonVaryingCDDDataset",
    "SYSUCDDataset",
]
