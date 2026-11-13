from marshmallow import fields

from aitlas.base.schemas import BaseDatasetSchema


class FAIREOSchema(BaseDatasetSchema):
    """
    Schema for configuring FAIR-EO datasets.
    """

    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/CESBIO/"},
    )
    selection = fields.String(
        load_default="rgb",
        metadata={"description": "Read RGB channels or more than 3 channels", "example": "all/rgb"},
    )
    mode = fields.String(
        load_default=None,
        metadata={"description": "train/val or test task", "example": "train"},
    )


class BandonSchema(BaseDatasetSchema):
    """
    Schema for configuring Bandon dataset.
    """

    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/CESBIO/"},
    )
    selection = fields.String(
        load_default="rgb",
        metadata={"description": "Read RGB channels or more than 3 channels", "example": "all/rgb"},
    )
    mode = fields.String(
        load_default=None,
        metadata={"description": "train/val or test task", "example": "train"},
    )
    split = fields.String(
        load_default="original",
        metadata={"description": "original or new 60/20/20 split", "example": "new"},
    )


class MSBCSchema(BaseDatasetSchema):
    """
    Schema for configuring MSBC dataset.
    """

    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/CESBIO/"},
    )
    selection = fields.String(
        load_default="rgb",
        metadata={"description": "Read RGB channels or more than 3 channels", "example": "all/rgb"},
    )
    mode = fields.String(
        load_default=None,
        metadata={"description": "train/val or test task", "example": "train"},
    )
    imagery = fields.String(
        load_default="s2",
        metadata={
            "description": "Consider only one of s1, s2, aerial or all sources",
            "example": "s1/s2/aerial/all",
        },
    )
    bands_s2 = fields.List(
        fields.String(),
        load_default=None,
        metadata={
            "description": "Select which S2 bands to use with a list of band names",
            "example": ["B01", "B02", "B03"],
        },
    )
    bands_s1 = fields.List(
        fields.String(),
        load_default=None,
        metadata={
            "description": "Select which S1 bands to use with a list of band names",
            "example": ["VV", "VH"],
        },
    )


class OmbriaSchema(BaseDatasetSchema):
    """
    Schema for configuring Ombria dataset.
    """

    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/CESBIO/"},
    )
    selection = fields.String(
        load_default="rgb",
        metadata={"description": "Read RGB channels or all original channels", "example": "all/rgb"},
    )
    mode = fields.String(
        load_default=None,
        metadata={"description": "train/val or test task", "example": "train"},
    )
    imagery = fields.String(
        load_default="S2",
        metadata={"description": "Consider only one of s1, s2 or all sources", "example": "S1/S2/all"},
    )
