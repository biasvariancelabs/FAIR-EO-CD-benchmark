"""Names, orders and loading helpers shared by the results scripts."""
import pandas as pd

# AiTLAS model class -> name in the paper, in the column order of the tables
MODELS = {
    "BIT": "BIT",
    "CGNet": "CGNet",
    "ChangeFormerV6": "ChangeFormerV6",
    "ChangeViT": "ChangeViT",
    "CSSM": "CSSM",
    "HRNetChangeDetection": "HRNet SiamConc",
    "SiamCRNN": "SiamCRNN",
    "STANet": "STANet",
    "TinyCD": "TinyCD",
    "UnetChangeDetection": "U-Net SiamConc",
}
# Dataset id -> name in the paper, in the row order of the tables
DATASETS = {
    "bandon": "BANDON",
    "clcd": "CLCD",
    "dsifn": "DSIFN",
    "egy_bcd": "EGY-BCD",
    "levir_cd+": "LEVIR-CD+",
    "msbc": "MSBC",
    "msoscd": "MSOSCD",
    "ombria": "OMBRIA",
    "season_varying_cdd": "Season-varying CDD",
    "sysu_cd": "SYSU-CD",
}
REGIMES = ["scratch", "pretrained"]
# AiTLAS has no ImageNet-1K weights for these models, so their "pretrained" runs start from random weights
NO_PRETRAINED = {"ChangeFormerV6", "CSSM"}


def load_runs(path, keep_no_pretrained=False):
    """Read results/runs.csv. By default, drop the pretrained runs of the models without pretrained weights."""
    df = pd.read_csv(path)
    unknown = (set(df["model"]) - set(MODELS)) | (set(df["dataset"]) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown models or datasets in {path}: {sorted(unknown)}")
    if not keep_no_pretrained:
        df = df[~((df["regime"] == "pretrained") & df["model"].isin(NO_PRETRAINED))]
    return df


def matrix(df, metric, regime):
    """Datasets x models table of one metric for one regime, in the order of the paper (NaN where there is no run)."""
    sub = df[df["regime"] == regime]
    return sub.pivot(index="dataset", columns="model", values=metric).reindex(index=list(DATASETS), columns=list(MODELS))
