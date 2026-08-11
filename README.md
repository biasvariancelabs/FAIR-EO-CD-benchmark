# FAIR-EO-CD-benchmark

## Datasets
| Dataset | Data source | Data splits | | | Dataloader |
|---------|-------------|-------------|--|--|------|
| Bandon  | [GitHub](https://github.com/fitzpchao/BANDON) | [Train](splits/bandon_train.csv)| [Validation](splits/bandon_val.csv) | [Test](splits/bandon_test.csv)| [Python file](dataloaders/bandon.py)|
| CLCD   | [GitHub](https://github.com/liumency/CropLand-CD) | [Train](splits/clcd_train.csv)| [Validation](splits/clcd_val.csv) | [Test](splits/clcd_test.csv)| [Python file](dataloaders/clcd.py)|
| DSIFN |[GitHub](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)|[Train](splits/dsifn_train.csv)|[Validation](splits/dsifn_val.csv) |[Test](splits/dsifn_test.csv)| [Python file](dataloaders/dsifn.py)|
| EGY-BCD  | [GitHub](https://github.com/oshholail/EGY-BCD) | [Train](splits/egy_bcd_train.csv)| [Validation](splits/egy_bcd_val.csv) | [Test](splits/egy_bcd_test.csv)| [Python file](dataloaders/egy_bcd.py)|
| LEVIR-CD+  | [GitHub](https://github.com/S2Looking/Dataset/) | [Train](splits/levir_cd_train.csv)| [Validation](splits/levir_cd_val.csv) | [Test](splits/levir_cd_test.csv)| [Python file](dataloaders/levir_cdplus.py)|
| MSBC   | [GitHub](https://github.com/Lihy256/MSCDUnet) | [Train](splits/msbc_train.csv)| [Validation](splits/msbc_val.csv) | [Test](splits/msbc_test.csv)| [Python file](dataloaders/msbc.py)|
| MSOSCD | [GitHub](https://github.com/Lihy256/MSCDUnet) | [Train](splits/msoscd_train.csv)| [Validation](splits/msoscd_val.csv) | [Test](splits/msoscd_test.csv)| [Python file](dataloaders/msoscd.py)|
| OMBRIA   | [GitHub](https://github.com/geodrak/OMBRIA) | [Train](splits/ombria_train.csv)| [Validation](splits/ombria_val.csv) | [Test](splits/ombria_test.csv)| [Python file](dataloaders/ombria.py)|
| Season-Varying CDD|[Google Drive](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9)|[Train](splits/season_varying_cdd_train.csv)|[Validation](splits/season_varying_cdd_val.csv)|[Test](splits/season_varying_cdd_test.csv)|[Python file](dataloaders/season_varying_cdd.py)|
| SYSU-CD   | [GitHub](https://github.com/liumency/SYSU-CD) | [Train](splits/sysu_cd_train.csv)| [Validation](splits/sysu_cd_val.csv) | [Test](splits/sysu_cd_test.csv)| [Python file](dataloaders/sysu_cd.py)|

## Performance
Mean Intersection over Union (mIoU) for models trained from scratch.
| Dataset/Model | BIT | CGNet | ChangeFormerV6 | ChangeViT | CSSM | HRNet SiamConc | SiamCRNN | STANet | TinyCD | U-Net SiamConc | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BANDON | 0.5015 | 0.4785 | **0.5661** | <u>0.5545</u> | 0.4787 | 0.5231 | 0.5511 | 0.4921 | 0.5306 | 0.4840 | 0.5160 |
| CLCD | 0.6920 | 0.6593 | 0.6569 | 0.6637 | 0.4596 | 0.6375 | <u>0.6950</u> | 0.6120 | 0.6612 | **0.7048** | 0.6442 |
| DSIFN | 0.7711 | **0.8109** | 0.7453 | <u>0.7919</u> | 0.6829 | 0.6409 | 0.7863 | 0.7224 | 0.7133 | 0.7593 | 0.7424 |
| EGY-BCD | 0.7783 | **0.8214** | 0.7594 | 0.7770 | 0.7005 | 0.7619 | <u>0.7929</u> | 0.7598 | 0.7619 | 0.7762 | 0.7689 |
| LEVIR-CD+ | 0.6378 | 0.6412 | 0.6561 | 0.7181 | 0.5785 | 0.6766 | 0.6907 | 0.5449 | <u>0.7201</u> | **0.7201** | 0.6584 |
| MSBC | 0.8613 | **0.8846** | <u>0.8730</u> | 0.8644 | 0.6554 | 0.8172 | 0.8729 | 0.7875 | 0.7809 | 0.8699 | 0.8267 |
| MSOSCD | 0.6603 | 0.6748 | 0.6612 | <u>0.6826</u> | 0.5330 | 0.6140 | **0.7248** | 0.6094 | 0.6003 | 0.6817 | 0.6442 |
| OMBRIA | 0.6438 | 0.6343 | **0.6934** | <u>0.6632</u> | 0.6396 | 0.6123 | 0.6302 | 0.5798 | 0.6138 | 0.6302 | 0.6341 |
| Season-varying CDD | 0.8896 | 0.6578 | 0.8110 | **0.9141** | 0.7507 | 0.8584 | <u>0.9133</u> | 0.8296 | 0.8556 | 0.8887 | 0.8369 |
| SYSU-CD | 0.7066 | **0.7646** | 0.7212 | <u>0.7512</u> | 0.6909 | 0.6654 | 0.7350 | 0.6709 | 0.6650 | 0.7090 | 0.7080 |
