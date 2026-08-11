<div align="center">
<h2>Trustworthy benchmarking of methods for change detection in remote sensing</h2>

[**Tadej Tomanič**](mailto:tadej@bvlabs.ai)<sup>1, 2</sup> · **Alice Baudhuin**<sup>1</sup> · **Jan Sotošek**<sup>1</sup> · **Jure Brence**<sup>1, 3</sup> · **Panče Panov**<sup>1, 3</sup> · **Nikola Simidjievski**<sup>1, 4</sup> · **Dragi Kocev**<sup>1, 3</sup>

<br>
<sup>1</sup>Bias Variance Labs, d.o.o.&emsp;&emsp;<sup>2</sup>University of Ljubljana, Faculty of Mathematics and Physics<br>
<sup>3</sup>Department of Knowledge Technologies, Jožef Stefan Institute&emsp;&emsp;<sup>4</sup>Télécom Paris, Institut Polytechnique de Paris

<br><br>
<a href='https://huggingface.co/bvlabs/FAIR-EO-CD-benchmark'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FAIR--EO--CD--benchmark-yellow'></a>
</div>

## Summary

Code and experiments for the paper, *Trustworthy benchmarking of methods for change detection in remote sensing*, by Tadej Tomanič, Alice Baudhuin, Jan Sotošek, Jure Brence, Panče Panov, Nikola Simidjievski, and Dragi Kocev (currently under review).

This study presents a standardized benchmark for change detection in Earth Observation, developed as part of the OSCARS-funded [FAIR-EO](https://oscars-project.eu/projects/fair-eo-fair-open-and-ai-ready-earth-observation-resources) project. The framework is integrated into the [AiTLAS](https://github.com/biasvariancelabs/aitlas) toolbox.

Despite rapid advancements in machine learning for remote sensing, accurately evaluating and comparing change detection models remains a significant challenge. Variations in dataset preprocessing, data splits, and evaluation metrics often lead to inconsistent results, making it difficult to determine whether a new method genuinely outperforms existing ones. To address this, we introduce a comprehensive, transparent, and trustworthy benchmarking framework designed to eliminate these ambiguities.

Aligned with the core principles of FAIR (Findable, Accessible, Interoperable, and Reusable) and open science, our framework provides a unified pipeline for the Earth Observation community. It establishes rigorous testing protocols, standardized evaluation metrics, and reproducible baselines.

## Datasets

Ah, got it! My mistake. Since this is going in a GitHub README, those relative links are exactly what you need so they point directly to the files in your repo.

Here is the table restored perfectly to the format you want:

| Dataset | Paper | Year | Data source | Data splits | Dataloader |
| --- | --- | --- | --- | --- | --- |
| BANDON | [Pang et al.](%23) | 2023 | [GitHub](https://github.com/fitzpchao/BANDON) | [Train](splits/bandon_train.csv) • [Validation](splits/bandon_val.csv) • [Test](splits/bandon_test.csv) | [Python file](dataloaders/bandon.py) |
| CLCD | [Liu et al.](%23) | 2022 | [GitHub](https://github.com/liumency/CropLand-CD) | [Train](splits/clcd_train.csv) • [Validation](splits/clcd_val.csv) • [Test](splits/clcd_test.csv) | [Python file](dataloaders/clcd.py) |
| DSIFN | [Zhang et al.](%23) | 2020 | [GitHub](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) | [Train](splits/dsifn_train.csv) • [Validation](splits/dsifn_val.csv) • [Test](splits/dsifn_test.csv) | [Python file](dataloaders/dsifn.py) |
| EGY-BCD | [Holail et al.](%23) | 2023 | [GitHub](https://github.com/oshholail/EGY-BCD) | [Train](splits/egy_bcd_train.csv) • [Validation](splits/egy_bcd_val.csv) • [Test](splits/egy_bcd_test.csv) | [Python file](dataloaders/egy_bcd.py) |
| LEVIR-CD+ | [Chen and Shi](%23) | 2020 | [GitHub](https://github.com/S2Looking/Dataset/) | [Train](splits/levir_cd_train.csv) • [Validation](splits/levir_cd_val.csv) • [Test](splits/levir_cd_test.csv) | [Python file](dataloaders/levir_cdplus.py) |
| MSBC | [Li et al.](%23) | 2022 | [GitHub](https://github.com/Lihy256/MSCDUnet) | [Train](splits/msbc_train.csv) • [Validation](splits/msbc_val.csv) • [Test](splits/msbc_test.csv) | [Python file](dataloaders/msbc.py) |
| MSOSCD | [Li et al.](%23) | 2022 | [GitHub](https://github.com/Lihy256/MSCDUnet) | [Train](splits/msoscd_train.csv) • [Validation](splits/msoscd_val.csv) • [Test](splits/msoscd_test.csv) | [Python file](dataloaders/msoscd.py) |
| OMBRIA | [Drakonakis et al.](%23) | 2022 | [GitHub](https://github.com/geodrak/OMBRIA) | [Train](splits/ombria_train.csv) • [Validation](splits/ombria_val.csv) • [Test](splits/ombria_test.csv) | [Python file](dataloaders/ombria.py) |
| Season-Varying CDD | [Lebedev et al.](%23) | 2018 | [Google Drive](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9) | [Train](splits/season_varying_cdd_train.csv) • [Validation](splits/season_varying_cdd_val.csv) • [Test](splits/season_varying_cdd_test.csv) | [Python file](dataloaders/season_varying_cdd.py) |
| SYSU-CD | [Shi et al.](%23) | 2022 | [GitHub](https://github.com/liumency/SYSU-CD) | [Train](splits/sysu_cd_train.csv) • [Validation](splits/sysu_cd_val.csv) • [Test](splits/sysu_cd_test.csv) | [Python file](dataloaders/sysu_cd.py) |

## Model checkpoints & TensorBoard logs

Model checkpoints and TensorBoard logs are available on [HuggingFace](https://huggingface.co/bvlabs/FAIR-EO-CD-benchmark).

| Model | Paper | Year | # Params (M) | GFLOPs | Model Checkpoints | TensorBoard Logs |
| --- | --- | --- | --- | --- | --- | --- |
| BIT | [Chen et al.](https://doi.org/10.1109/TGRS.2021.3095166) | 2021 | 3.50 | 10.88 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| CGNet | [Han et al.](https://doi.org/10.1109/JSTARS.2023.3310208) | 2023 | 33.68 | 87.55 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| ChangeFormerV6 | [Bandara and Patel](https://doi.org/10.1109/IGARSS46834.2022.9883686) | 2022 | 41.03 | 138.77 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| ChangeViT | [Zhu et al.](https://doi.org/10.1016/j.patcog.2025.112539) | 2026 | 20.66 | 26.36 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| CSSM | [Ghazaei et a.](https://doi.org/10.1109/LGRS.2025.3629303) | 2025 | 2.59 | 2.78 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| HRNet | [Sun et al.](https://doi.org/10.1109/CVPR.2019.00584) | 2019 | 21.48 | 9.55 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| SiamCRNN | [Chen et al.](https://doi.org/10.1109/TGRS.2019.2956756) | 2020 | 28.51 | 65.49 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| STANet | [Chen and Shi](https://doi.org/10.3390/rs12101662) | 2020 | 12.21 | 19.15 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| TinyCD | [Codegoni et al.](https://doi.org/10.1007/s00521-022-08122-3) | 2023 | 0.29 | 1.46 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |
| U-Net SiamConc | [Caye Daudt et al.](https://doi.org/10.1109/ICIP.2018.8451652) | 2018 | 40.35 | 19.37 | [Scratch](%23) • [Pre-trained](%23) | [Scratch](%23) • [Pre-trained](%23) |

## Performance

Mean Intersection over Union (mIoU) for models trained from scratch. The best performance is indicated in bold, and the second-best is underlined.

| Dataset/Model | BIT | CGNet | ChangeFormerV6 | ChangeViT | CSSM | HRNet SiamConc | SiamCRNN | STANet | TinyCD | U-Net SiamConc |
|---|---|---|---|---|---|---|---|---|---|---|
| **BANDON** | 0.5015 | 0.4785 | **0.5661** | <ins>0.5545</ins> | 0.4787 | 0.5231 | 0.5511 | 0.4921 | 0.5306 | 0.4840 | 
| **CLCD** | 0.6920 | 0.6593 | 0.6569 | 0.6637 | 0.4596 | 0.6375 | <ins>0.6950</ins> | 0.6120 | 0.6612 | **0.7048** | 
| **DSIFN** | 0.7711 | **0.8109** | 0.7453 | <ins>0.7919</ins> | 0.6829 | 0.6409 | 0.7863 | 0.7224 | 0.7133 | 0.7593 | 
| **EGY-BCD** | 0.7783 | **0.8214** | 0.7594 | 0.7770 | 0.7005 | 0.7619 | <ins>0.7929</ins> | 0.7598 | 0.7619 | 0.7762 | 
| **LEVIR-CD+** | 0.6378 | 0.6412 | 0.6561 | 0.7181 | 0.5785 | 0.6766 | 0.6907 | 0.5449 | <ins>0.7201</ins> | **0.7201** | 
| **MSBC** | 0.8613 | **0.8846** | <ins>0.8730</ins> | 0.8644 | 0.6554 | 0.8172 | 0.8729 | 0.7875 | 0.7809 | 0.8699 | 
| **MSOSCD** | 0.6603 | 0.6748 | 0.6612 | <ins>0.6826</ins> | 0.5330 | 0.6140 | **0.7248** | 0.6094 | 0.6003 | 0.6817 | 
| **OMBRIA** | 0.6438 | 0.6343 | **0.6934** | <ins>0.6632</ins> | 0.6396 | 0.6123 | 0.6302 | 0.5798 | 0.6138 | 0.6302 | 
| **Season-varying CDD** | 0.8896 | 0.6578 | 0.8110 | **0.9141** | 0.7507 | 0.8584 | <ins>0.9133</ins> | 0.8296 | 0.8556 | 0.8887 | 
| **SYSU-CD** | 0.7066 | **0.7646** | 0.7212 | <ins>0.7512</ins> | 0.6909 | 0.6654 | 0.7350 | 0.6709 | 0.6650 | 0.7090 | 

Mean Intersection over Union (mIoU) for models with pre-trained backbones.

| Dataset/Model | BIT | CGNet | ChangeFormerV6 | ChangeViT | CSSM | HRNet SiamConc | SiamCRNN | STANet | TinyCD | U-Net SiamConc |
|---|---|---|---|---|---|---|---|---|---|---|
| **BANDON** | 0.4780 | 0.4623 | 0.5644 | 0.5844 | 0.4787 | 0.5672 | <ins>0.5957</ins> | 0.4979 | 0.5568 | **0.6114** |
| **CLCD** | 0.7104 | 0.5061 | 0.6559 | 0.7101 | 0.4596 | <ins>0.7182</ins> | 0.7088 | 0.5198 | 0.6821 | **0.7521** |
| **DSIFN** | 0.7551 | 0.8396 | 0.7425 | 0.8282 | 0.6829 | 0.8044 | <ins>0.8459</ins> | 0.8075 | 0.7819 | **0.8569** |
| **EGY-BCD** | 0.8065 | 0.7409 | 0.7685 | 0.8066 | 0.7005 | <ins>0.8433</ins> | **0.8445** | 0.8074 | 0.7619 | 0.8338 |
| **LEVIR-CD+** | 0.6690 | 0.5592 | 0.6904 | <ins>0.7317</ins> | 0.5785 | 0.7249 | 0.7250 | 0.7031 | 0.7254 | **0.7478** |
| **MSBC** | 0.8849 | 0.7223 | 0.8613 | 0.8738 | 0.6554 | 0.8892 | **0.8950** | 0.8592 | 0.7762 | <ins>0.8943</ins> |
| **MSOSCD** | 0.7924 | 0.5820 | 0.6798 | 0.7735 | 0.5330 | 0.7750 | <ins>0.8076</ins> | 0.7280 | 0.5749 | **0.8169** |
| **OMBRIA** | 0.6684 | <ins>0.7051</ins> | 0.6998 | 0.6992 | 0.6396 | 0.6952 | 0.6903 | 0.6282 | 0.6593 | **0.7114** |
| **Season-varying CDD** | 0.8946 | 0.7924 | **0.9438** | 0.9243 | 0.7507 | 0.9224 | 0.9334 | 0.8964 | 0.8501 | <ins>0.9407</ins> |
| **SYSU-CD** | 0.7227 | 0.7279 | 0.7347 | **0.7774** | 0.6909 | 0.7166 | 0.7508 | 0.6642 | 0.7298 | <ins>0.7656</ins> |

## Citation
