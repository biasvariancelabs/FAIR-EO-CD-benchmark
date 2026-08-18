## Schema overview

The schema used for semantic enrichment of the benchmarking framework can be seen below. 

<img width="4419" height="3885" alt="fair-eo-core" src="https://github.com/user-attachments/assets/c7579137-c29e-4c00-a893-d3a94ddd5268" />

The schema unifies four complementary vocabularies into a single, cross-referenced structure for describing benchmarking experiments end-to-end — from the datasets and methods used, through to how each run was executed and evaluated.

**Dataset layer — AI4QC.** Describes the training datasets themselves. The latter were semantically annotated using the [AI4QC ontology](https://github.com/biasvariancelabs/AI4QC). Each dataset is linked to a `dcat:Dataset`, connecting this layer to the standard DCAT data catalog vocabulary. Metadata fields include the EO task the datasets support, the spectral bands involved, class definitions, distribution format, any metrics reported for the dataset in the literature, etc. The complete metadata can be found in the [datasets.ttl](datasets.ttl) file.

**Method layer — AiTLAS.** Describes the models and algorithms being benchmarked, using `feo:Method` as the anchor concept. This includes the model's architecture, model type, and the task it supports. The metadata can be found in the [models.ttl](models.ttl) file. 

**Experiment layer — MLDCAT-AP.** Sits at the center of the schema and ties everything together. A `Task` connects to a `Run`, which records the parameters and resource usage of a specific execution. Each run realizes an `MachineLearningModel`.

**Evaluation layer — FAIR-EO.** Captures the results of each run as an `EvaluationResult`, modeled as a `dqv:QualityMeasurement`. Each result specifies the measure used (e.g. F1, IoU, mAP), the data split it was computed on (train/val/test), the aggregation method (micro/macro), and the resulting numeric value.

Together, these layers make it possible to trace a single evaluation result all the way back to the exact dataset, model configuration, and experimental run that produced it — supporting full reproducibility and structured comparison across the benchmark. The metadata of the runs can be found in the [runs.ttl](runs.ttl) file. 
