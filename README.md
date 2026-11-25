<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

**Computes per-tract macrostructure (streamline count, mean length, mean curvature, voxel count, centroid orientation) from a classified tractogram with DIPY, as a CSV.**

For each tract in the classification, computes streamline count, average length, mean curvature and mean orientation of the QuickBundles centroid, and the number of voxels the tract occupies (density map on the DWI grid), and writes tractmeasures.csv. Use it to compare tract shape and size across subjects.

Run it on brainlife.io: [https://doi.org/10.25663/brainlife.app.439](https://doi.org/10.25663/brainlife.app.439)

| Input | Datatype | Required | Description |
|---|---|---|---|
| `dwi` | `neuro/dwi` | yes | DWI image; only its grid and affine are used as the reference space. |
| `wmc` | `neuro/wmc` | yes | White matter classification (classification.mat) assigning streamlines to tracts. |
| `track` | `neuro/track/tck` | yes | Tractogram (.tck) the classification refers to. Must match the classification and be in the DWI space. |

| Output | Datatype | Description |
|---|---|---|
| `tractmeasures` | `neuro/tractmeasures` | tractmeasures.csv with one row per tract: streamline_count, average_length, average_curvature, voxel_count and centroid_x/y/z (mean orientation of the tract centroid). |

Runtime: container `brainlife/pyafq:dev`; 1 CPUs.

Please cite:
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)
- Garyfallidis E, Brett M, Amirbekian B, et al. DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics 8:8 (2014).

<!-- brainlife-app:end -->

# app-compute-tract-stats