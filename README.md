<h1 align="center">Compute macro-structural statistics of tracts</h1>

<p align="center">
  Streamline count, length, curvature, centroid orientation and volume for every tract in a white matter classification, computed with DIPY into one CSV.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.439"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.439-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="MIT license"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Computes per-tract macrostructure (streamline count, mean length, mean curvature, voxel count, centroid orientation) from a classified tractogram with DIPY, as a CSV.**

For each tract in the classification, computes streamline count, average length, mean curvature and mean orientation of the QuickBundles centroid, and the number of voxels the tract occupies (density map on the DWI grid), and writes tractmeasures.csv. Use it to compare tract shape and size across subjects.

| | |
|---|---|
| **Inputs** | `neuro/dwi` · `neuro/wmc` · `neuro/track/tck` |
| **Outputs** | `neuro/tractmeasures` |
| **Container** | `brainlife/pyafq:dev` |
| **Resources** | 1 CPUs |
| **Run it** | [doi.org/10.25663/brainlife.app.439](https://doi.org/10.25663/brainlife.app.439) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `dwi` | `neuro/dwi` | yes | DWI image; only its grid and affine are used as the reference space. |
| `wmc` | `neuro/wmc` | yes | White matter classification (classification.mat) assigning streamlines to tracts. |
| `track` | `neuro/track/tck` | yes | Tractogram (.tck) the classification refers to. Must match the classification and be in the DWI space. |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `tractmeasures` | `neuro/tractmeasures` | tractmeasures.csv with one row per tract: streamline_count, average_length, average_curvature, voxel_count and centroid_x/y/z (mean orientation of the tract centroid). |

<!-- brainlife-app:end -->

## Running the App

The App runs from the [`v1.0`](https://github.com/brainlife/app-compute-tract-stats/tree/v1.0) branch of this repository.

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.439](https://doi.org/10.25663/brainlife.app.439) and choose, for one subject:

- a **DWI** (`neuro/dwi`); only its voxel grid and affine are used, as the reference space;
- a **white matter classification** (`neuro/wmc`) that assigns streamlines to tracts;
- the **tractogram** (`neuro/track/tck`) the classification refers to, in the same space as the DWI.

Press **Execute**. The App has no options. The statistics are archived to your project as a `neuro/tractmeasures` dataset tagged `macro`, which can be merged with tract profiles by the [Compile tract macro-structural and profile data](https://doi.org/10.25663/brainlife.app.397) App.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/); DIPY, nibabel and pandas come from the `brainlife/pyafq:dev` container. Without Singularity, install [DIPY](https://dipy.org/), nibabel, SciPy and pandas and run `quickStats.py` directly.

1. Clone this repository and check out the App branch (`git checkout v1.0`).
2. Create `config.json` with the paths to your inputs and the subject ID (on brainlife.io the subject is read from `_inputs`); `config.json_sample` has a full example:

   ```json
   {
       "dwi": "./data/dwi/dwi.nii.gz",
       "tractogram": "./data/track/track.tck",
       "classification": "./data/wmc/classification.mat",
       "_inputs": [ { "meta": { "subject": "subj062" } } ]
   }
   ```

3. Run the App:

   ```bash
   ./main
   ```

### Sample data

Download sample datasets from brainlife.io with the [brainlife CLI](https://github.com/brainlife/cli):

```bash
npm install -g brainlife
bl login
mkdir input
bl dataset download <dataset id>
```

## Outputs

```text
tractmeasures/
└── tractmeasures.csv   one row per tract
```

Columns: `subjectID`, `structureID`, `nodeID`, `streamline_count`, `average_length`, `average_curvature` (mean curvature of the tract's QuickBundles centroid), `voxel_count` (voxels the tract's density map occupies on the DWI grid) and `centroid_x`, `centroid_y`, `centroid_z` (mean orientation of the centroid).

## Troubleshooting

- **`stats computations failed somewhere. please check logs and derivatives`**: `tractmeasures.csv` was not written. Check that the classification was made from this tractogram and that the tractogram is in the same space as the DWI.

## Citing

> [!TIP]
> If you use this App in your work, please cite the App, the software it builds on and brainlife.io (below). GitHub's **Cite this repository** button gives the App citation in APA and BibTeX.

- Compute macro-structural statistics of tracts using Dipy. brainlife.io App. [doi:10.25663/brainlife.app.439](https://doi.org/10.25663/brainlife.app.439)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)
- Garyfallidis E, Brett M, Amirbekian B, et al. DIPY, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics 8:8 (2014).
- Garyfallidis et al. Dipy, a library for the analysis of diffusion MRI data. *Frontiers in Neuroinformatics* 8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Kruper et al. Evaluating the Reliability of Human Brain White Matter Tractometry. *Aperture Neuro* 2021 (2021). [doi:10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669](https://doi.org/10.52294/e6198273-b8e3-4b63-babb-6e6b0da10669)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)

## Authors

- Brad Caron

**Contributors:** Soichi Hayashi

**Project director:** Franco Pestilli

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io is publicly funded, and acknowledging that support helps keep it running. This App was developed with support from Wellcome, the U.S. National Institutes of Health (including the BRAIN Initiative BRAIN CONNECTS program: The Axonal Projectome EXchange, U24NS140384, and the Center for Mesoscale Connectomics, UM1NS132207) and the U.S. National Science Foundation.</sub>

## License

[MIT](LICENSE). Copyright (c) 2020 brainlife.io, The University of Texas at Austin and Indiana University. DIPY is distributed under its own BSD license.
