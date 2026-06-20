# cpg0000 test data

One site of real Cell Painting images from the [cpg0000-jump-pilot](https://github.com/broadinstitute/cellpainting-gallery) dataset, used in integration tests.

**Source:** `s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/` (public, no auth required)

**Plate:** BR00116991 | **Well:** A01 (r01c01) | **Site:** 1 (f01)

## Files

| File | Channel | Dye |
|---|---|---|
| `r01c01f01p01-ch5sk1fk1fl1.tiff` | DNA | HOECHST 33342 |
| `r01c01f01p01-ch4sk1fk1fl1.tiff` | ER | Alexa 488 |
| `r01c01f01p01-ch3sk1fk1fl1.tiff` | RNA | 488 long |
| `r01c01f01p01-ch1sk1fk1fl1.tiff` | AGP | Alexa 647 |
| `r01c01f01p01-ch2sk1fk1fl1.tiff` | Mito | Alexa 568 |
| `Nuclei.csv` | — | CellProfiler nuclei measurements (109 cells) |

Images are 1080×1080 uint16 TIFF.
`Nuclei.csv` is the raw CellProfiler output; the integration test converts `AreaShape_Center_X/Y` to `Nuclei_Location_Center_X/Y` for DeepProfiler.

## Citation

Chandrasekaran et al., 2024. Jump cell painting dataset. [https://doi.org/10.1038/s41592-024-02241-6](https://doi.org/10.1038/s41592-024-02241-6)
