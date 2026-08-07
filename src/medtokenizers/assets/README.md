# Bundled assets

## `t1w_brainweb.nii.gz`

A simulated T1-weighted brain volume, bundled so the examples and tests run
without a download. It is **not** a scan of a human subject: it is the output of
an MRI simulator applied to a digital brain phantom, so no patient data,
consent, or de-identification question arises.

| Property | Value |
| --- | --- |
| Source | BrainWeb Simulated Brain Database, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University |
| URL | https://brainweb.bic.mni.mcgill.ca/brainweb/ |
| Original file | `t1_icbm_normal_1mm_pn3_rf20.mnc` (normal anatomical model, T1 modality, 1 mm isotropic, 3% noise, 20% intensity non-uniformity) |
| Anatomy | Normal. There is no pathology in this volume. |
| Shape | 181 x 217 x 181, 1 mm isotropic, ICBM space |
| Orientation | RAS |
| Stored as | uint8, gzip-compressed NIfTI-1 |

### Local modifications

Converted from MINC to NIfTI-1, reoriented to closest-canonical RAS, and
rescaled so the 99.9th intensity percentile maps to 255 and stored as uint8.
The conversion is lossy in intensity precision only; the spatial grid, voxel
size, and affine are unchanged. Downstream code min-max normalises to [0, 1],
so the 8-bit quantisation does not affect the example or test behaviour.

### Terms

BrainWeb does not publish a formal licence or an explicit redistribution
grant. It states a citation requirement, and this file is redistributed here on
that use-with-acknowledgement basis. If you use it, cite the sources BrainWeb
asks for:

- Cocosco, Kollokian, Kwan, Evans. "BrainWeb: Online Interface to a 3D MRI
  Simulated Brain Database." NeuroImage, 1997.
- Kwan, Evans, Pike. "MRI simulation-based evaluation of image-processing and
  classification methods." IEEE Transactions on Medical Imaging, 1999.
- Kwan, Evans, Pike. "An Extensible MRI Simulator for Post-Processing
  Evaluation." Visualization in Biomedical Computing, 1996.
- Collins et al. "Design and Construction of a Realistic Digital Brain
  Phantom." IEEE Transactions on Medical Imaging, 1998.
- https://brainweb.bic.mni.mcgill.ca/brainweb/

This asset is not covered by the repository's MIT licence, which applies to the
source code.
