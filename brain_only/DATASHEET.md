# Brain-Only Dataset — Technical Datasheet

## Overview

| Property | Value |
|---|---|
| Total subjects | 130 |
| Subjects with imaging data | 107 |
| Subjects with no NIfTI files | 23 |
| Total folder size | ~1.4 GB |
| Source dataset | Forithmus/MR-RATE-coreg (Hugging Face) |

---

## Folder Structure

```
brain_only/
├── {STUDY_UID}/               # one folder per subject (130 total)
│   ├── brain_mask.nii.gz      # binary brain mask
│   ├── t1_brain.nii.gz        # T1-weighted, skull-stripped
│   ├── t2_axi_brain.nii.gz    # T2-weighted axial, skull-stripped
│   ├── t2_cor_brain.nii.gz    # T2-weighted coronal, skull-stripped
│   └── flair_brain.nii.gz     # FLAIR, skull-stripped
├── reports.csv                # clinical reports for all 130 subjects
└── DATASHEET.md               # this file
```

---

## File Descriptions

### `brain_mask.nii.gz`
Binary mask delineating the brain region. Voxels inside the brain = 1, outside = 0.
Used to skull-strip the other modalities. Present in 107/130 subjects.

### `t1_brain.nii.gz`
T1-weighted MRI after skull-stripping using the brain mask.
T1 provides good grey/white matter contrast and is the anatomical reference sequence.
Present in 107/130 subjects.

### `t2_axi_brain.nii.gz`
T2-weighted MRI acquired in the **axial** plane, skull-stripped.
T2 highlights fluid and oedema (hyperintense). Axial is the standard clinical orientation.
Present in 99/130 subjects.

### `t2_cor_brain.nii.gz`
T2-weighted MRI acquired in the **coronal** plane, skull-stripped.
Coronal T2 is particularly useful for hippocampal and temporal lobe assessment.
Present in 76/130 subjects.

### `flair_brain.nii.gz`
Fluid-Attenuated Inversion Recovery MRI, skull-stripped.
FLAIR suppresses CSF signal, making periventricular and cortical lesions more conspicuous.
Present in 90/130 subjects.

---

## NIfTI File Properties

All files are in **NIfTI-1 gzip-compressed** format (`.nii.gz`).

| Modality | Subjects | Avg size | Size range | Most common shape (X×Y×Z) | Mean voxel size (mm) |
|---|---|---|---|---|---|
| brain_mask | 107 | 0.1 MB | 0.0–0.7 MB | 512×512×20 | 0.65×0.62×5.59 |
| t1_brain | 107 | 3.2 MB | 0.4–37.2 MB | 512×512×20 | 0.65×0.62×5.59 |
| t2_axi_brain | 99 | 4.6 MB | 0.9–38.1 MB | 512×512×20 | 0.66×0.63×5.70 |
| t2_cor_brain | 76 | 2.6 MB | 0.9–19.6 MB | 512×512×20 | 0.63×0.63×6.44 |
| flair_brain | 90 | 4.1 MB | 0.9–27.1 MB | 512×512×20 | 0.65×0.61×6.06 |

**Notes:**
- In-plane resolution is high (~0.6×0.6 mm) but slice thickness is thick (~5.5–6.5 mm) — typical of clinical 2D MRI protocols.
- Volume shape is variable; 512×512×20 is the most common but not universal.

---

## Modality Availability per Subject

| Modality combination | Subjects |
|---|---|
| T1 + T2-axi + T2-cor + FLAIR + mask (full set) | 74 |
| T1 + T2-axi + FLAIR + mask | 16 |
| T1 + T2-axi + mask | 9 |
| T1 + mask only | 6 |
| T1 + T2-cor + mask | 2 |
| No NIfTI files (folder exists, report only) | 23 |

---

## `reports.csv`

Clinical radiology reports for all 130 subjects.

| Column | Description |
|---|---|
| `study_uid` | Unique study identifier — matches the subject folder name |
| `report` | Full free-text radiology report |
| `clinical_information` | Clinical indication / reason for scan |
| `technique` | MRI acquisition protocol description |
| `findings` | Structured findings section of the report |
| `impression` | Radiologist summary / conclusion |

File size: ~500 KB. All 130 subjects have a report entry.

---

## Known Limitations

- **23 subjects** have a folder and report but no NIfTI imaging files — likely failed co-registration or missing source data.
- **Slice thickness is thick** (~5–6 mm): volumes are anisotropic and not suitable for direct 3D analysis without resampling.
- **Variable modality availability**: downstream models must handle missing modalities gracefully.
- Images are **skull-stripped** — original full-head volumes are not included here.
