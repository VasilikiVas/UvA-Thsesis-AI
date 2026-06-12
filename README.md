# UvA-Thsesis-AI

Repository for the MSc Artificial Intelligence thesis at the University of Amsterdam: **Language Driven 3D Scene Understanding**.

This project explores multimodal 3D understanding by aligning 3D point-cloud representations with language and image representations. The codebase contains experiments for processing the Amazon Berkeley Objects (ABO) dataset, generating CLIP-based image/text embeddings, training PointNet-style 3D representations, adapting ULIP-style contrastive pretraining, and running language-guided segmentation experiments.

## Thesis

The official thesis PDF is available through UvA DSpace:

- **Title:** [Language Driven 3D Scene Understanding](https://dspace.uba.uva.nl/server/api/core/bitstreams/140e8cb7-1a2f-48f3-a3e0-0aa371aa5df6/content)
- **Author:** Vasiliki I. Vasileiou
- **Programme:** MSc Artificial Intelligence, University of Amsterdam
- **Credits:** 48 EC
- **Project period:** November 2022 – July 2023
- **Submission date:** July 3, 2023
- **Supervisors / advisors:** Prof. Martin R. Oswald and Dr. Silvan Weder
- **Examiner:** Prof. Martin R. Oswald
- **Second reader:** Dr. Ozzy Ülger
- **Research context:** University of Amsterdam / ATLAS Lab
- **Repository copy:** `thesis_ai_vasileiou_lfzs2wax.pdf`

The thesis investigates multimodal 3D scene understanding using features from 2D images, text metadata, and 3D point clouds. The central goal is to retrieve and identify objects in 3D scenes, including properties such as material, color, and shape, while reducing reliance on manually labeled 3D datasets.

## Thesis notes

### Motivation

Traditional 3D scene understanding pipelines often depend on manually annotated 3D data and task-specific training. This thesis studies whether language and 2D visual supervision can help build richer, more flexible 3D representations for open-vocabulary or weakly supervised scene understanding.

### Research questions

The thesis focuses on two main questions:

1. Can visual CLIP embeddings, textual CLIP embeddings, and 3D shape embeddings be fused into a richer joint co-embedding for indoor scene understanding? Which fusion strategy works best?
2. Can the fused representation capture object properties such as color and material, making it useful for applications such as room furnishing or indoor design?

### Data

The experiments primarily use the **Amazon Berkeley Objects (ABO)** dataset because it provides 3D objects, product images, and metadata. Due to the size and noise of the full dataset, the thesis uses a cleaned subset with 12 indoor-object categories:

```text
CHAIR, TABLE, BED, SOFA, DRESSER, DESK,
RUG, LAMP, PILLOW, VASE, PLANTER, CURTAIN
```

The cleaned subset contains **842 training objects** and **120 test objects**, sampled from up to 100 cleaned objects per category. In the thesis, the term “label” refers to the ABO product category rather than a manually assigned object-level label.

### Prompting strategy

A key part of the project is aligning text and image embeddings in CLIP space. Several text prompt variants are compared using image/text correlation matrices. The selected prompt for most experiments is:

```text
a picture of <color> <material> <label>
```

The thesis also explores using additional product images and cropped image regions to capture material and fine-grained details. Simple averaging of multiple image embeddings did not clearly outperform the base setup, so more advanced image-fusion methods, such as attention-based fusion, are left as future work.

### Method summary

The implementation builds on **ULIP**, which learns a unified representation across language, images, and point clouds. The thesis studies several modifications to make ULIP more suitable for object-property-aware retrieval:

- modify the 3D encoder so object-property information can be concatenated into the learned representation;
- test alternative image/text inputs, including RGB images, grayscale images, labels, and richer textual descriptions;
- add a learned merging mechanism that combines point-cloud, image, and text information;
- extend the merging setup with RGB and grayscale image features, labels, full descriptions, and BLIP-generated descriptions.

Across the main experiments, the CLIP image and text encoders are frozen, while the 3D encoder and additional trainable layers are optimized. The thesis reports experiments using **8192 sampled points**, PyTorch, AdamW, A100 GPUs, a batch size of 64, and pretraining runs of roughly 50–100 epochs depending on the experiment.

### Results at a glance

The thesis compares multiple experimental settings using zero-shot 3D classification metrics. Among the comparable experimental sets, the reported best results include:

| Method | Best top-1 accuracy | Best top-5 accuracy |
| --- | ---: | ---: |
| `tryabo_train_842_test_120_label_grey_pretrained` | 89.17 | 99.17 |
| `tryabo_train_842_test_120_label_rgb` | 83.33 | 94.17 |
| `tryabo_train_842_test_120_description_rgb` | 80.00 | 99.17 |
| `tryabo_train_842_test_120_label_grey` | 79.17 | 97.50 |
| `tryabo_train_842_test_120_only_text` | 73.33 | 91.67 |
| `train_842` | 71.67 | 98.33 |

The thesis notes that the pretraining/fine-tuning method achieves the highest accuracy but may not generalize as well to unseen labels because it trains the model twice. The proposed property-aware methods are positioned as a step toward handling duplicate or near-duplicate 3D objects that share shape but differ in material, color, or other metadata.

### Future work from the thesis

The thesis identifies several follow-up directions:

- extend ABO with duplicate or near-duplicate objects to better test color/material-aware retrieval;
- develop an end-to-end image-to-point-cloud retrieval application based on object properties;
- extend the application to full indoor scenes by segmenting objects and retrieving shape-conditioned alternatives with matching materials or colors;
- include additional object properties such as dimensions, which could support indoor design and VR applications.

## Repository contents

```text
.
├── README.md                         # Project overview
├── thesis_ai_vasileiou_lfzs2wax.pdf  # Local copy of the thesis PDF
├── JointSpace/                       # ABO preprocessing and CLIP embedding scripts
├── Pointnet_3D_CLIP/                 # PointNet-based 3D-to-CLIP representation learning
├── ULIP/                             # ULIP-based 3D-language-image pretraining experiments
└── lang-seg/                         # Language-driven 2D/scene segmentation experiments
```

### `JointSpace/`

Scripts and notebooks for preparing a shared embedding space between 3D assets, images, and text:

- `msh_to_ptc.py` converts ABO 3D meshes into sampled point clouds.
- `image_emb.py` extracts CLIP visual embeddings from product images.
- `text_emb.py` builds text prompts from metadata such as object label, color, and material, then extracts CLIP text embeddings.
- `selected_dataset.ipynb` and `visualizations.ipynb` support dataset selection and exploratory analysis.

### `Pointnet_3D_CLIP/`

PointNet-based training pipeline for learning 3D features aligned with CLIP embeddings.

Main files:

- `train_representation.py` trains a PointNet-style model on point clouds and image/text embeddings.
- `test_representation.py` evaluates a trained representation model.
- `data_utils/ABODataLoader.py` loads selected ABO point clouds and corresponding CLIP features.
- `models/pointnet_features.py` defines a PointNet encoder with a 512-dimensional projection head.

### `ULIP/`

Adapted ULIP-style pipeline for 3D-language-image pretraining and evaluation.

Main files:

- `main.py` contains the training and evaluation entry point.
- `requirements.txt` contains ULIP-specific Python dependencies.
- `data/` contains dataset configuration files, including ABO and ModelNet40 configs.
- `models/`, `scripts/`, and `utils/` contain the ULIP model, training scripts, and utilities.

### `lang-seg/`

Language-driven segmentation experiments, including ScanNet-related utilities and LSeg-style training/testing scripts.

Main files:

- `train_lseg.py` trains the language-guided segmentation model.
- `test_lseg.py` evaluates the segmentation model.
- `lseg_app.py` provides an interactive/demo-style entry point.
- `download-scannet.py` and `SensorData.py` support ScanNet data handling.

## Expected data layout

Several scripts currently use absolute UvA/Snellius-style paths such as:

```text
/projects/0/vvasileiou4/ABO/
```

Before running the code locally, update the paths in the relevant scripts or refactor them into a config file/environment variable.

The PointNet pipeline expects a processed ABO subset with a structure similar to:

```text
ABO/
└── selected_962/
    ├── pointcl/        # .txt point-cloud files
    ├── img_emb/        # CLIP visual embeddings, *_vis.pt
    └── txt_emb_desc/   # CLIP text embeddings, *_text.pt
```

The preprocessing scripts also expect access to the original ABO structure, including:

```text
ABO/
├── listings/metadata/
├── 3dmodels/
│   ├── metadata/3dmodels.csv
│   └── original/
└── images/
    ├── metadata/images.csv
    └── original/
```

## Installation

Create an isolated Python environment:

```bash
git clone https://github.com/VasilikiVas/UvA-Thsesis-AI.git
cd UvA-Thsesis-AI

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the shared dependencies used across the project:

```bash
pip install torch torchvision torchaudio
pip install numpy tqdm pillow open3d transformers googletrans==4.0.0-rc1 wandb matplotlib opencv-python
```

For ULIP experiments, also install the ULIP requirements:

```bash
pip install -r ULIP/requirements.txt
```

For `lang-seg`, install the packages listed in `lang-seg/requirements.txt` and any additional dependencies required by the original LSeg/encoding setup.

## Typical workflow

### 1. Prepare ABO point clouds

Update the ABO root path inside `JointSpace/msh_to_ptc.py`, then run:

```bash
python JointSpace/msh_to_ptc.py
```

This samples each 3D mesh into a `.txt` point-cloud file.

### 2. Generate CLIP text and image embeddings

Update the ABO paths in `JointSpace/text_emb.py` and `JointSpace/image_emb.py`, then run:

```bash
python JointSpace/text_emb.py
python JointSpace/image_emb.py
```

The text script uses object metadata to construct prompts such as:

```text
a picture of <color> <material> <label>
```

The image script extracts visual CLIP embeddings from the main and additional product images.

### 3. Train the PointNet-to-CLIP representation model

```bash
cd Pointnet_3D_CLIP
python train_representation.py \
  --gpu 0 \
  --model pointnet_features \
  --batch_size 124 \
  --epoch 400 \
  --num_point 1024 \
  --process_data
```

Training logs and checkpoints are written under `Pointnet_3D_CLIP/log/`.

### 4. Evaluate a trained PointNet model

```bash
cd Pointnet_3D_CLIP
python test_representation.py \
  --gpu 0 \
  --log_dir <experiment-directory> \
  --num_point 1024
```

Replace `<experiment-directory>` with the directory created during training.

### 5. Run ULIP experiments

```bash
cd ULIP
python main.py \
  --pretrain_dataset_name abodata \
  --validate_dataset_name modelnet40 \
  --model ULIP_PN_SSG \
  --npoints 8192 \
  --batch-size 2
```

Use the dataset YAML files under `ULIP/data/` to adapt paths and dataset-specific settings.

### 6. Run language segmentation experiments

```bash
cd lang-seg
python train_lseg.py
python test_lseg.py
```

For ScanNet experiments, first prepare the ScanNet data with the provided download and preprocessing utilities.

## Outputs

Depending on the experiment, outputs may include:

- sampled point clouds (`.txt`),
- CLIP image/text embeddings (`.pt`),
- PointNet checkpoints (`best_model.pth`),
- WandB training logs,
- segmentation predictions and visualizations,
- experiment logs under module-specific `log/`, `logs/`, or `outputs/` directories.

## Reproducibility notes

- Many scripts currently contain hard-coded absolute paths. Update those paths before running.
- Some scripts are designed for GPU/HPC execution and may require CUDA, Slurm job scripts, or project-specific storage paths.
- The repository includes notebooks and experimental code; exact thesis results may require the same dataset subset, random seeds, checkpoints, and environment used during the original experiments.
- Large datasets such as ABO and ScanNet are not included in this repository.

## Citation

If you use this repository in academic work, please cite the thesis:

```bibtex
@mastersthesis{vasileiou2023language3d,
  title  = {Language Driven 3D Scene Understanding},
  author = {Vasileiou, Vasiliki I.},
  school = {University of Amsterdam},
  year   = {2023},
  type   = {MSc Thesis},
  url    = {https://dspace.uba.uva.nl/server/api/core/bitstreams/140e8cb7-1a2f-48f3-a3e0-0aa371aa5df6/content}
}
```

## License

This repository does not currently include a top-level license file. Some subdirectories contain code adapted from external projects and may be governed by their original licenses. In particular, the `ULIP/` directory includes its own license file. Add a top-level license before distributing or reusing the full repository.

## Acknowledgements

This repository builds on work and ideas from CLIP, PointNet, ULIP, LSeg, BLIP, and related language-driven 2D/3D vision research. Dataset-specific experiments use resources such as ABO, ModelNet40, and ScanNet.

The thesis was supervised by Prof. Martin R. Oswald and Dr. Silvan Weder, examined by Prof. Martin R. Oswald, and second-read by Dr. Ozzy Ülger at the University of Amsterdam / ATLAS Lab.
