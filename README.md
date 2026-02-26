#  RNA 3D Structure Prediction via Geometric Deep Learning & Diffusion Modeling

## Overview

This project explores **geometric deep learning for RNA 3D structure generation**, inspired by the 2025 Stanford RNA 3D Folding challenge.

Instead of directly regressing Cartesian coordinates, the pipeline:

1. Predicts **backbone torsion angles** using an equivariant graph neural network (EGNN)  
2. Trains a **diffusion-based generative model** over torsion space  
3. Reconstructs 3D coordinates via an internal-coordinate decoder  
4. Samples multiple conformations per RNA sequence  

The focus of this repository is not leaderboard optimization, but **research-driven modeling of RNA geometry under limited biological data conditions**.

---

##  Upstream Data & Feature Engineering Pipeline

The models in this repository were trained on a dataset curated through a larger preprocessing pipeline developed during the Kaggle RNA 3D competition.

That upstream pipeline included:

- Temporal leakage control (CASP-style cutoff enforcement)  
- Structural RMSD-based deduplication  
- Multi-conformer structure merging  
- MSA-derived features (PSSM + positional entropy)  
- RNAfold secondary structure integration  
- Graph spectral features  
- Backbone torsion extraction from PDB  
- Coordinate normalization and augmentation  
- Structural motif detection  

---

#  Modeling Philosophy

RNA structure prediction is challenging due to:

- Limited experimentally solved RNA structures  
- High conformational variability  
- Periodic angular representations  
- Global topology-based evaluation (e.g., TM-score)  
- Small dataset regime compared to proteins  

To address this, the modeling pipeline separates:

- **Geometric representation (torsions)**
- **Equivariant learning (EGNN)**
- **Generative diversity (diffusion sampling)**
- **Deterministic geometric reconstruction**

---

#  Architecture

## 1️ Equivariant Graph Neural Network (EGNN)

Located in: `models/egnn.py`

Key properties:

- Radius graph construction  
- Coordinate-aware message passing  
- E(n)-equivariant coordinate updates  
- Residue-level embeddings  
- Torsion prediction head  

The EGNN serves as the denoising backbone inside the diffusion model.

---

## 2️ Diffusion-Based Torsion Model

Located in: `models/diffusion_model.py`

Features:

- Linear beta schedule (DDPM-style)  
- Forward noising (`q_sample`)  
- Reverse denoising (`p_sample_ddpm`)  
- Timestep conditioning  
- Cyclic angular loss  
- Smoothness regularization  

The model learns to generate plausible torsion configurations via reverse diffusion.

Torsions are predicted in **degrees**, using a cosine-based cyclic loss to respect angular periodicity.

---

## 3️ Internal Coordinate Decoder

Located in: `utils/internal_coordinate_decoder.py`

This module:

- Converts torsion angles → Cartesian coordinates  
- Uses Rodrigues’ rotation formula  
- Enforces bond-length continuity  
- Produces C1′ backbone traces  

It provides a lightweight geometric reconstruction layer suitable for ML pipelines.

---

#  Training

Training entry point:

`training/train_diffusion.py`

Example:

```bash
python training/train_diffusion.py   --features features/final_merged_features.csv   --torsions features/torsion_labels.csv   --outdir checkpoints   --epochs 20   --batch-size 8
```

What happens:

- Features are grouped per RNA  
- Variable-length sequences are padded  
- Torsions are noised at random timesteps  
- The EGNN denoiser predicts clean torsions  
- Loss = cyclic angle loss + smoothness regularization  
- Model checkpoint is saved  

---

#  Inference & Structure Sampling

Inference entry point:

`inference/sample_structures.py`

Example (generate PDBs):

```bash
python inference/sample_structures.py   --features features/final_merged_features.csv   --checkpoint checkpoints/diffusion_denoiser.pt   --outdir outputs   --num-samples 20   --num-final 5   --write-pdb
```

Pipeline:

1. Load trained model  
2. Sample torsions via reverse diffusion  
3. Decode torsions → 3D coordinates  
4. Cluster multiple samples (KMeans)  
5. Select representative conformations  
6. Optionally export PDB files or Kaggle-style submission.csv  



# Experimental Observations

- Diffusion over torsion space enables structural diversity  
- Cyclic loss stabilizes angular prediction  
- Small RNA datasets limit generative stability  
- Torsion MSE does not perfectly align with global topology metrics  
- Reverse diffusion quality is sensitive to schedule design  

This highlights the gap between local angular objectives and global structural evaluation.

---

# Data

The dataset originated from the Stanford RNA 3D Folding Kaggle competition (2025).

Due to Kaggle rules, data is not redistributed here.

To reproduce:

1. Download competition dataset from Kaggle  
2. Place relevant CSVs under a local `features/` directory  
3. Run training and inference scripts as shown above  

---

# Future Directions

- Von Mises / circular distributions for torsion uncertainty  
- Coordinate-level equivariant diffusion  
- Direct TM-score–aware training  
- Distance-matrix consistency regularization  
- Improved alignment between local loss and global topology metrics  

---

# Technical Stack

- PyTorch  
- PyTorch Geometric  
- NumPy / SciPy  
- scikit-learn  
- Diffusion modeling  
- Geometric deep learning  

---

# Project Context

This project represents an independent geometric modeling exploration inspired by a Kaggle competition, emphasizing:

- Structured model design  
- Mathematical consistency  
- Generative diversity  
- Research-oriented experimentation  

It demonstrates:

- Equivariant neural networks  
- Diffusion generative modeling  
- Angular loss design  
- Internal coordinate geometry  
- 3D structure reconstruction pipelines  
