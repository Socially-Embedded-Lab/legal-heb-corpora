# MOJ Project

## Overview

The **MOJ Project** is a pipeline for processing, tagging, and training machine learning models on legal verdicts. This document outlines the project’s structure, organization, and the necessary steps to execute it.

---

## 📁 Project Structure

### 1. `flow/`
Contains the main training script for running [SETFIT](https://huggingface.co/docs/setfit/) models on labeled legal data.

### 2. `resources/`
This folder includes configuration files and training datasets.

#### a. `configs/`  
Contains 3 YAML configuration files:
- `main_config.yaml` – Defines the legal domain (e.g., drugs, weapons) for the current task.
- `drugs_sentence_cls.yaml` – Specifies paths for saving models/results, label settings, training parameters for drug-related verdicts, etc.
- `weapon_sentence_cls.yaml` – Same as above, tailored for weapon domain.

#### b. `data/`  
Includes training datasets in Pickle format.

The files were created using the `scripts/sentence_classification/training_data_preparation.ipynb notebook`, based on data located in the `resources/data directory`.

### 3. `scripts/`
Contains essential training scripts:
- Data preparation
- Model training
- Saving models
- Metric evaluation and results

### 4. `utils/`
A utility module with helper functions and classes for:
- File I/O
- Logging
- Metric evaluation

This module promotes modularity and reduces redundancy across pipeline stages.

---

## 🚀 Running the Project

### ✅ Prerequisites

- Make sure all dependencies are installed.
- Refer to `requirements.txt` for Python packages and setup instructions.

### 🧪 Step-by-Step Execution

1. **Prepare the Data**  
   In this section, we create the Pickle files used for training, based on the data located in `resources/data directory`.

   Open the Jupyter notebook: `Scripts/sentence_classification/training_data_prepararion.ipynb`.
   - Choose your domain (`drugs` or `weapon`).
   - Set a custom `experiment_name`.
   - Run the relevant cells to prepare the data.

2. **Update Configurations**  
   Edit the domain-specific config file in `resources/configs/{domain}`:
   - Set `experiment_name` (used for model storage paths).
   - Set `data_folder` in `data_path` variable to match your chosen experiment name as you choose in section 1.
   - Adjust `save_dir` and `save_model_path` as needed.

3. **Train the Model**  
   Run the training script:  
   `flow/train_sentence_cls.py`

4. **View the Results**  
Trained models and results will be saved in the path you specified under `save_dir` and `save_model_path` in the configuration file.

---

## 📝 Additional Notes

- Inline comments within scripts and configuration files provide further guidance.
- Always verify configuration values before executing scripts.

---

For further assistance, contact the project maintainer.
