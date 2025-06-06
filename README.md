# Model Overview

This repository contains a suite of Large Language Models (LLMs) designed for high-throughput mining and generation of antimicrobial peptides (AMPs). Each model serves a unique purpose in the discovery and evaluation of potential AMPs:

- **ProteoGPT**: A pre-trained model for generating and analyzing amino acid sequences.
- **AMPGenix**: A sequence generator capable of producing potential AMP sequences.
- **AMPSorter**: A classifier designed to identify AMPs from peptide datasets.
- **BioToxiPept**: A classifier aimed at determining the cytotoxicity of short peptides.

All models and config files can be downloaded from [here](https://drive.google.com/drive/folders/19cOtRtZzU3JAglaRFLbc5M1aMmjYTUgV?usp=drive_link). 

## Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/W1V1995/AMP_Project.git
```

Download the `AMP_models` and place it in the `AMP_Project` directory.

Navigate into the cloned directory:

```bash
cd AMP_Project
```

Ensure that all dependencies are installed by following the installation instructions provided in the `requirements.txt` file or the dedicated installation guide.

## Usage

### Fine-tuning

To create a classifier by fine-tuning, execute the following command:

```bash
sh Fine-tuning_classifier.sh
```

To create a generator by fine-tuning, execute the following command:

```bash
sh Fine-tuning_generator.sh
```

Parameters such as `batch_size`, `epochs`, etc., and output path can be customized.

### AMP Generation

To generate sequences using AMPGenix, run:

```bash
sh AMPGenix.sh
```

Parameters such as `ntokens`, `nsamples`, `prefix`, `model_path`, `save_samples_path`, etc., can be adjusted as per your requirements.

### AMP Prediction

For identifying AMPs from a peptide dataset using AMPSorter, execute:

```bash
sh AMPSorter_predictor.sh
```

Customize parameters including `batch_size`, `raw_data_path`, `model_path`, `classifier_path`, `output_path`, `candidate_amp_path`, etc., to fit your dataset and path.

### Cytotoxicity Prediction

To predict the cytotoxicity of short peptides with BioToxiPept, use:

```bash
sh BioToxiPept.sh
```

Adjustable parameters are `batch_size`, `raw_data_path`, `model_path`, `classifier_path`, `output_path`, `candidate_pep_path`, etc.

### QSAR Prediction

To predict the antimicrobial activity of short peptides based on charged residues and hydrophobic residues, use:

```bash
python QSAR.py
```

Adjustable parameters are `samples_path`,  `output_path`.

## Data Preparation

To utilize AmpSorter or BioToxiPept for predictions, prepare a CSV file containing your sequence data. Ensure the file includes a column named "Sequence". Example format:

```csv
Sequence
<sequence_1>
<sequence_2>
...
```

Save your dataset in the format `/Data/Sequence.csv` or modify the script parameters to point to your custom data path.
