# YOLO Model Training, Validation, and Ensemble Inference

## Overview

This project implements an end-to-end pipeline for object detection using YOLO models for leaf desease detection. The pipeline includes:

- Training YOLO models (v11 and v12)
- Validating trained models using a test dataset
- Performing ensemble inference using multiple models to improve detection accuracy
- API and user interface for easy web integration

For easy explanation please see under `Notebook/GFL_Notebook.ipynb`

## Project Structure

```
project_root/
│── modules/
    ├── datasetDownloader.py  # Download the dataset from Roboflow
│   ├── trainer.py  # Training module
│   ├── inference.py  # Validation module
│   ├── ensembleInfer.py  # Ensemble inference module
│   ├── yamlModifier.py  # modify data.yaml according to the need.
│── models/
│   ├── yolo_v11.pt  # Trained YOLO v11 model
│   ├── yolo_v12.pt  # Trained YOLO v12 model
│── mdtech-1/
│   ├── data.yaml  # Dataset configuration file
│── main.py  # Main script to run the training and inference visualization pipeline
│── app.py  # Main script to run the application file
│── README.md  # Documentation
```

As the project involves 2 steps, for using application (API & UI) using docker is recommended. And for training and visualization use the main.py upon proper requirements installation.

## Training and Inference Visualization Setup
Use `Notebook/GFL_Notebook.ipynb` for easy interpretation.

### Prerequisites

Ensure you have the following installed:

- Python 3.12

### Clone Repository
```bash
git clone https://github.com/bipin-saha/Leaf-Disease-Detection.git
cd Leaf-Disease-Detection.git
```

### Create Conda Environment

Create a new conda environment with Python 3.12:

```bash
conda create --name leaf-disease-detection python=3.12
conda activate leaf-disease-detection
```

### Install Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

## Usage (main.py)

### Step 1: Train YOLO Models

The script initializes and trains two YOLO models (v11 and v12). If the models are already trained, set `do_train=False`.

```python
trainer_v11 = YOLOTrainer(
    dataset_name="medicalimage-z4t5b",
    workspace="mdtech",
    version=1,
    base_model="yolo11m.pt",
    epochs=100,
    batch_size=16,
    do_train=False  # Set to True to train
)
result_v11 = trainer_v11.run()
```

Repeat the process for `trainer_v12`.

### Step 2: Validate YOLO Models

Each trained model is validated against the dataset using `YOLOValidator`.

```python
validator_v11 = YOLOValidator(model_v11_path, dataset_yaml_path)
metrics_v11 = validator_v11.validate()
print(metrics_v11)
```

### Step 3: Perform Ensemble Inference

Combining multiple YOLO models improves detection accuracy. The script performs ensemble inference using `YOLOEnsemble`. Change `iou_threshold` according to the need.

```python
detector = YOLOEnsemble([model_v11_path, model_v12_path], yaml_path, iou_threshold=0.7)
detector.process_images()
```

## Dataset Configuration (`data.yaml`)

```yaml
names:
- Object_Class_Name
nc: 1  # Number of classes
test: /path/to/test/images
train: /path/to/train/images
val: /path/to/val/images
```


## Usage (app.py)
## Documentation: Running the Application Using Docker

### Prerequisites
1. Ensure Docker is installed and running on your system.
2. Verify that the Dockerfile is located in the project root and correctly configured to use "app.py" as the application entry point.
3. Confirm that any required environment variables or additional configurations are set within the Dockerfile.

### Instructions

1. Build the Docker image:  
    Command:  
    ```bash
    docker build -t leaf_desease .
    ```

2. Run a container from the image:  
    Command:  
    ```bash
    docker run -d -p 5017:5017 leaf_desease  
    ```

    Notes:  
    - The "-d" flag runs the container in detached mode.
    - The port mapping (-p 5017:5017) reflects the port exposed in the Dockerfile. Adjust if necessary.
    - Ensure that the application (app.py) listens on the exposed port 5017.



