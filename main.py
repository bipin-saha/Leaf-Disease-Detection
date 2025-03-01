import os
from modules.trainer import YOLOTrainer
from modules.inference import YOLOValidator
from modules.ensembleInfer import YOLOEnsemble

# Step 1: Train YOLO Models
# Instantiate YOLOTrainer for version 11 model
trainer_v11 = YOLOTrainer(
    dataset_name="medicalimage-z4t5b", 
    workspace="mdtech", 
    version=1, 
    base_model="yolo11m.pt", #Other models: yolo11n.pt, yolo11s.pt, yolo11l.pt, yolo11xl.pt
    epochs=100, 
    batch_size=16, 
    do_train=False  # Set to False if models are already trained
)
result_v11 = trainer_v11.run()

# Instantiate YOLOTrainer for version 12 model
trainer_v12 = YOLOTrainer(
    dataset_name="medicalimage-z4t5b", 
    workspace="mdtech", 
    version=1, 
    base_model="yolo12m.pt", #Other models: yolo12n.pt, yolo12s.pt, yolo12l.pt, yolo12xl.pt
    epochs=100, 
    batch_size=16, 
    do_train=False # Set to False if models are already trained
)
result_v12 = trainer_v12.run()

# Step 2: Define Paths to Trained YOLO Models and Dataset Configuration
dataset_yaml_path = os.path.join(os.getcwd(), "mdtech-1/data.yaml")  # Path to dataset configuration file
model_v11_path = os.path.join(os.getcwd(), "models/yolo_v11.pt")  # Path to trained YOLO v11 model
model_v12_path = os.path.join(os.getcwd(), "models/yolo_v12.pt")  # Path to trained YOLO v12 model

# Step 3: Validate YOLO Models
# Validate Model V11
validator_v11 = YOLOValidator(model_v11_path, dataset_yaml_path)
metrics_v11 = validator_v11.validate()
print("Evaluation Metrics for Model V11:")
print(metrics_v11)

# Validate Model V12
validator_v12 = YOLOValidator(model_v12_path, dataset_yaml_path)
metrics_v12 = validator_v12.validate()
print("Evaluation Metrics for Model V12:")
print(metrics_v12)

# Step 4: Perform Ensemble Inference
# Define the list of trained model paths for ensemble inference
model_paths = [model_v11_path, model_v12_path]
yaml_path = dataset_yaml_path

# Instantiate YOLOEnsemble for combining predictions from both models
detector = YOLOEnsemble(model_paths, yaml_path, iou_threshold=0.7)

# Process and visualize images with ensemble inference
detector.process_images()