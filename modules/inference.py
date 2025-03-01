import os
import sys
import torch
from ultralytics import YOLO  # Import the YOLO model from Ultralytics

# Ensure the modules directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.yamlModifier import DatasetConfigModifier  # Custom YAML modifier module

class YOLOValidator:
    def __init__(self, model_path, dataset_yaml_path):
        """
        Initializes the YOLOValidator with the model and dataset paths.
        :param model_path: Path to the trained YOLO model (.pt file)
        :param dataset_yaml_path: Path to the dataset configuration file (.yaml)
        """
        self.model_path = model_path
        self.dataset_yaml_path = dataset_yaml_path
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        # Modify YAML configuration to use the test set
        self.yaml_modifier = DatasetConfigModifier(self.dataset_yaml_path)
        self.yaml_modifier.modify_yaml(use_test=True)
        
        # Load the model
        self.model = YOLO(self.model_path)

    def validate(self):
        """
        Runs validation on the model using the dataset configuration.
        :return: Dictionary containing evaluation metrics.
        """
        metrics = self.model.val(data=self.dataset_yaml_path)
        return metrics

if __name__ == "__main__":
    # Define paths to trained YOLO models and dataset config
    dataset_yaml_path = os.path.join(os.getcwd(), "mdtech-1/data.yaml")
    model_v11_path = "/home/bipin/Bipin/mdtech_assignment/models/yolo_v11.pt"
    model_v12_path = "/home/bipin/Bipin/mdtech_assignment/models/yolo_v12.pt"

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