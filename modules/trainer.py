import os
import sys
import torch
from ultralytics import YOLO
from dotenv import load_dotenv
from modules.datasetDownloader import RoboflowDatasetDownloader
from modules.yamlModifier import DatasetConfigModifier



# Load environment variables
load_dotenv()

class YOLOTrainer:
    def __init__(self, dataset_name, workspace, version, base_model, epochs=100, batch_size=16, img_size=640, device=None, do_train=False):
        self.dataset_name = dataset_name
        self.workspace = workspace
        self.version = version
        self.base_model = base_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.dataset_yaml_path = None
        self.do_train = do_train
    
    def download_dataset(self):
        downloader = RoboflowDatasetDownloader(self.dataset_name, self.workspace, self.version)
        dataset = downloader.download()
        self.dataset_yaml_path = os.path.join(os.getcwd(), f"{self.workspace}-{self.version}/data.yaml")
        return dataset
    
    def modify_yaml(self, use_test=False):
        if self.dataset_yaml_path:
            yaml_modifier = DatasetConfigModifier(self.dataset_yaml_path)
            yaml_modifier.modify_yaml(use_test=use_test)
    
    def train(self):
        if not self.dataset_yaml_path:
            raise ValueError("Dataset YAML path is not set. Please download the dataset first.")
        
        model = YOLO(self.base_model)
        result = model.train(
            data=self.dataset_yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            device=self.device
        )
        return result
    
    def run(self):
        self.download_dataset()
        self.modify_yaml(use_test=False)
        if self.do_train:
            return self.train()

if __name__ == "__main__":
    trainer_v11 = YOLOTrainer(dataset_name="medicalimage-z4t5b", workspace="mdtech", version=1, base_model="yolo11n.pt", epochs=3, batch_size=2, do_train=True)
    result_v11 = trainer_v11.run()
    trainer_v12 = YOLOTrainer(dataset_name="medicalimage-z4t5b", workspace="mdtech", version=1, base_model="yolo12n.pt", epochs=3, batch_size=2, do_train=True)
    result_v12 = trainer_v12.run()
