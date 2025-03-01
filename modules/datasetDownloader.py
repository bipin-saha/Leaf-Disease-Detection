import os
from roboflow import Roboflow

class RoboflowDatasetDownloader:
    """
    A class to handle the downloading of a YOLOv11 dataset from Roboflow.
    """
    
    def __init__(self, workspace: str, project_name: str, version_number: int):
        """
        Initializes the downloader with workspace, project name, and version number.
        
        :param workspace: The workspace ID on Roboflow.
        :param project_name: The project name on Roboflow.
        :param version_number: The dataset version to download.
        """
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if self.api_key is None:
            raise ValueError("Error: ROBOFLOW_API_KEY environment variable not set.")
        
        self.rf = Roboflow(api_key=self.api_key)
        self.workspace = workspace
        self.project_name = project_name
        self.version_number = version_number
    
    def download(self, format_type: str = "yolov11"):
        """
        Downloads the dataset in the specified format.
        
        :param format_type: The format in which the dataset should be downloaded (default is YOLOv11).
        :return: The downloaded dataset object.
        """
        project = self.rf.workspace(self.workspace).project(self.project_name)
        version = project.version(self.version_number)
        dataset = version.download(format_type)
        return dataset
