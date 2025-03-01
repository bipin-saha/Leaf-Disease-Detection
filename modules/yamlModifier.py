import os
import yaml

class DatasetConfigModifier:
    """
    A class to modify dataset configuration YAML files.
    """
    
    def __init__(self, file_path: str):
        """
        Initializes the DatasetConfigModifier with a YAML file path.
        
        :param file_path: The path to the YAML file that needs modification.
        """
        self.file_path = file_path
    
    def modify_yaml(self, use_test: bool = True):
        """
        Modifies the YAML file by updating paths for train, validation, and optionally test datasets.
        
        :param use_test: If True, validation set is set to test images; otherwise, to valid images.
        """
        # Load the existing YAML file
        with open(self.file_path, 'r') as file:
            data = yaml.safe_load(file)  # Safely parse YAML content
        
        # Get current working directory and define the dataset path
        content_path = os.path.join(os.getcwd(), "mdtech-1")
        
        # Update dataset paths in the YAML file
        data["train"] = f"{content_path}/train/images"  # Path to training images
        data["val"] = f"{content_path}/test/images" if use_test else f"{content_path}/valid/images"
        if use_test:
            data["test"] = f"{content_path}/test/images"  # Path to test images (optional)
        
        # Write the modified data back to the YAML file
        with open(self.file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)  # Save YAML with human-readable formatting


# if __name__ == "__main__":    
#     yaml_modifier = DatasetConfigModifier("/content/mdtech-1/data.yaml")
#     yaml_modifier.modify_yaml(use_test=True)