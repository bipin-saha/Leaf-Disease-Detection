import os
import cv2
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.ops import nms

class YOLOEnsemble:
    def __init__(self, model_paths, yaml_path, iou_threshold=0.7):
        self.models = [YOLO(model_path) for model_path in model_paths]
        self.yaml_path = yaml_path
        self.iou_threshold = iou_threshold
        self.load_yaml()
        self.image_paths = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir)]

    def load_yaml(self):
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        self.class_names = data.get("names", [])
        self.image_dir = data.get("test", "")

    def predict(self):
        all_results = [model.predict(self.image_paths, iou=self.iou_threshold) for model in self.models]
        return all_results

    @staticmethod
    def extract_detections(results):
        labels_list, scores_list, boxes_list = [], [], []
        for result in results:
            labels = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            labels_list.append(labels)
            scores_list.append(scores)
            boxes_list.append(boxes)
        return labels_list, scores_list, boxes_list

    def combine_results(self, results1, results2):
        labels_v1, scores_v1, boxes_v1 = self.extract_detections(results1)
        labels_v2, scores_v2, boxes_v2 = self.extract_detections(results2)

        combined_labels, combined_scores, combined_boxes = [], [], []
        for i in range(len(labels_v1)):
            if len(scores_v1[i]) == len(scores_v2[i]):
                combined_labels.append(labels_v1[i])
                combined_scores.append((scores_v1[i] + scores_v2[i]) / 2)
                combined_boxes.append(boxes_v1[i])
            else:
                if np.mean(scores_v1[i]) > np.mean(scores_v2[i]):
                    combined_labels.append(labels_v1[i])
                    combined_scores.append(scores_v1[i])
                    combined_boxes.append(boxes_v1[i])
                else:
                    combined_labels.append(labels_v2[i])
                    combined_scores.append(scores_v2[i])
                    combined_boxes.append(boxes_v2[i])
        return combined_labels, combined_scores, combined_boxes

    @staticmethod
    def apply_nms(labels, scores, boxes, iou_threshold=0.5):
        if len(boxes) == 0:
            return [], [], []
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        return [labels[i] for i in keep_indices], [scores[i] for i in keep_indices], [boxes[i] for i in keep_indices]

    def draw_predictions(self, image_path, labels, scores, boxes, save_path=None):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read {image_path}")
            return
        labels, scores, boxes = self.apply_nms(labels, scores, boxes)
        for label, score, box in zip(labels, scores, boxes):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f"Class {label}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"{class_name}: {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def process_images(self, output_dir="ensemble_results"):
        os.makedirs(output_dir, exist_ok=True)
        results = self.predict()
        combined_labels, combined_scores, combined_boxes = self.combine_results(results[0], results[1])
        
        for i, image_path in enumerate(self.image_paths):
            save_path = os.path.join(output_dir, f"ensemble_{i}.jpg")
            processed_image = self.draw_predictions(image_path, combined_labels[i], combined_scores[i], combined_boxes[i], save_path)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(processed_image)
            plt.axis("off")
            plt.show()


# # Example usage
# if __name__ == "__main__":
#     model_paths = ["/content/runs/detect/train7/weights/best.pt", "/content/runs/detect/train8/weights/best.pt"]
#     yaml_path = "/path/to/dataset.yaml"
#     iou_threshold = 0.7
    
#     detector = YOLOEnsemble(model_paths, yaml_path, iou_threshold)
#     detector.process_images()
