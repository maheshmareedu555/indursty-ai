import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YOLOv11Inspector:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Fallback to standard yolo11n.pt
            self.model = YOLO("yolo11n.pt")

    def train_model(self, data_yaml, epochs=5, imgsz=512, batch=16, project="runs", name="yolo11_defect"):
        """
        Trains YOLOv11 model on the dataset specified in data_yaml.
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            exist_ok=True,
            plots=True
        )
        return results

    def predict(self, image_input, conf=0.25):
        """
        Performs object detection on image input.
        Returns:
            annotated_img (ndarray): Image with bounding box overlays
            boxes (list): List of detected bounding boxes dict [{box, conf, cls, label}]
            is_defective (bool): True if defect box detected
        """
        if isinstance(image_input, (str, Path)):
            img_cv = cv2.imread(str(image_input))
        elif isinstance(image_input, np.ndarray):
            img_cv = image_input.copy()
        else:
            img_cv = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        results = self.model.predict(source=image_input, conf=conf, verbose=False)
        res = results[0]
        
        annotated_img = img_cv.copy()
        boxes = []
        
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            confidence = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            
            # Draw glowing bounding box for defect
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 242, 254), 2)
            cv2.putText(annotated_img, f"defect {confidence*100:.1f}%", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 2)
            
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 4),
                "class_id": 0,
                "label": "defect"
            })

        is_defective = len(boxes) > 0
        return annotated_img, boxes, is_defective
