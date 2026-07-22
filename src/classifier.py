import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class BehaviorClassifier:
    def __init__(self, weights_path=None, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        # Must match ImageFolder alphabetical order:
        # hand_raising, mobile, reading, sitting, sleeping
        self.class_names = ["HandRaising", "Mobile", "Reading", "Sitting", "Sleeping"]

        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        self.model = self.model.to(self.device).eval()

        # Transform expects a numpy BGR crop (from cv2)
        self.transform = T.Compose([
            T.ToPILImage(),       # numpy BGR -> PIL
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, crop, return_all_probs=False):
        """Classify a cropped person image.

        Args:
            crop: numpy BGR image (from cv2, e.g. frame[y1:y2, x1:x2]).

        Returns:
            (behavior_name, confidence) tuple if return_all_probs is False.
            (behavior_name, confidence, all_probs_dict) if True.
        """
        if crop.size == 0:
            return ("Unknown", 0.0, {}) if return_all_probs else ("Unknown", 0.0)

        # Convert BGR -> RGB for the model
        import cv2
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred = torch.max(probs, 0)

        best_class = self.class_names[pred.item()]
        best_conf = conf.item()

        if return_all_probs:
            all_probs = {self.class_names[i]: probs[i].item() for i in range(self.num_classes)}
            return best_class, best_conf, all_probs
        
        return best_class, best_conf