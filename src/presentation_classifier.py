"""
A separate, specialized classifier class for the "Presentation Model".
This class defines the model architecture to exactly match the one used in
'train_presentation.py' (a simple nn.Linear head).
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import cv2

class PadToSquare(object):
    """Pads a PIL Image to a square."""
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        hp = (max_dim - w) // 2
        vp = (max_dim - h) // 2
        padding = (hp, vp, max_dim - w - hp, max_dim - h - vp)
        return T.functional.pad(img, padding, fill=128)

class PresentationBehaviorClassifier:
    def __init__(self, weights_path=None, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = ["HandRaising", "Mobile", "Reading", "Sitting", "Sleeping"]

        # Define the model architecture to match train_presentation.py
        self.model = mobilenet_v2(weights=None) # No pretrained weights needed for loading state_dict
        self.model.classifier = nn.Linear(self.model.last_channel, num_classes)

        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        self.model = self.model.to(self.device).eval()

        # Define the simple transform used for the presentation model
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, crop, return_all_probs=False):
        """Classifies a cropped person image."""
        if crop.size == 0:
            return ("Unknown", 0.0, {}) if return_all_probs else ("Unknown", 0.0)

        # Convert BGR (from cv2) to RGB for the model
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
