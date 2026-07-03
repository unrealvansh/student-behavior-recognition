import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class BehaviorClassifier:
    def __init__(self, weights_path=None, num_classes=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = ["Sitting", "Sleeping", "Mobile", "Reading", "HandRaising"]
        
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model = self.model.to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, crop):
        if crop.size == 0:
            return "Unknown", 0.0
        
        rgb = T.ToPILImage()(crop).convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        return self.class_names[pred.item()], conf.item()