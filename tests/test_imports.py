import torch
import cv2
from ultralytics import YOLO
from flask import Flask
from flask_socketio import SocketIO
import scipy

print(f"PyTorch: {torch.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Ultralytics: {YOLO}")
print(f"SciPy: {scipy.__version__}")
print("All imports successful!")