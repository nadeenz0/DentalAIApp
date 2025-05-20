import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms, models
import gradio as gr
from PIL import Image
import random
import time
from tqdm import tqdm
import warnings


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Config parameters
class Config:
    DATA_PATH = "/kaggle/input/dataset/dataset"
    IMG_SIZE = 224  # Standard size for most pretrained models
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "C:/Users/DELL/Downloads/best_model (1).pth"
    CLASS_NAMES = [
        "Dental benign tumors",
        "Dental Caries",
        "Dental Malignant tumors",
        "Gingivitis",
        "Hypodontia",
        "Mouth Ulcer",
        "Tooth Discoloration"
    ]
    NUM_CLASSES = len(CLASS_NAMES)
######################################################################################################################################################################


class DentalModel(nn.Module):
    def __init__(self, num_classes=7, model_name="efficientnet_b2"):
        super(DentalModel, self).__init__()
        self.model_name = model_name
        
        # Choose model based on name
        if model_name.startswith("efficientnet"):
            # EfficientNet has better performance and efficiency for medical imaging tasks
            self.model = getattr(models, model_name)(weights="DEFAULT")
            if model_name == "efficientnet_b0":
                num_ftrs = 1280
            elif model_name == "efficientnet_b1":
                num_ftrs = 1280
            elif model_name == "efficientnet_b2":
                num_ftrs = 1408
            else:
                num_ftrs = self.model.classifier[1].in_features
                
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name.startswith("resnet"):
            self.model = getattr(models, model_name)(weights="DEFAULT")
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    def freeze_layers(self, freeze_percentage=0.7):
        """Freeze a percentage of the early layers"""
        if self.model_name.startswith("efficientnet"):
            # Get total number of layers in features
            total_layers = len(list(self.model.features))
            freeze_layers = int(total_layers * freeze_percentage)
            
            # Freeze the first X% layers
            for i, param in enumerate(self.model.features.parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
        elif self.model_name.startswith("resnet"):
            # Freeze early layers (layer1, layer2) but keep layer3, layer4 and fc trainable
            for name, param in self.model.named_parameters():
                if "layer3" not in name and "layer4" not in name and "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)
######################################################################################################################################################################

class Inferencer:
    def __init__(self, model_path, config):
        self.config = config
        self.device = config.DEVICE
        
        # Load model
        self.model = DentalModel(num_classes=config.NUM_CLASSES, model_name="efficientnet_b2")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms for inference
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """Make prediction on a single image"""
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Apply transforms
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        class_name = self.config.CLASS_NAMES[predicted_class]
        
        # Create a dict of all class probabilities
        all_probs = {self.config.CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return class_name, confidence, all_probs
    
    def create_gradio_interface(self):
        """Create a Gradio interface for interactive predictions"""
        def predict_image(input_img):
            class_name, confidence, all_probs = self.predict(input_img)
            result = f"Prediction: {class_name}\nConfidence: {confidence:.2%}"
            
            # Create bar chart of probabilities
            plt.figure(figsize=(10, 6))
            names = list(all_probs.keys())
            values = list(all_probs.values())
            
            # Sort by probability
            sorted_indices = np.argsort(values)[::-1]
            names = [names[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]
            
            # Plot horizontal bar chart
            bars = plt.barh(names, values)
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{values[i]:.2%}', va='center')
            
            plt.xlabel('Probability')
            plt.title('Class Probabilities')
            plt.tight_layout()
            
            # Save the plot
            prob_chart_path = "probability_chart.png"
            plt.savefig(prob_chart_path)
            plt.close()
            
            return input_img, result, prob_chart_path
            
        # Create the interface
        iface = gr.Interface(
            fn=predict_image,
            inputs=gr.Image(type="pil"),
            outputs=[
                gr.Image(type="pil", label="Input Image"),
                gr.Textbox(label="Prediction"),
                gr.Image(type="filepath", label="Probability Distribution")
            ],
            title="Dental Image Classification",
            description="Upload a dental image to classify it into one of seven categories: Dental benign tumors, Dental Caries, Dental Malignant tumors, Gingivitis, Hypodontia, Mouth Ulcer, or Tooth Discoloration.",
            examples=[
                # You can add example images here if available
            ]
        )
        return iface
    ######################################################################################################################################################################
    # Main execution
def main():

    print("Starting Dental Image Classification Project...")
    config = Config()
    
    # Check for GPU
    print(f"Using device: {config.DEVICE}")
    
   
    
    # Initialize model
    print("Initializing EfficientNet-B2 model...")
    model = DentalModel(num_classes=config.NUM_CLASSES, model_name="efficientnet_b2")
    
    # Freeze early layers
    model.freeze_layers(freeze_percentage=0.7)
    
    
    
    # Load best model for evaluation
    best_model = DentalModel(num_classes=config.NUM_CLASSES, model_name="efficientnet_b2")
    # best_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    best_model = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    # best_model.to(config.DEVICE)
    
   
    # Create inference interface
    print("Creating Gradio interface...")
    inferencer = Inferencer(config.MODEL_SAVE_PATH, config)
    iface = inferencer.create_gradio_interface()
    iface.launch(share=True)
    
if __name__ == "__main__":
    main()
