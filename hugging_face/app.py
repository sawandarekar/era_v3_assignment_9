import gradio as gr
from model import ResNet50_Model
from torchvision import transforms
from PIL import Image
import torch
from classes import i2d  # Import the i2d dictionary

# Generate a sorted list of class IDs from i2d keys
class_ids = sorted(i2d.keys())

MODEL_SAVE_PATH= "resnet50_imagenet.pt"

# Load the model
model = ResNet50_Model()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Classify function
def classify_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    # Map predicted index to class ID and label
    predicted_class_id = class_ids[predicted.item()]
    predicted_label = i2d.get(predicted_class_id, "Unknown")

    #return f"Predicted Class: {predicted.item()}"
    return f"Predicted Class: {predicted_label}"

# Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="ResNet-50 Tiny ImageNet Classifier"
)

demo.launch()
