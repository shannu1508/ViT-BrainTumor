import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from transformer import TumorClassifierViT


# ===============================
# Device configuration
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===============================
# Load model
# ===============================
model = TumorClassifierViT(num_classes=4)

checkpoint_path = "best_model.pth"
assert os.path.exists(checkpoint_path), "best_model.pth not found"

state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()


# ===============================
# Image transformations
# ===============================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ===============================
# Prediction function
# ===============================
def predict_image(image_path, model, transform, device):
    assert os.path.exists(image_path), "Image path does not exist"

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs[0], dim=0)

    confidence, predicted_class = torch.max(probabilities, dim=0)

    return (
        predicted_class.item(),
        confidence.item(),
        probabilities.cpu().numpy(),
        image
    )


# ===============================
# Image path (CHANGE IF NEEDED)
# ===============================
image_path = "./test6.jpg"


# ===============================
# Load class names
# ===============================
train_dataset = ImageFolder("./data/Training", transform=data_transforms)
class_names = train_dataset.classes
print("Class names:", class_names)


# ===============================
# Run prediction
# ===============================
predicted_class, confidence, probabilities, image = predict_image(
    image_path, model, data_transforms, device
)

print(f"Predicted class : {class_names[predicted_class]}")
print(f"Confidence      : {confidence * 100:.2f}%")


# ===============================
# Visualization
# ===============================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Show image
axs[0].imshow(image)
axs[0].set_title(f"Prediction: {class_names[predicted_class]}")
axs[0].axis("off")

# Confidence bar chart
prob_percent = probabilities * 100
bars = axs[1].barh(class_names, prob_percent, color="black", height=0.5)
axs[1].set_xlim(0, 100)
axs[1].set_title("Confidence Level")
axs[1].xaxis.grid(True, linestyle="--", alpha=0.3)

# Add percentage labels
for bar in bars:
    width = bar.get_width()
    axs[1].text(
        width + 1 if width < 50 else width - 5,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center",
        ha="left" if width < 50 else "right",
        color="black" if width < 50 else "white",
        fontweight="bold"
    )

# Save result
output_path = "prediction_result.png"
plt.savefig(output_path, bbox_inches="tight")
print(f"Prediction figure saved to: {output_path}")

plt.show()
