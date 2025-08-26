import torch
from HiFi_Net_loc import HiFiNetLoc  # adjust import according to model class
from PIL import Image
import torchvision.transforms as transforms
import os

# Paths
model_path = "pretrained/hifi_netpp.pth"
input_dir = "../../images_to_test"
output_dir = "../../results"
os.makedirs(output_dir, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HiFiNetLoc()  # adjust constructor as in the repo
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # adjust size
    transforms.ToTensor()
])

# Run inference
for img_file in os.listdir(input_dir):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        # Save output (for example as a numpy array or image)
        output_path = os.path.join(output_dir, img_file)
        output_np = output.squeeze().cpu().numpy()
        from matplotlib import pyplot as plt
        plt.imsave(output_path, output_np, cmap='hot')
