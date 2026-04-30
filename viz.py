#########################################################

############### AI generated code #######################


#########################################################



import torch
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend so it doesn't crash on server
import matplotlib.pyplot as plt
from model import YOLOx
from PIL import Image
import torchvision.transforms as T

image_path = "/home/kenny/Downloads/F-35A_flight.jpg"

print(f"Loading image from: {image_path}")
# 1. Load and preprocess the image using PIL and torchvision
img = Image.open(image_path).convert('RGB')
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(), # Automatically converts to [C, H, W] and normalizes to 0-1
])

img_tensor = transform(img).unsqueeze(0) # Add batch dimension -> [1, C, H, W]
print(f"Loaded image tensor shape: {img_tensor.shape}")

# ==========================================
# METHOD 1: Using Forward Hooks (Recommended)
# ==========================================
print("\n--- Running Method 1: Forward Hooks ---")
model = YOLOx(num_classes=80)
model.eval()

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Hook into stage1
model.backbone.stage1.register_forward_hook(get_activation('stage1'))

with torch.no_grad():
    model(img_tensor)

feat_map_m1 = activations['stage1']
print(f"Method 1 Captured shape (Stage 1): {feat_map_m1.shape}")

# Plotting Method 1
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('Method 1: First 16 Channels of Stage 1', fontsize=16)
for i in range(16):
    row, col = i // 4, i % 4
    channel_image = feat_map_m1[0, i, :, :].numpy()
    axes[row, col].imshow(channel_image, cmap='viridis')
    axes[row, col].axis('off')
    axes[row, col].set_title(f"Ch {i}")
plt.tight_layout()
plt.savefig('method1_feature_map.png')
print("Saved Method 1 visualization to method1_feature_map.png")

# ==========================================
# METHOD 2: Modifying Forward Pass
# ==========================================
print("\n--- Running Method 2: Modifying Forward Pass ---")

class YOLOx_Visualizer(YOLOx):
    def forward(self, x):
        # We override the forward pass to ONLY return the backbone features!
        out_feature_1, out_feature_2, out_feature_3 = self.backbone(x)
        return out_feature_1 # This is the output of stage3

model_m2 = YOLOx_Visualizer(num_classes=80)
model_m2.eval()

with torch.no_grad():
    # It directly returns the feature map now because of our override
    feat_map_m2 = model_m2(img_tensor)
    
print(f"Method 2 Captured shape (Stage 3 / out_feature_1): {feat_map_m2.shape}")

# Plotting Method 2
fig2, axes2 = plt.subplots(4, 4, figsize=(10, 10))
fig2.suptitle('Method 2: First 16 Channels of Stage 3', fontsize=16)
for i in range(16):
    row, col = i // 4, i % 4
    channel_image = feat_map_m2[0, i, :, :].numpy()
    axes2[row, col].imshow(channel_image, cmap='magma') # Using a different colormap
    axes2[row, col].axis('off')
    axes2[row, col].set_title(f"Ch {i}")
plt.tight_layout()
plt.savefig('method2_feature_map.png')
print("Saved Method 2 visualization to method2_feature_map.png")
