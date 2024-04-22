import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from io import BytesIO
from torchvision.models import vit_b_16, regnet_y_128gf, efficientnet_v2_s, convnext_large, swin_v2_b
from torchvision.models import alexnet, vgg16, googlenet, resnet152, resnext101_64x4d

class ImageNetParquetDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list of str): List of file paths to the .parquet files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.concat([pd.read_parquet(f) for f in file_paths])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming 'image' is the column name for the image dictionary
        img_data = self.data.iloc[idx]['image']
        label = self.data.iloc[idx]['label']

        # Extracting image bytes using the 'bytes' key from the dictionary
        img_bytes = img_data['bytes']

        if not isinstance(img_bytes, bytes):
            raise TypeError("The image data is not in bytes format. Found type: {}".format(type(img_bytes)))

        # Loading the image from bytes
        image = Image.open(BytesIO(img_bytes))

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Data collected from https://huggingface.co/datasets/imagenet-1k
ROOT = './imagenet_1k_resized_256/data'
file_paths = [
    f'{ROOT}/val-00000-of-00002-b5248be478d25e41.parquet',
    f'{ROOT}/val-00001-of-00002-85f3d9c8fa1edb63.parquet',
]

print("Loading ImageNet1K Data...")
imagenet_dataset = ImageNetParquetDataset(file_paths, transform=transform)
data_loader = DataLoader(imagenet_dataset, batch_size=256, shuffle=False)
print("Done.")

print("Loading models...")
# models = {
#     "ViT-B-16": vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1"),
#     "RegNet-Y-128GF": regnet_y_128gf(weights="RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1"),
#     "EfficientNet-V2-s": efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1"),
#     "ConvNeXt-Large": convnext_large(weights="ConvNeXt_Large_Weights.IMAGENET1K_V1"), # Try tiny if this doesnt work ConvNeXt_Tiny_Weights.IMAGENET1K_V1
#     "Swin-V2-B": swin_v2_b(weights="Swin_V2_B_Weights.IMAGENET1K_V1")
# }
models = {
    "AlexNet": alexnet(weights="DEFAULT"),
    "VGG16": vgg16(weights="DEFAULT"),
    "GoogLeNet": googlenet(weights="DEFAULT"),
    "ResNet152": resnet152(weights="DEFAULT"),
    "RexNext101": resnext101_64x4d(weights="DEFAULT")
}
print("Done.")

# Test loop
for m in models:
    # Move models[m] to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models[m].to(device)

    # Classify Images
    print(f"Classifying Imagenet images with {m}...")
    models[m].eval()  # Set models[m] to evaluation mode
    total = 0
    correct = 0
    scores = torch.tensor([], device=device)
    ground_truth = torch.tensor([], dtype=torch.long, device=device)
    with torch.no_grad():  # No need to track gradients
        for images, labels in data_loader:
            images = images.to(device)
            outputs = models[m](images)
            scores = torch.cat((scores, outputs), dim=0)
            ground_truth = torch.cat((ground_truth, labels.to(device)), dim=0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum().item()
    accuracy = correct / total
    print(f"Done! {m} Acc: {accuracy}%")

    scores = scores.cpu().numpy()  # Move to CPU and convert to NumPy array if not already on CPU

    # Save to CSV file
    pd.DataFrame(scores).to_csv(f"./Imagenet1k/{m}_scores{accuracy:.2f}.csv")

    if m == "Swin-V2-B":
        pd.DataFrame(ground_truth.cpu()).to_csv(f"./Imagenet1k/ground_truths.csv")

print("Done!")
# This is a better way to save model results
# np.savetxt(f"./mnist_scores/{MODEL}.csv", scores, delimiter=",")