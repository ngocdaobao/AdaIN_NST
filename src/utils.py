import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

transforms_pipeline = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomCrop((256, 256)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image):
    image = transforms_pipeline(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=transforms_pipeline):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files_total = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        #Sample a subset about 40k images for faster training
        self.image_files = self.image_files_total[:20000]
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_files[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Get DataLoader
def data_loader(batch_size=32, shuffle=True, num_workers=4, seed=42):
    torch.manual_seed(seed)

    #for style images
    style_dataset = ImageFolder(root='/kaggle/input/best-artworks-of-all-time/images/images', transform=transforms_pipeline)
    style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #for content images
    content_dataset = ImageDataset(root_dir='/kaggle/input/coco-2014-dataset-for-yolov3/coco2014/images/train2014', transform=transforms_pipeline)
    content_loader = DataLoader(content_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return style_loader, content_loader

# Example usage
if __name__ == "__main__":
    style_loader, content_loader = data_loader()
    #Number of images
    num_style_images = len(style_loader.dataset)
    num_content_images = len(content_loader.dataset)
    print(f"Number of style images: {num_style_images}")
    print(f"Number of content images: {num_content_images}")
    for style_batch in style_loader:
        print(f"Style batch shape: {style_batch[0].shape}")
        for content_batch in content_loader:
            print(f"Content batch shape: {content_batch[0].shape}")
            break
        break


    
    
