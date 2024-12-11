import os
import json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.optim as optim
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

# Custom Dataset class for COCO-style dataset
class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = [f for f in sorted(os.listdir(root)) if f.endswith(".jpg") or f.endswith(".png")]
        self.annotations_file = os.path.join(root, '_annotations.coco.json')

        # Load annotations using COCO API
        self.coco = COCO(self.annotations_file)
        self.ids = list(self.coco.imgs.keys())
        
    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Extract bounding boxes, labels, and masks
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:  # Check for valid segmentation
                xmin = ann['bbox'][0]
                ymin = ann['bbox'][1]
                xmax = xmin + ann['bbox'][2]
                ymax = ymin + ann['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id'])
                mask = coco.annToMask(ann)
                masks.append(mask)

        if not boxes:
            return None, None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

# Custom transform function
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = T.ToTensor()(image)
        return image, target

# Function to collate data in DataLoader
def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))  # Filter out None values
    return tuple(zip(*batch))

# Define transformations (adjust as needed)
transform = Compose([
    ToTensor()
])

# Paths to your dataset directories
train_root = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\train"
val_root = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\valid"
test_root = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\test"

# Create datasets and data loaders
# train_dataset = CustomDataset(train_root, transforms=transform)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
train_dataset = CustomDataset(train_root, transforms=transform)
train_dataset.imgs = train_dataset.imgs[:10]
train_dataset.ids = train_dataset.ids[:10]
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


val_dataset = CustomDataset(val_root, transforms=transform)
val_dataset.imgs = val_dataset.imgs[:10]
val_dataset.ids = val_dataset.ids[:10]
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

test_dataset = CustomDataset(test_root, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Define Mask R-CNN model
backbone = resnet_fpn_backbone('resnet50', pretrained=True)
model = MaskRCNN(backbone, num_classes=6)  # Adjust num_classes as per your dataset

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training function
def train_model(model, train_loader, val_loader, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print("Training start ----------------------->")
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print(loss_dict)
            losses.backward()
            optimizer.step()
            

            train_loss += losses.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for images, targets in val_loader:
        #         images = list(image.to(device) for image in images)
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #         loss_dict = model(images, targets)
        #         losses = sum(loss for loss in loss_dict.values())
        #         val_loss += losses.item()

        # val_loss /= len(val_loader)

        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        val_loss = 0.0
        total_batches = len(val_loader)
        print("Validation Starts --------------->")
        for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Ensure targets list is not empty
                if targets:
                    list_loss_dict = model(images, targets)

                    # Check if list_loss_dict is not empty
                    if list_loss_dict:
                        # Look for appropriate loss key in list_loss_dict
                        losses = 0.0
                        for loss_dict in list_loss_dict:
                            if 'losses' in loss_dict:
                                losses += sum(loss_dict['losses'])  # Sum up losses if 'losses' key is present
                                print(losses)

                        val_loss += losses

                else:
                    total_batches -= 1  # Decrease total_batches count if targets is empty

            # Calculate average validation loss
        if total_batches >  0:
                val_loss /= total_batches

        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}")

# Test function (optional)
def test_model(model, test_loader):
    model.eval()
    # Add code to evaluate on test set (predictions, metrics, etc.)

# Train the model
train_model(model, train_loader, val_loader, optimizer)

def get_prediction(model, image_path, threshold):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img = T.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img)

    pred_score = list(prediction[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (prediction[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [i for i in prediction[0]['labels'].detach().cpu().numpy()]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in prediction[0]['boxes'].detach().cpu().numpy()]
    masks = masks[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]

    return masks, pred_boxes, pred_class

# Optionally, test the model
# test_model(model, test_loader)

image_path = r"C:\Users\pc\Downloads\Shampoo_5class.v2-only_non_defective-21-06-2024.coco\test\HT-GE505GC-T1-C-Snapshot-20240617-175034-718-200687408979_JPG_jpg.rf.e37b511d6273539ab18217e4b5725687.jpg"
masks, boxes, labels = get_prediction(model, image_path, threshold=0.5)
print("Masks:", masks)
print("Boxes:", boxes)
print("Labels:", labels)
