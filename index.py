# -*- coding: utf-8 -*-
"""index.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-e-2AjA5lZex9duz7o62RXaEJ7K01bl2
"""

#
from google.colab import drive

drive.mount('./Drive')
#

#
!git clone https://github.com/cloner174/UNetModel.git

# Commented out IPython magic to ensure Python compatibility.
# %cd UNetModel
#

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from datasets import FullyAnnotatedDataset, WeaklyAnnotatedDataset
from UNet import MultitaskAttentionUNet
from utils import visualize_predictions
from losses import dice_loss, CombinedLoss

import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

capability = torch.cuda.get_device_capability(device)

print(f"Compute capability: {capability[0]}.{capability[1]}")

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', f"{capability[0]}.{capability[1]}")

print(f'Using device: {device}')

#
base_dir = '/content/Drive/MyDrive/LunaProject' #/main

X = np.load(base_dir + '/X.npy')
masks = np.load(base_dir + '/masks.npy')
boxes = np.load(base_dir + '/y_centroids.npy')
y_class = np.load(base_dir + '/y_class.npy')

X_weak = np.load(base_dir + '/images.npy')
y_train_weak = np.load(base_dir + '/labels.npy')

X_train, X_val, masks_train, masks_val, boxes_train, boxes_val, y_train, y_val = train_test_split(
        X , masks, boxes, y_class,
        test_size=0.2, random_state=42, shuffle=True
)

annotated_dataset = FullyAnnotatedDataset(X_train, masks_train, boxes_train, y_train, transform=None)

weak_dataset = WeaklyAnnotatedDataset(X_weak, y_train_weak, transform=None)

val_dataset = FullyAnnotatedDataset(X_val, masks_val, boxes_val, y_val, transform=None)

annotated_loader = DataLoader(annotated_dataset, batch_size=64, shuffle=True, num_workers=2)

weak_loader = DataLoader(weak_dataset, batch_size=64, shuffle=True, num_workers=2)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

"""مدل معمولی"""

model = MultitaskAttentionUNet(input_channels=1, num_classes=1, bbox_size=4).to(device)

import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

criterion_seg = CombinedLoss()
criterion_cls = nn.BCEWithLogitsLoss()
criterion_loc = nn.MSELoss()

#
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.4, patience=4)
#

base_dir = './'
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):

    model.train()
    train_seg_loss = 0.0
    train_cls_loss = 0.0
    train_loc_loss = 0.0
    train_total = 0

    for batch in tqdm(annotated_loader, desc=f'Epoch {epoch}/{num_epochs} - Annotated-Training'):

        images, masks, labels, boxes = batch
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device).unsqueeze(1)
        boxes = boxes.to(device)

        optimizer.zero_grad()

        outputs_seg, outputs_cls, outputs_loc = model(images)


        loss_seg = criterion_seg(outputs_seg, masks.float())
        loss_cls = criterion_cls(outputs_cls, labels)
        loss_loc = criterion_loc(outputs_loc, boxes)

        loss = loss_seg + loss_cls + loss_loc

        loss.backward()
        optimizer.step()

        train_seg_loss += loss_seg.item() * images.size(0)
        train_cls_loss += loss_cls.item() * images.size(0)
        train_loc_loss += loss_loc.item() * images.size(0)
        train_total += images.size(0)

    avg_train_seg_loss = train_seg_loss / train_total
    avg_train_cls_loss = train_cls_loss / train_total
    avg_train_loc_loss = train_loc_loss / train_total
    avg_train_loss = avg_train_seg_loss + avg_train_cls_loss + avg_train_loc_loss

    model.eval()
    val_seg_loss = 0.0
    val_cls_loss = 0.0
    val_loc_loss = 0.0
    val_total = 0
    correct_cls = 0
    total_cls = 0

    with torch.no_grad():

        for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} - Validation'):

            images, masks, labels, boxes = batch
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device).unsqueeze(1)
            boxes = boxes.to(device)

            outputs_seg, outputs_cls, outputs_loc = model(images)

            loss_seg = criterion_seg(outputs_seg, masks.float())
            loss_cls = criterion_cls(outputs_cls, labels)
            loss_loc = criterion_loc(outputs_loc, boxes)

            val_seg_loss += loss_seg.item() * images.size(0)
            val_cls_loss += loss_cls.item() * images.size(0)
            val_loc_loss += loss_loc.item() * images.size(0)
            val_total += images.size(0)

            preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            correct_cls += (preds_cls == labels).sum().item()
            total_cls += labels.size(0)

    avg_val_seg_loss = val_seg_loss / val_total
    avg_val_cls_loss = val_cls_loss / val_total
    avg_val_loc_loss = val_loc_loss / val_total
    avg_val_loss = avg_val_seg_loss + avg_val_cls_loss + avg_val_loc_loss
    val_accuracy = correct_cls / total_cls

    scheduler.step(avg_val_loss)

    print(f'Epoch [{epoch}/{num_epochs}] \n'
          f'Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, '
          f'Cls: {avg_train_cls_loss:.4f}, Loc: {avg_train_loc_loss:.4f}) \n'
          f'Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, '
          f'Cls: {avg_val_cls_loss:.4f}, Loc: {avg_val_loc_loss:.4f}) \n'
          f'Val Acc: {val_accuracy:.4f} \n')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(base_dir, 'UNet.pth'))
        print('Best model saved! \n')

#train_model(model, annotated_loader, weak_loader, val_loader, device, num_epochs=2, patience=10, base_dir='./')

model.load_state_dict(torch.load(os.path.join('./', 'UNet.pth'), weights_only=True))
model.to(device)

visualize_predictions(model, val_loader, device, num_samples=15, num_classes=1)

import matplotlib.pyplot as plt
import torch


def visualize_predictions(model, dataloader, device, num_samples=5, num_classes=1):

    model.eval()
    images_shown = 0
    plt.figure(figsize=(20, num_samples * 5))

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                images, masks, labels, boxes = batch
                is_annotated = True
            else:
                images, labels = batch
                is_annotated = False

            images = images.to(device)

            if is_annotated:
                masks = masks.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)

            outputs_seg, outputs_cls, outputs_loc = model(images)

            preds_seg = torch.argmax(outputs_seg, dim=1)
            preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            preds_loc = outputs_loc * 64.0

            for i in range(images.size(0)):
                if images_shown >= num_samples:
                    break

                img = images[i].cpu().squeeze().numpy()
                if is_annotated:
                    true_mask = masks[i].cpu().numpy()
                    true_box = boxes[i].cpu().numpy() * 64.0
                    true_label = labels[i].item()
                else:
                    true_mask = None
                    true_box = None
                    true_label = 1


                pred_cls = preds_cls[i].item()

                if int(pred_cls) == 1 :
                    pred = torch.sigmoid(outputs_seg[i])
                    pred_mask = pred.squeeze().cpu().numpy()
                else:
                    pred_mask = preds_seg[i].cpu().numpy()

                pred_box = preds_loc[i].cpu().numpy()


                plt.subplot(num_samples, 6, images_shown * 6 + 1)
                plt.imshow(img, cmap='gray')
                plt.title('Input Image')
                plt.axis('off')

                if is_annotated:
                    plt.subplot(num_samples, 6, images_shown * 6 + 2)
                    plt.imshow(true_mask, cmap='jet')
                    plt.title('True Mask')
                    plt.axis('off')
                else:
                    plt.subplot(num_samples, 6, images_shown * 6 + 2)
                    plt.axis('off')

                plt.subplot(num_samples, 6, images_shown * 6 + 3)
                plt.imshow(pred_mask, cmap='jet')
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.subplot(num_samples, 6, images_shown * 6 + 4)
                if is_annotated:
                    plt.text(0.1, 0.5, f'True: {int(true_label)}\nPred: {int(pred_cls)}',
                             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
                else:
                    plt.text(0.1, 0.5, f'Pred: {int(pred_cls)}',
                             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

                plt.title('Classification')
                plt.axis('off')

                if is_annotated:
                    plt.subplot(num_samples, 6, images_shown * 6 + 5)
                    plt.imshow(img, cmap='gray')
                    plt.title('True Boxes')
                    plt.axis('off')
                    for b in range(0, len(true_box), 2):
                        plt.plot(true_box[b], true_box[b+1], 'go-')
                else:
                    plt.subplot(num_samples, 6, images_shown * 6 + 5)
                    plt.axis('off')

                plt.subplot(num_samples, 6, images_shown * 6 + 6)
                plt.imshow(img, cmap='gray')
                plt.title('Pred Boxes')
                plt.axis('off')
                for b in range(0, len(pred_box), 2):
                    plt.plot(pred_box[b], pred_box[b+1], 'ro-')

                images_shown += 1

            if images_shown >= num_samples:
                break

    plt.tight_layout()
    plt.show()
#cloner174

#
!cp ./UNet.pth {base_dir}
#

"""استفاده از رزنت

**به هیچ وجه توضیه نمیشه چون اولا برای داده های سه بعدی سه کاناله مناسبه هم اینکه اطلاعات حیاتی از تصویر رو توی مراحل کارش از دست میدیم**

فقط جهت اطلاع -- نتایج غیرقابل قبول
"""

#from UNet import MultitaskAttentionUNet_Pretrained

#from UNet import MultitaskAttentionUNet_Pretrained
#model = MultitaskAttentionUNet_Pretrained(input_channels=1, num_classes=1, bbox_size=4).to(device)

#train_model(model, annotated_loader, weak_loader, val_loader, device, num_epochs=2, patience=10, base_dir='./')



from metrics import calculate_segmentation_metrics, calculate_classification_metrics, calculate_localization_metrics

preds_cls_v = (torch.sigmoid(outputs_cls_v.squeeze()) > 0.5).float()

#outputs_seg, outputs_cls, outputs_loc = model(images)

            loss_seg = criterion_seg(outputs_seg, masks.float())
            loss_cls = criterion_cls(outputs_cls, labels)
            loss_loc = criterion_loc(outputs_loc, boxes)

            val_seg_loss += loss_seg.item() * images.size(0)
            val_cls_loss += loss_cls.item() * images.size(0)
            val_loc_loss += loss_loc.item() * images.size(0)
            val_total += images.size(0)

            preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            correct_cls += (preds_cls == labels).sum().item()
            total_cls += labels.size(0)

preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
actual_cls = (torch.sigmoid(labels) > 0.5).float()

preds_cls.sum().tolist(), actual_cls.sum().tolist(),

#
dice_v, iou_v = calculate_segmentation_metrics(outputs_seg, masks)

prec_v, rec_v, f1_v, roc_auc_v = calculate_classification_metrics(outputs_cls, labels)

loc_metr = calculate_localization_metrics(outputs_loc, boxes)
#
print(f'Dice: {dice_v}, IoU: {iou_v}')
print(f'ROC_AUC: {roc_auc_v}, F1Score: {f1_v}')
print(f'Prec: {prec_v}, Rec: {rec_v}')
print(f'Loc_Metr: {loc_metr}')

#model.load_state_dict(torch.load(os.path.join('./', 'best_model.pth'), weights_only=True))
#model.to(device)

#val_dataset_for_visual = FullyAnnotatedDataset(X_val, masks_val, boxes_val, y_val, transform=None, unsqueeze_mask=False)

#val_loader_for_visual = DataLoader(val_dataset_for_visual, batch_size=64, shuffle=False, num_workers=4)

#visualize_predictions(model, val_loader_for_visual, device, num_samples=15, num_classes=1)

outputs_seg.size()

images.squeeze().size()

images.squeeze().cpu().numpy().shape

outputs_seg.squeeze().cpu().numpy().shape

masks.squeeze().cpu().numpy().shape

import matplotlib.pyplot as plt

plt.imshow(images.squeeze().cpu().numpy()[30] , cmap= 'gray')

import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_circles_on_masked_regions(images, masks):
    if len(images) != len(masks):
        raise ValueError("The number of images and masks must be the same")
    for i, (image, mask) in enumerate(zip(images, masks)):
        binary_mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 255, 0), 2)
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Creating some example images and masks
    image1 = np.zeros((200, 200, 3), dtype=np.uint8)
    image2 = np.zeros((200, 200, 3), dtype=np.uint8)
    mask1 = np.zeros((200, 200), dtype=np.uint8)
    mask2 = np.zeros((200, 200), dtype=np.uint8)

    # Draw some example regions in masks
    cv2.circle(mask1, (100, 100), 40, 255, -1)
    cv2.rectangle(mask2, (50, 50), (150, 150), 255, -1)

    images = [image1, image2]
    masks = [mask1, mask2]

    # Call the function to draw circles on masked regions
    draw_circles_on_masked_regions(images, masks)

import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_circles_on_masked_regions(images, masks):
    if len(images) != len(masks):
        raise ValueError("The number of images and masks must be the same")
    for i, (image, mask) in enumerate(zip(images, masks)):
        binary_mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 255, 0), 2)
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

masks.squeeze().cpu().numpy()[5].any()

binary_mask = masks.squeeze().cpu().numpy()[5].astype(np.uint8)


contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

plt.subplot(1, len(images), i + 1)

contours

cv2.minEnclosingCircle(contour)

for contour in contours:
  (x, y), radius = cv2.minEnclosingCircle(contour)
  center = (int(x), int(y))
  radius = int(radius)
  cv2.circle(images.squeeze().cpu().numpy()[5], center, radius, (0, 255, 0), 2)

plt.show()



for contour in contours:

plt.imshow(images.squeeze().cpu().numpy()[5] , cmap= 'gray')

plt.imshow(masks.squeeze().cpu().numpy()[5] , cmap= 'gray')

plt.imshow(outputs_seg.squeeze().cpu().numpy()[5] , cmap= 'gray')

pred = torch.sigmoid(outputs_seg[5])
plt.imshow(pred.squeeze().cpu().numpy() , cmap= 'gray')

pred.squeeze().cpu().numpy()