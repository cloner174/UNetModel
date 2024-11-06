import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from datasets import PredDataset
from torch.utils.data import DataLoader

def set_size(img):
    if img.size[0] != 64 or img.size[1] != 64:
        img = img.resize((64,64))
    return img

def handle_img_data(image_data):
    imgs = []
    if image_data.ndim == 2:
        image = Image.fromarray(image_data.astype(np.uint8))
        image = set_size(image)
        imgs.append(np.array(image))
    elif image_data.ndim == 3:
        for slic in range(image_data.shape[2]):
            image = Image.fromarray(image_data[:, :, slic].astype(np.uint8))
            image = set_size(image)
            imgs.append(np.array(image))
    else:
        raise ValueError("Can not Handle Images with dim bigger than 3 or smaller than 2! please Contact to owner!")
    return imgs


def get_nib(in_path):
    try:
        import nibabel as nib
    except ModuleNotFoundError:
        print('Please install nibabel , using command: pip install nibabel')
        raise
    return nib.load(in_path).get_fdata()


def get_mhd(in_path):
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError:
        print('Please install SimpleITK , using command: pip install SimpleITK')
        raise
    return sitk.GetArrayFromImage(sitk.ReadImage(in_path))


def predict(sample, model, device = None):
    images = []
    if isinstance(sample, str):
        
        if os.path.isfile(sample) :
            
            if sample.endswith('.png') or sample.endswith('.jpg'):
                image = Image.open(sample).convert('L')
                image = set_size(image)
                images.append( np.array( image ) )
            
            elif sample.endswith('.nii.gz'):
                img_data = get_nib(sample)
                images = handle_img_data(img_data)
            
            elif sample.endswith('.mhd'):
                img_data = get_mhd(sample)
                images = handle_img_data(img_data)
            else:
                raise TypeError("This Function Just Can Handle files in one of  '.png' or '.jpg' or '.nii.gz' or '.mhd'  format!")
        
        elif os.path.isdir(sample):
            for any_image in os.listdir(sample):
                image_path = os.path.join(sample, any_image )
                
                if any_image.endswith('.png') or any_image.endswith('.jpg'):
                    image = Image.open(image_path).convert('L')
                    image = set_size(image)
                    images.append(np.array( image ))
                
                elif any_image.endswith('.nii.gz'):
                    img_data = get_nib(image_path)
                    images.extend( handle_img_data(img_data) )
                
                elif sample.endswith('.mhd'):
                    img_data = get_mhd(image_path)
                    images.extend( handle_img_data(img_data) )
                
                else:
                    raise TypeError("This Function Just Can Handle files in one of  '.png' or '.jpg' or '.nii.gz' or '.mhd'  format!")
        else:
            raise ValueError("The Input Must be 1. Path to image or 2. Path to images directory!")
    
    else:
        raise ValueError("Sorry! Currently This Function Just Supports Path to image or Path to images directory!")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pred_dataset = PredDataset(images)
    pred_loader = DataLoader(pred_dataset, batch_size = 1, shuffle=False)
    
    num_samples = len(pred_loader)
    model.to(device)
    model.eval()
    
    plt.figure(figsize=(20, num_samples * 5))
    
    images_shown = 0
    
    with torch.no_grad():
      for batch in pred_loader:
        batch = batch.to(device)
        outputs_seg, outputs_cls, _ = model(batch)
        
        preds_seg = torch.argmax(outputs_seg, dim=1)
        preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            
        for i in range(batch.size(0)):
            
            img = batch[i].cpu().squeeze().numpy()
            
            ax_img = plt.subplot(num_samples, 6, images_shown * 6 + 1)
            
            ax_img.imshow(img, cmap='gray')
            ax_img.set_title('Input Image')
            ax_img.axis('off')
            
            pred_cls = preds_cls[i].item()
                
            if int(pred_cls) == 1:
                pred = torch.sigmoid(outputs_seg[i])
                pred_mask = pred.squeeze().cpu().numpy()
            else:
                pred_mask = preds_seg[i].cpu().numpy()
            
            ax_cls = plt.subplot(num_samples, 6, images_shown * 6 + 4)
            ax_cls.text(0.1, 0.5, f'Pred: {int(pred_cls)}',
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            
            ax_cls.set_title('Classification')
            ax_cls.axis('off')
            
            ax_pred_circle = plt.subplot(num_samples, 6, images_shown * 6 + 3)
            ax_pred_circle.imshow(img, cmap='gray')
            ax_pred_circle.set_title('Pred Mask')
            ax_pred_circle.axis('off')
            
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(pred_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) > 0:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    circle = plt.Circle(center, radius, edgecolor='red', facecolor='none', lw=0.7)
                    ax_pred_circle.add_patch(circle)
            
            images_shown += 1
    
    plt.tight_layout()
    plt.show()
    
#cloner174
