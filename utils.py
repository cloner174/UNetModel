import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np



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
                    true_mask = masks[i].squeeze().cpu().numpy()
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



def pro_prediction(model, dataloader, device, num_samples=5):
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
                labels = labels.to(device)
            
            outputs_seg, outputs_cls, _ = model(images)
            
            preds_seg = torch.argmax(outputs_seg, dim=1)
            preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            
            for i in range(images.size(0)):
                if images_shown >= num_samples:
                    break
                
                img = images[i].cpu().squeeze().numpy()
                
                ax_img = plt.subplot(num_samples, 6, images_shown * 6 + 1)
                ax_img.imshow(img, cmap='gray')
                ax_img.set_title('Input Image')
                ax_img.axis('off')
                if is_annotated:
                    true_mask = masks[i].squeeze().cpu().numpy()
                    true_label = labels[i].item()
                else:
                    true_label = 1
                
                pred_cls = preds_cls[i].item()
                
                if int(pred_cls) == 1:
                    pred = torch.sigmoid(outputs_seg[i])
                    pred_mask = pred.squeeze().cpu().numpy()
                else:
                    pred_mask = preds_seg[i].cpu().numpy()

                ax_cls = plt.subplot(num_samples, 6, images_shown * 6 + 4)
                if is_annotated:
                    ax_cls.text(0.1, 0.5, f'True: {int(true_label)}\nPred: {int(pred_cls)}',
                                fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
                else:
                    ax_cls.text(0.1, 0.5, f'Pred: {int(pred_cls)}',
                                fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
                
                ax_cls.set_title('Classification')
                ax_cls.axis('off')

                if is_annotated:
                    ax_true_circle = plt.subplot(num_samples, 6, images_shown * 6 + 2)
                    ax_true_circle.imshow(img, cmap='gray')
                    ax_true_circle.set_title('True Mask & Circles')
                    ax_true_circle.axis('off')
                    
                    mask_binary = (true_mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if len(contour) > 0:
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            center = (int(x), int(y))
                            radius = int(radius)
                            circle = plt.Circle(center, radius, edgecolor='blue', facecolor='none', lw=0.7)
                            ax_true_circle.add_patch(circle)
                
                ax_pred_circle = plt.subplot(num_samples, 6, images_shown * 6 + 3)
                ax_pred_circle.imshow(img, cmap='gray')
                ax_pred_circle.set_title('Pred Mask & Circles')
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
            
            if images_shown >= num_samples:
                break
    
    plt.tight_layout()
    plt.show()


#cloner174