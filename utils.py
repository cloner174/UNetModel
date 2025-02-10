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
                    ax_true_circle.set_title('True Mask')
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
            
            if images_shown >= num_samples:
                break
    
    plt.tight_layout()
    plt.show()



def compute_metrics(model, dataloader, device, threshold=0.5, epsilon=1e-7):
    """
    Compute average segmentation metrics (IoU, Dice, Precision, and Recall) 
    over the entire dataset.
    
    This function assumes that the dataloader returns batches with annotated
    segmentation masks in the form:
         (images, masks, labels, boxes)
    If a batch does not include masks, that batch is skipped.
    
    The model is assumed to return three outputs:
         outputs_seg, outputs_cls, outputs_loc
    and the segmentation prediction is chosen as follows:
      - If the predicted classification (after sigmoid) is positive (i.e. >0.5),
        then the predicted mask is obtained by applying sigmoid to the segmentation 
        output and thresholding it.
      - Otherwise, the predicted mask is computed using argmax over the segmentation output.
    
    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): Dataloader providing validation/test data.
        device (torch.device): Device on which computation is performed.
        threshold (float): Threshold to binarize predicted masks.
        epsilon (float): Small constant to avoid division by zero.
    
    Returns:
        dict: Dictionary with average metrics: IoU, Dice Score, Precision, and Recall.
    """
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            # Expect annotated data with segmentation masks.
            if len(batch) == 4:
                images, masks, labels, boxes = batch
            else:
                # Skip batches without segmentation masks.
                continue
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Run the model; adjust according to your model's output
            outputs_seg, outputs_cls, outputs_loc = model(images)
            preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            
            batch_size = images.size(0)
            for i in range(batch_size):
                # Decide which segmentation prediction to use based on classification:
                pred_cls = preds_cls[i].item()
                if int(pred_cls) == 1:
                    # Use the sigmoid probabilities and threshold them.
                    pred_prob = torch.sigmoid(outputs_seg[i])
                    # If the segmentation head returns a single channel
                    if pred_prob.shape[0] == 1:
                        pred_mask = (pred_prob > threshold).float()
                        pred_mask = pred_mask.squeeze(0)  # shape: (H, W)
                    else:
                        # For multi-channel segmentation, assume channel 1 is the positive class.
                        pred_mask = (pred_prob[1] > threshold).float()
                else:
                    pred_mask = torch.argmax(outputs_seg[i], dim=0).float()
                
                # Get the corresponding ground-truth mask.
                # Handle both (1, H, W) or (H, W) formats.
                if masks[i].dim() == 3:
                    true_mask = masks[i].squeeze(0)
                else:
                    true_mask = masks[i]
                
                true_mask = (true_mask > 0.5).float()
                
                intersection = (pred_mask * true_mask).sum()
                union = pred_mask.sum() + true_mask.sum() - intersection
                iou = (intersection + epsilon) / (union + epsilon)
                
                dice = (2 * intersection + epsilon) / (pred_mask.sum() + true_mask.sum() + epsilon)
                
                precision = (intersection + epsilon) / (pred_mask.sum() + epsilon)
                recall = (intersection + epsilon) / (true_mask.sum() + epsilon)
                
                total_iou += iou.item()
                total_dice += dice.item()
                total_precision += precision.item()
                total_recall += recall.item()
                total_samples += 1
    
    if total_samples == 0:
        return {
            "IoU": None,
            "Dice": None,
            "Precision": None,
            "Recall": None
        }
    
    metrics = {
        "IoU": total_iou / total_samples,
        "Dice": total_dice / total_samples,
        "Precision": total_precision / total_samples,
        "Recall": total_recall / total_samples
    }
    
    return metrics


#cloner174
