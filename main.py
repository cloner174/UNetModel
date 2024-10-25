import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from metrics import calculate_segmentation_metrics, calculate_classification_metrics, calculate_localization_metrics



def generate_pseudo_labels(outputs_seg, threshold=0.7):
    """
    Generates pseudo-labels based on model confidence.
    Args:
        outputs_seg (Tensor): Raw segmentation outputs from the model. Shape [B, C, H, W].
        threshold (float): Confidence threshold for pseudo-labeling.
    Returns:
        pseudo_masks (Tensor): Pseudo-labels with shape [B, H, W].
        mask_confident (Tensor): Boolean mask indicating high-confidence pixels. Shape [B, H, W].
    """
    probs = F.softmax(outputs_seg, dim=1)  # [B, C, H, W]
    max_probs, pseudo_masks = torch.max(probs, dim=1)  # [B, H, W], [B, H, W]
    mask_confident = max_probs > threshold  # [B, H, W]
    return pseudo_masks, mask_confident



def train_model(model, annotated_loader, weak_loader, val_loader, device, num_epochs=50, patience=10, base_dir='./'):
    
    criterion_seg = nn.CrossEntropyLoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_loc = nn.SmoothL1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(1, num_epochs + 1):
        
        model.train()
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_loc_loss = 0.0
        train_total = 0
        
        for batch in tqdm(annotated_loader, desc=f'Epoch {epoch}/{num_epochs} - Annotated Training'):
            images_a, masks_a, labels_a, boxes_a = batch
            images_a = images_a.to(device)
            masks_a = masks_a.to(device)
            boxes_a = boxes_a.to(device)
            labels_a = labels_a.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs_seg_a, outputs_cls_a, outputs_loc_a = model(images_a)
            
            loss_seg_a = criterion_seg(outputs_seg_a, masks_a)
            loss_cls_a = criterion_cls(outputs_cls_a, labels_a)
            loss_loc_a = criterion_loc(outputs_loc_a, boxes_a)
            
            loss_a = loss_seg_a + loss_cls_a + loss_loc_a
            
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_seg_loss += loss_seg_a.item() * images_a.size(0)
            train_cls_loss += loss_cls_a.item() * images_a.size(0)
            train_loc_loss += loss_loc_a.item() * images_a.size(0)
            train_total += images_a.size(0)
        
        for batch in tqdm(weak_loader, desc=f'Epoch {epoch}/{num_epochs} - Weak Training'):
            images_w, labels_w = batch
            images_w = images_w.to(device)
            labels_w = labels_w.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs_seg_w, outputs_cls_w, outputs_loc_w = model(images_w)
            
            loss_cls_w = criterion_cls(outputs_cls_w, labels_w)
            
            preds_seg_w = torch.argmax(outputs_seg_w, dim=1)
            pseudo_masks_w = preds_seg_w  # Hard pseudo-labels
            
            loss_seg_w = criterion_seg(outputs_seg_w, pseudo_masks_w)            
            
            loss_w = loss_cls_w + loss_seg_w
            
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_seg_loss += loss_seg_w.item() * images_w.size(0)
            train_cls_loss += loss_cls_w.item() * images_w.size(0)
            train_total += images_w.size(0)
        
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
                images, masks, boxes, labels = batch
                images = images.to(device)
                masks = masks.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs_seg, outputs_cls, outputs_loc = model(images)
                
                loss_seg_val = criterion_seg(outputs_seg, masks)
                loss_cls_val = criterion_cls(outputs_cls, labels)
                loss_loc_val = criterion_loc(outputs_loc, boxes)
                
                val_seg_loss += loss_seg_val.item() * images.size(0)
                val_cls_loss += loss_cls_val.item() * images.size(0)
                val_loc_loss += loss_loc_val.item() * images.size(0)
                val_total += images.size(0)
                
                preds_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
                correct_cls += (preds_cls == labels).sum().item()
                total_cls += labels.size(0)
        
                dice, iou = calculate_segmentation_metrics(torch.argmax(outputs_seg, dim=1), masks, num_classes=13)
                dice_score += dice * images.size(0)
                iou_score += iou * images.size(0)
                
                prec, rec, f1_score_val, roc_auc_val = calculate_classification_metrics(outputs_cls, labels)
                precision += prec * images.size(0)
                recall += rec * images.size(0)
                f1 += f1_score_val * images.size(0)
                roc_auc += roc_auc_val * images.size(0)
                
                mae_val = calculate_localization_metrics(outputs_loc, boxes)
                mae += mae_val * images.size(0)
        
        avg_val_seg_loss = val_seg_loss / val_total
        avg_val_cls_loss = val_cls_loss / val_total
        avg_val_loc_loss = val_loc_loss / val_total
        avg_val_loss = avg_val_seg_loss + avg_val_cls_loss + avg_val_loc_loss
        val_accuracy = correct_cls / total_cls
        
        dice_score /= val_total
        iou_score /= val_total
        precision /= val_total
        recall /= val_total
        f1 /= val_total
        roc_auc /= val_total
        mae /= val_total
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, '
              f'Cls: {avg_train_cls_loss:.4f}, Loc: {avg_train_loc_loss:.4f}) '
              f'Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, '
              f'Cls: {avg_val_cls_loss:.4f}, Loc: {avg_val_loc_loss:.4f}) '
              f'Val Acc: {val_accuracy:.4f} '
              f'Dice: {dice_score:.4f} IoU: {iou_score:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} '
              f'ROC-AUC: {roc_auc:.4f} MAE: {mae:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            print('Best model saved!')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break



def train_model_pro(model, annotated_loader, weak_loader, val_loader, device, num_epochs=50, patience=10, base_dir='./'):
    # Define loss functions
    criterion_seg = nn.CrossEntropyLoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_loc = nn.SmoothL1Loss(reduction='mean')  # or any other suitable loss

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_loc_loss = 0.0
        train_total = 0

        # Iterate over annotated data
        for batch in tqdm(annotated_loader, desc=f'Epoch {epoch}/{num_epochs} - Annotated Training'):
            images_a, masks_a, boxes_a, labels_a = batch
            images_a = images_a.to(device)
            masks_a = masks_a.to(device)
            boxes_a = boxes_a.to(device)
            labels_a = labels_a.to(device).unsqueeze(1)

            optimizer.zero_grad()
            
            outputs_seg_a, outputs_cls_a, outputs_loc_a = model(images_a)
            
            loss_seg_a = criterion_seg(outputs_seg_a, masks_a)
            loss_cls_a = criterion_cls(outputs_cls_a, labels_a)
            
            pos_mask = labels_a.squeeze(1) == 1  # [B]
            
            if pos_mask.sum() > 0:
                
                outputs_loc_a_pos = outputs_loc_a[pos_mask]
                boxes_a_pos = boxes_a[pos_mask]
                
                loss_loc_a = criterion_loc(outputs_loc_a_pos, boxes_a_pos)
            else:
                
                loss_loc_a = torch.tensor(0.0, device=device)
            
            loss_a = loss_seg_a + loss_cls_a + loss_loc_a
            
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_seg_loss += loss_seg_a.item() * images_a.size(0)
            train_cls_loss += loss_cls_a.item() * images_a.size(0)
            train_loc_loss += loss_loc_a.item() * images_a.size(0)
            train_total += images_a.size(0)
        
        for batch in tqdm(weak_loader, desc=f'Epoch {epoch}/{num_epochs} - Weak Training'):
            images_w, labels_w = batch
            images_w = images_w.to(device)
            labels_w = labels_w.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            outputs_seg_w, outputs_cls_w, outputs_loc_w = model(images_w)
            
            loss_cls_w = criterion_cls(outputs_cls_w, labels_w)
            
            pseudo_masks_w, mask_confident = generate_pseudo_labels(outputs_seg_w, threshold=0.7)
            
            if mask_confident.sum() > 0:
                outputs_seg_confident = outputs_seg_w[mask_confident]
                pseudo_masks_confident = pseudo_masks_w[mask_confident]
                
                loss_seg_w = criterion_seg(outputs_seg_confident, pseudo_masks_confident)
                
                loss_w = loss_cls_w + loss_seg_w
            else:
                
                loss_w = loss_cls_w

            
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_seg_loss += loss_seg_w.item() * mask_confident.sum().item() if mask_confident.sum() > 0 else 0.0
            train_cls_loss += loss_cls_w.item() * images_w.size(0)
            train_total += images_w.size(0)
        
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
                images, masks, boxes, labels = batch
                images = images.to(device)
                masks = masks.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs_seg, outputs_cls, outputs_loc = model(images)
                
                loss_seg_val = criterion_seg(outputs_seg, masks)
                loss_cls_val = criterion_cls(outputs_cls, labels)
                
                pos_mask_val = labels.squeeze(1) == 1
                if pos_mask_val.sum() > 0:
                    outputs_loc_val_pos = outputs_loc[pos_mask_val]
                    boxes_val_pos = boxes[pos_mask_val]
                    loss_loc_val = criterion_loc(outputs_loc_val_pos, boxes_val_pos)
                else:
                    loss_loc_val = torch.tensor(0.0, device=device)
                
                val_seg_loss += loss_seg_val.item() * images.size(0)
                val_cls_loss += loss_cls_val.item() * images.size(0)
                val_loc_loss += loss_loc_val.item() * images.size(0)
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
        
        print(f'Epoch [{epoch}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, '
              f'Cls: {avg_train_cls_loss:.4f}, Loc: {avg_train_loc_loss:.4f}) '
              f'Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, '
              f'Cls: {avg_val_cls_loss:.4f}, Loc: {avg_val_loc_loss:.4f}) '
              f'Val Acc: {val_accuracy:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            print('Best model saved!')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
    
#cloner174