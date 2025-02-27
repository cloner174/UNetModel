import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from metrics import calculate_segmentation_metrics, calculate_classification_metrics, calculate_localization_metrics
from losses import CombinedLoss


def train_model(model, 
                annotated_loader, 
                weak_loader, 
                val_loader,
                device, 
                num_epochs=50, 
                patience=10, 
                base_dir='./', 
                num_positive = None,
                num_negative = None, 
                w_seg = None,
                w_cls = None,
                w_loc = None):
    """
    Train the model using the specified approach.
    
    Args:
        model: The multitask model to be trained.
        annotated_loader: DataLoader for fully annotated data (images, masks, labels, boxes).
        weak_loader: DataLoader for weakly annotated data (images, labels).
        val_loader: DataLoader for validation data.
        device: Device to run the training on (CPU or GPU).
        num_epochs: Number of epochs to train.
        patience: Number of epochs to wait for improvement before early stopping.
        base_dir: Directory to save the best model.
    
    Returns:
        None
    """
    if num_positive is not None and num_negative is not None:
        pos_weight_value = num_negative / (num_positive + 1e-5)
        pos_weight = torch.tensor([pos_weight_value]).to(device)
        criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion_cls = nn.BCEWithLogitsLoss()
    
    criterion_seg = CombinedLoss()
    criterion_loc = nn.SmoothL1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    lr_current = optimizer.param_groups[0]['lr']
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(1, num_epochs + 1):
        if optimizer.param_groups[0]['lr'] != lr_current:
            print(f"Update Learning Rate: {optimizer.param_groups[0]['lr']/lr_current}")
            lr_current = optimizer.param_groups[0]['lr']
        
        model.train()
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_loc_loss = 0.0
        train_total = 0
        # ----------------------
        for batch in tqdm(annotated_loader, desc=f'Epoch {epoch}/{num_epochs} - Annotated Training'):
            images_a, masks_a, labels_a, boxes_a = batch
            images_a = images_a.to(device)
            masks_a = masks_a.to(device)
            labels_a = labels_a.to(device).view(-1)  # [B]
            boxes_a = boxes_a.to(device)
            
            optimizer.zero_grad()
            
            outputs_seg_a, outputs_cls_a, outputs_loc_a = model(images_a)
            
            loss_seg_a = criterion_seg(outputs_seg_a, masks_a.unsqueeze(1).float())
            
            loss_cls_a = criterion_cls(outputs_cls_a.squeeze(), labels_a.float())
            
            has_nodule = labels_a == 1
            if has_nodule.any():
                loc_pred = outputs_loc_a[has_nodule]
                loc_target = boxes_a[has_nodule]
                loss_loc_a = criterion_loc(loc_pred, loc_target)
            else:
                loss_loc_a = torch.tensor(0.0, device=device)
            
            if w_seg is not None and w_cls is not None and w_loc is not None:
                loss_a = w_seg * loss_seg_a + w_cls * loss_cls_a + w_loc * loss_loc_a
            else:
                loss_a = loss_seg_a + loss_cls_a + loss_loc_a
            
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_seg_loss += loss_seg_a.item() * images_a.size(0)
            train_cls_loss += loss_cls_a.item() * images_a.size(0)
            train_loc_loss += loss_loc_a.item() * images_a.size(0)
            train_total += images_a.size(0)
        # ----------------------------
        for batch in tqdm(weak_loader, desc=f'Epoch {epoch}/{num_epochs} - Weak Training'):
            images_w, labels_w = batch
            images_w = images_w.to(device)
            labels_w = labels_w.to(device).view(-1)  # [B]
            
            optimizer.zero_grad()
            
            _, outputs_cls_w, _ = model(images_w)
            
            loss_cls_w = criterion_cls(outputs_cls_w.squeeze(), labels_w.float())
            
            loss_w = loss_cls_w
            
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_cls_loss += loss_cls_w.item() * images_w.size(0)
            train_total += images_w.size(0)
        
        avg_train_seg_loss = train_seg_loss / train_total
        avg_train_cls_loss = train_cls_loss / train_total
        avg_train_loc_loss = train_loc_loss / train_total
        avg_train_loss = avg_train_seg_loss + avg_train_cls_loss + avg_train_loc_loss
        # ----------------------
        model.eval()
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        val_loc_loss = 0.0
        val_total = 0
        correct_cls = 0
        total_cls = 0
        dice_score = 0.0
        iou_score = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        roc_auc = 0.0
        mae = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} - Validation'):
                images_v, masks_v, labels_v, boxes_v = batch
                images_v = images_v.to(device)
                masks_v = masks_v.to(device)
                labels_v = labels_v.to(device).view(-1)  # [B]
                boxes_v = boxes_v.to(device)
                
                outputs_seg_v, outputs_cls_v, outputs_loc_v = model(images_v)
                
                loss_seg_v = criterion_seg(outputs_seg_v, masks_v.unsqueeze(1).float())
                
                loss_cls_v = criterion_cls(outputs_cls_v.squeeze(), labels_v.float())
                
                has_nodule_v = labels_v == 1
                if has_nodule_v.any():
                    loc_pred_v = outputs_loc_v[has_nodule_v]
                    loc_target_v = boxes_v[has_nodule_v]
                    loss_loc_v = criterion_loc(loc_pred_v, loc_target_v)
                else:
                    loss_loc_v = torch.tensor(0.0, device=device)
                
                val_seg_loss += loss_seg_v.item() * images_v.size(0)
                val_cls_loss += loss_cls_v.item() * images_v.size(0)
                val_loc_loss += loss_loc_v.item() * images_v.size(0)
                val_total += images_v.size(0)
                
                preds_cls_v = (torch.sigmoid(outputs_cls_v.squeeze()) > 0.5).float()
                correct_cls += (preds_cls_v == labels_v.float()).sum().item()
                total_cls += labels_v.size(0)
                
                dice_v, iou_v = calculate_segmentation_metrics(outputs_seg_v, masks_v)
                dice_score += dice_v * images_v.size(0)
                iou_score += iou_v * images_v.size(0)
                
                prec_v, rec_v, f1_v, roc_auc_v = calculate_classification_metrics(outputs_cls_v, labels_v)
                precision += prec_v * images_v.size(0)
                recall += rec_v * images_v.size(0)
                f1 += f1_v * images_v.size(0)
                roc_auc += roc_auc_v * images_v.size(0)
                
                if has_nodule_v.any():
                    mae_v = calculate_localization_metrics(outputs_loc_v[has_nodule_v], boxes_v[has_nodule_v])
                    mae += mae_v * has_nodule_v.sum().item()
                else:
                    mae += 0.0
        
        avg_val_seg_loss = val_seg_loss / val_total
        avg_val_cls_loss = val_cls_loss / val_total
        avg_val_loc_loss = val_loc_loss / val_total
        
        if w_seg is not None and w_cls is not None and w_loc is not None:
            avg_val_loss = w_seg * avg_val_seg_loss + w_cls * avg_val_cls_loss + w_loc * avg_val_loc_loss
        else:
            avg_val_loss = avg_val_seg_loss + avg_val_cls_loss + avg_val_loc_loss
        
        val_accuracy = correct_cls / total_cls
        
        dice_score /= val_total
        iou_score /= val_total
        precision /= val_total
        recall /= val_total
        f1 /= val_total
        roc_auc /= val_total
        mae /= val_total if mae != 0 else 1
        
        scheduler.step(avg_val_loss)
        print('\n\n'
              f'Epoch [{epoch}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, '
              f'Cls: {avg_train_cls_loss:.4f}, Loc: {avg_train_loc_loss:.4f}) '
              f'Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, '
              f'Cls: {avg_val_cls_loss:.4f}, Loc: {avg_val_loc_loss:.4f}) \n'
              f'Val Acc: {val_accuracy:.4f} '
              f'Dice: {dice_score:.4f} IoU: {iou_score:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} '
              f'ROC-AUC: {roc_auc:.4f} MAE: {mae:.4f}'
              '\n')
        
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
    


def train_unet(model, 
                annotated_loader, 
                val_loader,
                device, 
                criterion_seg=None, 
                criterion_cls = None,
                criterion_loc = None, 
                num_epochs=5, 
                base_dir='./'):
    
    best_val_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
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
            
                pred = torch.sigmoid(outputs_seg) 
            
                loss_seg = criterion_seg(pred, masks.float())
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
        print(f'Epoch [{epoch}/{num_epochs}] \n'
            f'Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, '
            f'Cls: {avg_train_cls_loss:.4f}, Loc: {avg_train_loc_loss:.4f}) \n'
            f'Val Loss: {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, '
            f'Cls: {avg_val_cls_loss:.4f}, Loc: {avg_val_loc_loss:.4f}) \n'
            f'Val Acc: {val_accuracy:.4f} \n')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            print('Best model saved! \n')



def train_with_weak(model , annotated_loader, weak_loader, val_loader , device , num_epochs = 5):

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_seg = CombinedLoss()
    criterion_loc = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_loc_loss = 0.0
        train_total_annotated = 0
        train_total_cls = 0
        
        for batch in tqdm(annotated_loader, desc=f'Epoch {epoch}/{num_epochs} - Annotated Training'):
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
            train_total_annotated += images.size(0)
            train_total_cls += images.size(0)
    
        for batch in tqdm(weak_loader, desc=f'Epoch {epoch}/{num_epochs} - Weak Training'):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
        
            outputs_seg, outputs_cls, outputs_loc = model(images)

            loss_cls = criterion_cls(outputs_cls, labels)
            loss = loss_cls

            loss.backward()
            optimizer.step()

            train_cls_loss += loss_cls.item() * images.size(0)
            train_total_cls += images.size(0)

        avg_train_seg_loss = train_seg_loss / train_total_annotated if train_total_annotated > 0 else 0
        avg_train_cls_loss = train_cls_loss / train_total_cls if train_total_cls > 0 else 0
        avg_train_loc_loss = train_loc_loss / train_total_annotated if train_total_annotated > 0 else 0
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
            torch.save(model.state_dict(), os.path.join('.', 'UNet_WithWeak.pth'))
            print('Best model saved! \n')



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

#cloner174