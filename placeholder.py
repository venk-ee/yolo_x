
import torch

def get_in_box_mask(gt_box, grid_coords):
    """
    gt_box: [1, 4] -> Ground truth box in [x1, y1, x2, y2] format
    grid_coords: [8400, 2] -> The flat list of X, Y coordinates for all grid cells
    """
    # 1. Extract the center points for all 8400 grid cells
    x_centers = grid_coords[:, 0]
    y_centers = grid_coords[:, 1]
    
    # 2. Extract the coordinates of the real dog bounding box
    x1, y1, x2, y2 = gt_box[0]
    
    # 3. Calculate the delta (distance) from every grid center to the 4 edges of the box
    # b_l = distance to Left edge
    # b_r = distance to Right edge
    # b_t = distance to Top edge
    # b_b = distance to Bottom edge
    b_l = x_centers - x1
    b_r = x2 - x_centers
    b_t = y_centers - y1
    b_b = y2 - y_centers
    
    # 4. Stack them into a matrix of shape [8400, 4]
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], dim=1)
    
    # 5. Check if the MINIMUM distance for each point is greater than 0
    # If the minimum distance is > 0, it means all 4 distances are > 0 (it is inside the box!)
    is_in_box = bbox_deltas.min(dim=1).values > 0.0
    
    return is_in_box



def simota_label_assignment(predictions, gt_box, gt_class, grid_coords):
    # This is the master wrapper that chains your 4 steps together!
    
    # 1. PRE-FILTER: Get the 25 candidates in the 5x5 room
    # Returns [1600] boolean mask
    in_box_mask = get_in_box_mask(gt_box, grid_coords) 
    
    # Filter our predictions so we only test the 25 candidates
    candidate_boxes = predictions[:, :4][in_box_mask]
    candidate_cls = predictions[:, 5:][in_box_mask]
    
    # 2. COST MATRIX: Grade the 25 candidates
    # Returns [25] cost scores
    cost_matrix, ious = calculate_loss_matrix(candidate_cls, candidate_boxes, gt_class, gt_box)
    
    # 3. DYNAMIC K: Figure out the budget based on overlaps
    # Returns an integer (e.g., 4)
    dynamic_k = calculate_dynamic_k(ious)
    
    # 4. FINAL MATCH: Hire the winners
    # Returns [25] boolean mask (e.g., 4 True, 21 False)
    candidate_winners_mask = assign_winners(cost_matrix, dynamic_k)
    
    # --- The Un-Zoom Plumbing ---
    # We map those 4 True values back to the master 1,600 list
    final_fg_mask = torch.zeros_like(in_box_mask, dtype=torch.bool)
    
    # We only update the slots where the candidates were standing
    final_fg_mask[in_box_mask] = candidate_winners_mask
    
    return final_fg_mask



################ concat the o/p from the model #######################


def process_and_concat_heads(head_outputs, strides=[8, 16, 32]):
    """
    head_outputs: A list of 3 tensors from your model: [pred_small, pred_med, pred_large]
                  Each tensor is shape [batch_size, 85, H, W]
    strides: The corresponding strides for those feature maps
    """
    batch_size = head_outputs[0].shape[0]
    
    all_decoded_boxes = []
    all_obj_preds = []
    all_cls_preds = []
    all_grid_coords = [] # We need this for the SimOTA 5x5 room!

    for i, pred in enumerate(head_outputs):
        stride = strides[i]
        _, _, H, W = pred.shape
        
        # 1. Generate the grid for THIS specific feature map
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((x_grid, y_grid), dim=-1).float().to(pred.device) # Shape: [H, W, 2]
        
        # Flatten the grid and multiply by stride to get real-world pixel coordinates
        flat_grid = grid.reshape(-1, 2) * stride # Shape: [H*W, 2]
        all_grid_coords.append(flat_grid)
        
        # 2. Flatten and Permute the Predictions
        # Change [batch, 85, H, W] to [batch, H*W, 85]
        flat_pred = pred.view(batch_size, 85, -1).permute(0, 2, 1)
        
        # Split into Box, Obj, and Class
        raw_boxes = flat_pred[..., :4]
        obj_preds = flat_pred[..., 4:5]
        cls_preds = flat_pred[..., 5:]
        
        # 3. Decode the boxes using the stride (similar to your utils.py)
        # Assuming raw_boxes are [dx, dy, dw, dh]
        cx = (raw_boxes[..., 0] + flat_grid[:, 0]) * stride
        cy = (raw_boxes[..., 1] + flat_grid[:, 1]) * stride
        w  = torch.exp(raw_boxes[..., 2]) * stride
        h  = torch.exp(raw_boxes[..., 3]) * stride
        decoded_boxes = torch.stack([cx, cy, w, h], dim=-1)
        
        # Add to our master lists
        all_decoded_boxes.append(decoded_boxes)
        all_obj_preds.append(obj_preds)
        all_cls_preds.append(cls_preds)

    # --- THE GREAT CONCATENATION ---
    # Glue the 3 levels together along the sequence dimension (dim=1)
    # 6400 + 1600 + 400 = 8400 total cells!
    final_boxes = torch.cat(all_decoded_boxes, dim=1) # [batch, 8400, 4]
    final_obj   = torch.cat(all_obj_preds, dim=1)     # [batch, 8400, 1]
    final_cls   = torch.cat(all_cls_preds, dim=1)     # [batch, 8400, 80]
    final_grids = torch.cat(all_grid_coords, dim=0)   # [8400, 2]
    
    # Recombine them into the exact format your loss function expects
    final_predictions = torch.cat([final_boxes, final_obj, final_cls], dim=-1) # [batch, 8400, 85]
    
    return final_predictions, final_grids



# --- THE TRAINING LOOP ---
for epoch in range(100):
    for images, targets in dataloader:
        
        # ACTOR 3: Wipe the detective's notebook clean from the last image!
        optimizer.zero_grad()
        
        # A. FORWARD PASS (What you built early on)
        # Pass the image through the network to get the raw guesses
        predictions = model(images) 
        
        # B. MATCHMAKING (What you just built!)
        # Use SimOTA to figure out who is supposed to learn from the targets
        fg_mask = assign_winners(predictions, targets)
        
        # C. CALCULATE LOSS (The Report Card)
        # Grade the predictions based on the SimOTA assignments
        total_loss = calculate_final_loss(predictions, targets.box, targets.cls, fg_mask)
        
        # ACTOR 1: THE DETECTIVE (Backpropagation)
        # Calculate the Gradients (the blame scores) for all 10 million weights
        total_loss.backward()
        
        # ACTOR 2: THE ENFORCER (Gradient Descent)
        # Actually update the 10 million weights so the network is smarter next time!
        optimizer.step()
        
    print(f"Epoch {epoch} finished! Loss is going down!")

# ... inside the training loop ...

        # Flatten the 3 feature maps into 8,400 predictions
        flat_preds, flat_grids = process_and_concat_heads(head_outputs)
        
        # CHAIN ALL 4 STEPS OF SIMOTA TOGETHER
        fg_mask = simota_label_assignment(flat_preds, target_box, target_class, flat_grids)
        
        # Calculate the final loss using the winning mask
        total_loss = calculate_final_loss(flat_preds, target_box, target_class, fg_mask)

# ... backward() and step() ...

