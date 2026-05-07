import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from model import YOLOX
from utils import calculate_final_loss,decode_box,prepare_for_nms,post_processing


model=YOLOX(num_classes=80)
optimizer=optim.SGD(model.parameters(),lr=0.001)

import torch
import torch.optim as optim

# 1. Setup the Network and the Optimizer
model = YOLOX_Model() # Your Backbone, Neck, and Head
optimizer = optim.SGD(model.parameters(), lr=0.01) # The Manager (Learning Rate = 0.01)

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


