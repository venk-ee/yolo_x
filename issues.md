Validation score threshold is probably too high
In train_and_val.py (line 93), validation calls post_process_nms(out) without lowering score_thresh, so it uses the default from utils.py (line 301).
That default is 0.05.
Early in training, your model may predict decent-looking boxes but with low confidence scores like 0.01 or 0.02.
If the threshold is 0.05, those boxes get thrown away before COCO evaluation ever sees them.
Result: mAP can look like 0.0000 even when the model is starting to learn.
Why this matters:
This is an evaluation issue, not necessarily a training issue.
It can make learning look worse than it really is.
For validation, 0.001 is usually safer because it keeps more candidate boxes for COCO to judge.

Validation is doing extra work you may not need
In train_and_val.py (line 79) to train_and_val.py (line 89), you compute full loss during validation.
That means validation is not just checking predictions, it is also doing SimOTA matching and all the loss math from loss.py (line 44).
SimOTA is expensive because it compares many anchors against ground truth boxes.
Why this matters:
Validation takes longer every epoch.
If you only care about checkpointing by mAP, the loss value is not helping much.
You are paying for a lot of compute that does not change which model gets saved.
In short:
Good for monitoring if you want it.
Not ideal if speed is the priority.

Checkpoints are saved, but resume support is missing
In train.py (line 49), metrics.csv is opened with "w".
That means every time you start training, the old CSV file is wiped and replaced.
Also, although train.py (line 97) saves last_checkpoint.pth, the script does not load it back in when restarted.
Why this matters:
If training crashes or you stop it, you cannot continue cleanly from the last point.
You lose the historical CSV log.
You also lose the continuity of optimizer/scheduler state, which matters for good training behavior.
What this means practically:
Right now training is more like “start fresh every time”.
For long runs, resume support is a big quality-of-life and reliability improvement.

NMS is class-agnostic
In utils.py (line 351) to utils.py (line 360), you use torchvision.ops.nms(boxes, scores, iou_thresh).
That NMS looks only at box overlap and score.
It does not care about class labels.
Why this matters:
For your current dataset, this is fine because the dataset has only 1 class.
But if you later train on multiple classes, a box for class A can suppress a box for class B if they overlap.
That is usually wrong for multi-class detectors.
So:
Current project: okay.
Future multi-class project: switch to batched_nms.

infer_on_val.py is hardcoded for 1 class
In infer_on_val.py (line 14), the model is created as YOLOX(num_classes=1).
That matches your current dataset, so this is correct right now.
Why it matters:
If you later train a model with more classes, this inference script will no longer match the trained checkpoint.
It may fail to load weights or give incorrect outputs.