"""Generate training graphs from metrics.csv."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Read CSV
epochs, train_loss, val_loss, mAP, best_mAP, lr = [], [], [], [], [], []
with open("assets/metrics.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row["epoch"].strip():
            continue
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))
        mAP.append(float(row["mAP"]))
        best_mAP.append(float(row["best_mAP"]))
        lr.append(float(row["lr"]))

# 1. Loss curves
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, train_loss, label="Train Loss")
ax.plot(epochs, val_loss, label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training & Validation Loss")
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig("assets/loss_curve.png", dpi=150)
plt.close()
print("Saved loss_curve.png")

# 2. mAP
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, mAP, label="mAP")
ax.plot(epochs, best_mAP, label="Best mAP", linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("mAP")
ax.set_title("Mean Average Precision (mAP)")
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.savefig("assets/map_curve.png", dpi=150)
plt.close()
print("Saved map_curve.png")

# 3. Learning rate
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, lr)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule")
ax.grid(True)
plt.tight_layout()
fig.savefig("assets/lr_schedule.png", dpi=150)
plt.close()
print("Saved lr_schedule.png")

print("Done!")
