import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = "/home/gunjan/Documents/ULM/semester_5/project Adv visual deep learning 2/results/loss"

files = {
    "Without Prompt": "train_loss_without_prompt.csv",
    "Semantic": "train_loss_LR_semantic.csv",
    "Texture": "train_loss_LR_texture.csv"
}

window = 1000

plt.figure(figsize=(9, 5))

for label, fname in files.items():

    df = pd.read_csv(os.path.join(base_path, fname))

    # moving average
    df["loss_ma"] = df["loss"].rolling(window=window).mean()

    # raw
    plt.plot(df["global_step"], df["loss"], alpha=0.15)

    # smoothed
    plt.plot(df["global_step"], df["loss_ma"], linewidth=2, label=f"{label} MA({window})")


plt.xlabel("Global step")
plt.ylabel("Loss")
plt.title("Training Loss Comparison (Moving Average)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ✅ SAVE FIGURE
save_path = os.path.join(base_path, "train_loss_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved to: {save_path}")
