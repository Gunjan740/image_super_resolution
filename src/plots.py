import pandas as pd
import matplotlib.pyplot as plt

# Load log
df = pd.read_csv("/home/gunjan/train_loss.csv")
plot_title = "without prompt"
# Moving average
window = 1000
df["loss_ma"] = df["loss"].rolling(window=window).mean()

# Plot for training loss of LR semantic
plt.figure(figsize=(8, 4))
plt.plot(df["global_step"], df["loss"], alpha=0.3, label="Raw loss")
plt.plot(df["global_step"], df["loss_ma"], linewidth=2, label=f"MA({window})")

plt.xlabel("Global step")
plt.ylabel("Loss")
plt.title(f"Training Loss {plot_title} (Moving Average)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
