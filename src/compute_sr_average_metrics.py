import os
import re
import pandas as pd
import numpy as np  

# ==================================================
# 🔹 CONFIG — only change this
# ==================================================
base_path = "/home/gunjan/Documents/ULM/semester_5/project Adv visual deep learning 2/results/texture/DIV2K_valid_HR_1024"
input_filename = "results.txt"
output_filename = "average_metrics.csv"
# ==================================================


input_file = os.path.join(base_path, input_filename)
output_file = os.path.join(base_path, output_filename)


# ==================================================
# PARSER
# ==================================================
sr_psnr, sr_ssim, sr_lpips = [], [], []
bi_psnr, bi_ssim, bi_lpips = [], [], []

pattern = re.compile(
    r"SR PSNR ([\d.]+), SSIM ([\d.]+), LPIPS ([\d.]+) \| "
    r"Bicubic PSNR ([\d.]+), SSIM ([\d.]+), LPIPS ([\d.]+)"
)

with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            s_psnr, s_ssim, s_lp, b_psnr, b_ssim, b_lp = map(float, match.groups())

            sr_psnr.append(s_psnr)
            sr_ssim.append(s_ssim)
            sr_lpips.append(s_lp)

            bi_psnr.append(b_psnr)
            bi_ssim.append(b_ssim)
            bi_lpips.append(b_lp)


# ==================================================
# COMPUTE AVERAGES 
# ==================================================
df = pd.DataFrame({
    "Method": ["SR", "Bicubic"],
    "PSNR_avg": [np.mean(sr_psnr), np.mean(bi_psnr)],
    "SSIM_avg": [np.mean(sr_ssim), np.mean(bi_ssim)],
    "LPIPS_avg": [np.mean(sr_lpips), np.mean(bi_lpips)]
})


# ==================================================
# SAVE
# ==================================================
df.to_csv(output_file, index=False)


# ==================================================
# PRINT
# ==================================================
print("\n Average Metrics")
print(df.round(4))  # nicer formatting
print(f"\nSaved to {output_file}")
