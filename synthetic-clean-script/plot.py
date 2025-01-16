import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', default="dummy_results", type=str)
args = parser.parse_args()

save_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f'/results/{args.res_dir}')
df = pd.read_csv(os.path.join(save_dir, 'r_p.csv')) 

cp = sns.color_palette("tab10")
fig, axs = plt.subplots(1, 3, figsize=(8,3)) 
target_vars = ["SE_S", "SE_A", "SE_Y1"]

x_min, x_max = float('inf'), float('-inf')
y_min, y_max = float('inf'), float('-inf')

for idx, key in enumerate(target_vars):
    axs[idx].set_xlabel(r"$\rho (b1(X)$" + f", {key})", fontsize=16)
    axs[idx].axhline(y=-np.log10(0.05), color='dimgray', linestyle='--', label='p = 0.05')
    
    hat_rho = df[key + "_r"]
    hat_p = df[key + "_p"]
    log_p_val = np.clip(-np.log10(hat_p), a_min=None, a_max=5)
    axs[idx].scatter(hat_rho, log_p_val, color=cp[idx], s=1, alpha=0.5)

    percent = int(100 * len(df.query(f"{key}_r > 0 & {key}_p < 0.05")) / len(df))
    axs[idx].text(0.5, 0.9, f"%{percent}", fontsize=12, ha='center', va='center', transform=axs[idx].transAxes)

    #  for plotting all four figures with the same x and y limits
    x0, x1 = axs[idx].get_xlim()
    y0, y1 = axs[idx].get_ylim()
    
    x_min, x_max = min(x_min, x0), max(x_max, x1)
    y_min, y_max = min(y_min, y0), max(y_max, y1)

axs[0].set_ylabel('-log10(p-value)', fontsize=16)

for ax in axs.flat:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()

#############################################################################################

