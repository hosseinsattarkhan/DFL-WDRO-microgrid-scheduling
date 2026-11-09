"""
Generate the graphical abstract for the manuscript.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

boxes = [
    (0.05, 0.4, 0.25, 0.2, 'Decision-Focused\nLearning (DFL)', '#e3f2fd'),
    (0.375, 0.4, 0.25, 0.2, 'Wasserstein\nDRO Model', '#f3e5f5'),
    (0.70, 0.4, 0.25, 0.2, 'Hybrid C&CG\n+ HABC-PM', '#e8f5e9'),
]

for (x, y, w, h, label, color) in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=14, fontweight='bold')

ax.annotate('', xy=(0.375, 0.5), xytext=(0.30, 0.5), arrowprops=dict(arrowstyle='->', lw=3))
ax.annotate('', xy=(0.70, 0.5), xytext=(0.625, 0.5), arrowprops=dict(arrowstyle='->', lw=3))

ax.text(0.5, 0.15, 'Results: Feasible integrated pipeline | HABC-PM higher fitness vs ABC (~13%) | \nDFL improves NLL/MSE but not decision cost/regret on the reported test set | Gap decomposition framework provided',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#d32f2f')

ax.set_title('Graphical Abstract: Decision-Focused Learning for Wasserstein DRO Microgrid Scheduling', fontsize=18, pad=20)
plt.savefig("results/figures/graphical_abstract.png", bbox_inches='tight', dpi=300)
plt.close()
print("Graphical abstract saved.")
