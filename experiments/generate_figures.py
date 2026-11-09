"""
Generate figures 1-4 for the manuscript.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Global figure settings (high quality)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

os.makedirs("results/figures", exist_ok=True)

# ============================================================
# Figure 1: Overall architecture
# ============================================================
def create_figure1():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    boxes = {
        'Input Data': (0.05, 0.5, 0.15, 0.15),
        'TCN-GMM\nForecaster': (0.25, 0.5, 0.15, 0.15),
        'Wasserstein\nDRO Model': (0.45, 0.5, 0.15, 0.15),
        'C&CG\nAlgorithm': (0.65, 0.5, 0.15, 0.15),
        'HABC-PM\nSeparation': (0.65, 0.2, 0.15, 0.15),
        'Optimal\nDispatch': (0.85, 0.5, 0.15, 0.15),
    }
    
    colors = ['#e0f7fa', '#fff3e0', '#f3e5f5', '#e8f5e9', '#ffebee', '#e0f2f1']
    
    for i, (label, (x, y, w, h)) in enumerate(boxes.items()):
        rect = plt.Rectangle((x, y), w, h, facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.annotate('', xy=(0.25, 0.575), xytext=(0.20, 0.575), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.45, 0.575), xytext=(0.40, 0.575), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.65, 0.575), xytext=(0.60, 0.575), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.725, 0.35), xytext=(0.725, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.85, 0.575), xytext=(0.80, 0.575), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.65, 0.35), xytext=(0.725, 0.35), arrowprops=dict(arrowstyle='->', lw=2, linestyle='--'))
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.65, 0.35), arrowprops=dict(arrowstyle='->', lw=2, linestyle='--'))
    
    ax.set_title("Figure 1: Overall Architecture of the Proposed DFL Framework", fontsize=16, pad=20)
    plt.savefig("results/figures/figure1_architecture.png", bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")

# ============================================================
# Figure 2: C&CG convergence
# ============================================================
def create_figure2():
    iterations = [1]
    lb_small = [2.2069]
    ub_small = [2.2069]
    lb_33 = [1.4462]
    ub_33 = [1.4462]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(iterations, lb_small, 'o-', label='LB', color='blue', linewidth=2, markersize=10)
    axes[0].plot(iterations, ub_small, 's--', label='UB', color='red', linewidth=2, markersize=10)
    axes[0].set_xlabel('C&CG Iteration')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('(a) Small (Single-Bus)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(iterations, lb_33, 'o-', label='LB', color='blue', linewidth=2, markersize=10)
    axes[1].plot(iterations, ub_33, 's--', label='UB', color='red', linewidth=2, markersize=10)
    axes[1].set_xlabel('C&CG Iteration')
    axes[1].set_ylabel('Objective Value')
    axes[1].set_title('(b) IEEE 33-bus')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Figure 2: C&CG Convergence History', fontsize=16)
    plt.tight_layout()
    plt.savefig("results/figures/figure2_convergence.png", bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")

# ============================================================
# Figure 3: Ablation study
# ============================================================
def create_figure3():
    instances = ['Small\n(Single-Bus)', 'IEEE 33-bus']
    habc_fitness = [2.3293, 2.2927]
    abc_fitness = [2.0597, 2.0549]
    
    x = np.arange(len(instances))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, habc_fitness, width, label='HABC-PM', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, abc_fitness, width, label='Standard ABC', color='#e74c3c', edgecolor='black')
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Worst-case Fitness')
    ax.set_title('Figure 3: Ablation Study - HABC-PM vs Standard ABC')
    ax.set_xticks(x)
    ax.set_xticklabels(instances)
    ax.legend()
    ax.set_ylim(0, 2.8)
    ax.grid(True, axis='y', alpha=0.3)
    
    for i, (h, a) in enumerate(zip(habc_fitness, abc_fitness)):
        improvement = (h - a) / a * 100
        ax.text(i, 0.5, f'+{improvement:.1f}%', ha='center', fontsize=12, 
                fontweight='bold', color='green', bbox=dict(boxstyle='round', facecolor='white'))
    
    plt.tight_layout()
    plt.savefig("results/figures/figure3_ablation.png", bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")

# ============================================================
# Figure 4: Sensitivity to epsilon
# ============================================================
def create_figure4():
    epsilons = [0.01, 0.05, 0.10, 0.20, 0.50]
    penalty = [3.60, 18.00, 36.00, 72.00, 180.00]
    objective = [5.81, 20.21, 38.21, 74.21, 182.21]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Wasserstein Radius ($\\epsilon$)')
    ax1.set_ylabel('Objective Value', color=color1)
    ax1.plot(epsilons, objective, 'o-', color=color1, linewidth=2, markersize=10, label='Objective (with $\\epsilon$)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Penalty Term', color=color2)
    ax2.bar(epsilons, penalty, width=0.02, alpha=0.3, color=color2, label='Penalty Term')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Figure 4: Sensitivity to Wasserstein Radius ($\\epsilon$)')
    plt.tight_layout()
    plt.savefig("results/figures/figure4_sensitivity.png", bbox_inches='tight')
    plt.close()
    print("Figure 4 saved.")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating figures...")
    create_figure1()
    create_figure2()
    create_figure3()
    create_figure4()
    print("\nAll figures saved to results/figures/.")
