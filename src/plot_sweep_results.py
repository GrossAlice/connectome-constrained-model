import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# List of sweep folders to process
data_roots = [
    'output_plots/stage2/sweep_beat_ridge',
    'output_plots/stage2/sweep_beat_ridge_v2',
    'output_plots/stage2/sweep_lag_nbr_routing',
    'output_plots/stage2/sweep_lag_routing_v2',
]

# Collect results
def collect_results(root):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        if 'summary.json' in filenames:
            summary_path = os.path.join(dirpath, 'summary.json')
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                config_name = os.path.relpath(dirpath, root)
                results.append({
                    'sweep': os.path.basename(root),
                    'config': config_name,
                    'cv_onestep_r2_mean': summary.get('cv_onestep_r2_mean'),
                    'cv_loo_r2_mean': summary.get('cv_loo_r2_mean'),
                    'cv_loo_r2_median': summary.get('cv_loo_r2_median'),
                })
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")
    return results

all_results = []
for root in data_roots:
    all_results.extend(collect_results(root))

df = pd.DataFrame(all_results)

# Plotting
sns.set(style="whitegrid")

for sweep in df['sweep'].unique():
    sub = df[df['sweep'] == sweep].sort_values('cv_loo_r2_mean', ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x='config', y='cv_loo_r2_mean', data=sub, palette='viridis')
    plt.title(f'LOO R² by config ({sweep})')
    plt.ylabel('LOO R² (mean)')
    plt.xlabel('Config')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{sweep}_loo_r2.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(x='config', y='cv_onestep_r2_mean', data=sub, palette='mako')
    plt.title(f'1-step R² by config ({sweep})')
    plt.ylabel('1-step R² (mean)')
    plt.xlabel('Config')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{sweep}_onestep_r2.png')
    plt.close()

print('Plots saved as *_loo_r2.png and *_onestep_r2.png in the current directory.')
