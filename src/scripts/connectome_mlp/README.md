# Connectome-constrained MLP comparison scripts

This folder contains scripts for comparing MLP neural activity prediction
with and without connectome constraints.

## Scripts

1. **run_connectome_mlp_batch.py**: 
   Batch runner that processes all worms through the connectome decoder.

2. **plot_connectome_summary.py**: 
   Generates summary plots aggregating results across all worms.

## Usage

```bash
# Run on all worms (T_e connectome, all neurons)
python -m scripts.connectome_mlp.run_connectome_mlp_batch \
    --device cuda \
    --neurons all \
    --T_matrices T_e

# Generate summary plots
python -m scripts.connectome_mlp.plot_connectome_summary \
    --results_dir output_plots/connectome_mlp_batch \
    --T_matrices T_e
```

## Conditions Compared

- **conn+self**: MLP using only connectome neighbours + self history
- **all+self**: MLP using all neurons + self history (unconstrained baseline)
- **conn_only**: MLP using only connectome neighbours (no self)
- **all_only**: MLP using all neurons (no self)

## Key Question

Does knowing the connectome structure help predict neural activity,
or can an unconstrained MLP learn the same from data alone?

If **conn+self ≈ all+self**, then the connectome provides no additional
structural prior—the MLP discovers the relevant connections from data.

If **conn+self > all+self**, then the connectome helps by reducing
the hypothesis space (fewer parameters, better generalization).

If **conn+self < all+self**, then the connectome is too sparse or
misses functional connections that the unconstrained model discovers.
