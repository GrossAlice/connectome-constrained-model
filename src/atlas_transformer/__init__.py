"""Atlas-indexed Transformer for multi-worm neural dynamics.

One model for all worms: each worm's neurons are embedded into the
canonical 302-neuron C. elegans atlas.  Unobserved neurons are zero-padded,
and an observation mask is concatenated so the model knows which neurons
are real vs. missing.

Architecture matches the best per-worm config (B_wide_256h8) but with
atlas-sized input/output.
"""
