"""Teacher forcing: human takes -> limb strokes (inverse kinematics) injected into motor neurons,
blended with weight alpha annealed 1 -> 0, training only the trainable subset.

Fits 8 GB VRAM: dt = 1 ms, truncated backprop windows of ~100-200 ms, torch.no_grad() elsewhere.
The scikit-learn readout baseline (the fallback demo) also lives here.
"""
