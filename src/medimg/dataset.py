"""Torch datasets that build phantom patches on demand.

The dataset is generated, not stored: a spec (an integer seed) deterministically
produces a volume, a lesion slice and a set of patches.  That keeps the repo
free of binary fixtures while still giving the training loop real, varying data.
"""
