"""Volume container plus the loading, resampling and slicing helpers.

A ``Volume`` is a NumPy array with the metadata that medical images always
carry: voxel spacing, patient-space origin and modality.  The class is frozen
because a volume is passed between pipeline stages and silent in-place mutation
of the array is the classic way to corrupt a training set.
"""
