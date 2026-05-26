"""Phantom geometry and intensity tests.

These assert the physical properties a reviewer would check by hand: the
ellipsoid mask encloses the right volume, the skull ring is bone-bright, the
lesion mask matches the painted voxels, and two runs with the same seed are
bit-identical.
"""
