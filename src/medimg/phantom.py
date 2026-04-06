"""Synthetic CT phantoms.

Everything downstream (windowing, patching, training, calibration) needs
volumes that behave like real CT series: Hounsfield-unit intensities, a skull
that windows to white, brain parenchyma around 35-45 HU, CSF near 0 HU and a
hyperdense lesion that a classifier can actually learn.  Real public datasets
are hundreds of gigabytes, so this module builds deterministic phantoms from a
seed instead.  The generator is intentionally geometric: a human inspecting a
slice should be able to see the head outline, the skull ring, the ventricles
and the lesion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.medimg.volume import Volume

# Hounsfield-unit anchors used by the generator and by the tests.
HU_AIR = -1000.0
HU_FAT = -90.0
HU_CSF = 6.0
HU_WHITE_MATTER = 32.0
HU_GRAY_MATTER = 44.0
HU_SOFT_TISSUE = 40.0
HU_BONE = 900.0
