"""Shared surface geometry for normalized binary HD and pooled HD95.

The boundary convention in this module intentionally matches
``confidence.normalized_penalized_hd95`` exactly:

* a surface is ``mask & ~scipy.ndimage.binary_erosion(mask)`` with SciPy's
  default structuring element and border handling;
* for two non-empty masks, directed nearest-surface distances from both
  directions are pooled before ``numpy.percentile(..., 95)`` is applied with
  NumPy's default interpolation method;
* distances are measured in native pixel coordinates and divided by
  ``hypot(height, width)``;
* empty--empty costs zero and a one-sided empty pair costs one.

Preparing the fixed reference computes its surface and distance transform once.
Each non-empty candidate then needs exactly one additional Euclidean distance
transform, and the same pooled distances produce both full normalized Hausdorff
distance (``nhd``) and normalized pooled HD95 (``nhd95``).

Here full HD is Hausdorff distance on the extracted digital surface *sets*.  It
is a metric on those sets; when pulled back to masks through surface extraction,
it should conservatively be described as a pseudometric unless injectivity of
that extraction is established for the mask class under study.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import ndimage


class BoundaryLosses(NamedTuple):
    """Normalized penalized full surface HD and pooled surface HD95."""

    nhd: float
    nhd95: float


def _as_binary_mask(mask, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D binary mask, got shape {array.shape}")
    if 0 in array.shape:
        raise ValueError(f"{name} must have non-empty spatial dimensions")
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain booleans or numeric 0/1 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(bool, copy=False)


def binary_surface(mask) -> np.ndarray:
    """Return the inner binary surface used by the existing nHD95 metric.

    This is deliberately not a contouring or connectivity-customizable helper:
    keeping SciPy's default erosion semantics is what makes pooled nHD95 values
    identical to :func:`selectseg.confidence.normalized_penalized_hd95`.
    """

    binary = _as_binary_mask(mask, name="mask")
    return binary & ~ndimage.binary_erosion(binary)


@dataclass(frozen=True, slots=True)
class PreparedBoundaryReference:
    """Surface geometry cached for one fixed binary reference mask.

    Use :meth:`from_mask` once per sample, then call :meth:`compare` for every
    thresholded candidate.  Non-empty references cache one EDT; empty references
    need none because the total empty-mask convention determines both losses.
    Cached arrays are read-only to keep comparisons stable if the caller later
    mutates the input mask.
    """

    shape: tuple[int, int]
    diagonal: float
    present: bool
    surface: np.ndarray | None
    distance_to_surface: np.ndarray | None

    @classmethod
    def from_mask(cls, reference) -> "PreparedBoundaryReference":
        """Validate and prepare a fixed reference mask."""

        reference = _as_binary_mask(reference, name="reference")
        shape = (int(reference.shape[0]), int(reference.shape[1]))
        diagonal = math.hypot(*shape)
        if not reference.any():
            return cls(
                shape=shape,
                diagonal=diagonal,
                present=False,
                surface=None,
                distance_to_surface=None,
            )

        surface = binary_surface(reference)
        distance_to_surface = ndimage.distance_transform_edt(~surface)
        surface.setflags(write=False)
        distance_to_surface.setflags(write=False)
        return cls(
            shape=shape,
            diagonal=diagonal,
            present=True,
            surface=surface,
            distance_to_surface=distance_to_surface,
        )

    def compare(self, candidate) -> BoundaryLosses:
        """Return ``(nhd, nhd95)`` against one candidate mask.

        A non-empty candidate uses one EDT.  The full surface HD is the maximum
        of the same pooled bidirectional nearest-surface distances whose 95th
        percentile defines pooled HD95.  Clipping at one is only a numerical
        safeguard: all in-frame Euclidean surface distances are smaller than
        ``hypot(height, width)``.
        """

        candidate = _as_binary_mask(candidate, name="candidate")
        if candidate.shape != self.shape:
            raise ValueError(
                "reference and candidate shapes differ: "
                f"{self.shape} != {candidate.shape}"
            )

        candidate_present = bool(candidate.any())
        if not self.present and not candidate_present:
            return BoundaryLosses(nhd=0.0, nhd95=0.0)
        if self.present != candidate_present:
            return BoundaryLosses(nhd=1.0, nhd95=1.0)

        if self.surface is None or self.distance_to_surface is None:
            raise AssertionError("non-empty reference is missing cached geometry")
        candidate_surface = binary_surface(candidate)
        distance_to_candidate = ndimage.distance_transform_edt(~candidate_surface)
        distances = np.concatenate(
            [
                distance_to_candidate[self.surface],
                self.distance_to_surface[candidate_surface],
            ]
        )
        nhd = float(np.max(distances)) / self.diagonal
        nhd95 = float(np.percentile(distances, 95)) / self.diagonal
        return BoundaryLosses(
            nhd=float(min(1.0, nhd)),
            nhd95=float(min(1.0, nhd95)),
        )


def prepare_boundary_reference(reference) -> PreparedBoundaryReference:
    """Prepare a fixed reference for repeated HD/HD95 comparisons."""

    return PreparedBoundaryReference.from_mask(reference)


def normalized_penalized_boundary_losses(reference, candidate) -> BoundaryLosses:
    """Convenience two-mask API returning ``(nhd, nhd95)``.

    Repeated-candidate callers should use :func:`prepare_boundary_reference` to
    retain the fixed reference EDT instead of rebuilding it for every pair.
    """

    return prepare_boundary_reference(reference).compare(candidate)
