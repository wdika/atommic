# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from https://github.com/bduffy0/motion-correction/blob/master/layer/motion_sim.py

import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from atommic.collections.common.parts import utils


def get_center_rect(image: torch.tensor, center_percentage: float = 0.02, dim: int = 0) -> torch.tensor:
    """Get a center rectangle of a given dimension.

    Parameters
    ----------
    image : torch.tensor
        The image to get the center rectangle from.
    center_percentage : float
        The percentage of the image to take as the center rectangle.
    dim : int
        The dimension to take the center rectangle from.

    Returns
    -------
    torch.tensor
        The center rectangle.
    """
    shape = (image[0].item(), image[1].item())
    mask = torch.zeros(shape)
    half_pct = center_percentage / 2
    center = [int(x / 2) for x in shape]
    mask = torch.swapaxes(mask, 0, dim)
    mask[:, center[1] - math.ceil(shape[1] * half_pct) : math.ceil(center[1] + shape[1] * half_pct)] = 1
    mask = torch.swapaxes(mask, 0, dim)
    return mask


def segment_array_by_locs(shape: Sequence[int], locations: Sequence[int]) -> torch.tensor:
    """Generate a segmentation mask based on a list of locations.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the array to segment.
    locations : Sequence[int]
        The locations to segment the array into.

    Returns
    -------
    torch.tensor
        The segmentation mask.
    """
    mask_out = torch.zeros(torch.prod(shape), dtype=int)
    for i in range(len(locations) - 1):
        loc = [locations[i], locations[i + 1]]
        mask_out[loc[0] : loc[1]] = i + 1
    return mask_out.reshape(shape)


def segments_to_random_indices(shape: Sequence[int], seg_lengths: Sequence[int]) -> torch.tensor:
    """Generate a segmentation mask based on a list of locations.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the array to segment.
    seg_lengths : Sequence[int]
        The lengths of the segments to generate.

    Returns
    -------
    torch.tensor
        The segmentation mask.
    """
    random_indices = torch.randint(low=0, high=shape, size=(sum(seg_lengths),)).sort()[0]
    seg_mask = torch.zeros(shape).type(torch.int)
    seg_new_indices = torch.cumsum(torch.tensor(seg_lengths), 0).tolist()
    seg_new_indices = [0] + seg_new_indices
    for i in range(len(seg_new_indices) - 1):
        seg_mask[random_indices[seg_new_indices[i] : seg_new_indices[i + 1]]] = i + 1
    return seg_mask


def segments_to_random_blocks(shape: Sequence[int], seg_lengths: Sequence[int]) -> torch.tensor:
    """Generate a segmentation mask based on a list of locations.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the array to segment.
    seg_lengths : Sequence[int]
        The lengths of the segments to generate.

    Returns
    -------
    torch.tensor
        The segmentation mask.
    """
    seg_mask = torch.zeros(shape).type(torch.int)
    seg_lengths_sorted = sorted(seg_lengths, reverse=True)
    for i, seg_len in enumerate(seg_lengths_sorted):
        loc = torch.randint(low=0, high=seg_mask.size()[0], size=(1,))
        while (sum(seg_mask[loc : loc + seg_len]) != 0) or (loc + seg_len > seg_mask.size()[0]):
            loc = torch.randint(low=0, high=seg_mask.size()[0], size=(1,))
        seg_mask[loc : loc + seg_len] = i + 1
    return seg_mask


def create_rand_partition(im_length: int, num_segments: int):
    """Create a random partition of an array.

    Parameters
    ----------
    im_length : int
        The length of the array to partition.
    num_segments : int
        The number of segments to partition the array into.

    Returns
    -------
    list
        The partition locations.
    """
    rand_segment_locs = sorted(list(torch.randint(im_length, size=(num_segments,))))
    rand_segment_locs[0] = 0
    rand_segment_locs[-1] = None
    return rand_segment_locs


def create_rotation_matrix_3d(angles: Sequence[float]) -> torch.tensor:
    """Create a 3D rotation matrix.

    Parameters
    ----------
    angles : Sequence[float]
        The angles to rotate the matrix by.

    Returns
    -------
    torch.tensor
        The rotation matrix.
    """
    mat1 = torch.FloatTensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angles[0]), math.sin(angles[0])],
            [0.0, -math.sin(angles[0]), math.cos(angles[0])],
        ]
    )
    mat2 = torch.FloatTensor(
        [
            [math.cos(angles[1]), 0.0, -math.sin(angles[1])],
            [0.0, 1.0, 0.0],
            [math.sin(angles[1]), 0.0, math.cos(angles[1])],
        ]
    )
    mat3 = torch.FloatTensor(
        [
            [math.cos(angles[2]), math.sin(angles[2]), 0.0],
            [-math.sin(angles[2]), math.cos(angles[2]), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (mat1 @ mat2) @ mat3


def translate_kspace(freq_domain: torch.tensor, translations: torch.tensor) -> torch.tensor:
    """Translate a k-space array.

    Parameters
    ----------
    freq_domain : torch.tensor
        The k-space array to translate.
    translations : torch.tensor
        The translations to apply to the k-space array.

    Returns
    -------
    torch.tensor
        The translated k-space array.
    """
    lin_spaces = [torch.linspace(-0.5, 0.5, x) for x in freq_domain.shape[:-1]]
    meshgrids = torch.meshgrid(*lin_spaces, indexing="ij")
    grid_coords = torch.stack([mg.flatten() for mg in meshgrids], 0)
    phase_shift = torch.multiply(grid_coords, translations).sum(axis=0)  # phase shift is added
    exp_phase_shift = torch.exp(-2j * math.pi * phase_shift).to(freq_domain.device)
    motion_kspace = torch.view_as_real(
        torch.multiply(exp_phase_shift, torch.view_as_complex(freq_domain).flatten()).reshape(freq_domain.shape[:-1])
    )

    return motion_kspace


class MotionSimulation:
    """Simulates random translations and rotations in the frequency domain.

    Examples
    --------
    >>> from atommic.collections.motioncorrection.parts import MotionSimulation
    >>> import torch
    >>> motion_simulation = MotionSimulation()
    >>> kspace = torch.randn(1, 1, 256, 256, 2)
    >>> motion_kspace = motion_simulation(kspace)
    >>> motion_kspace.shape
    torch.Size([1, 1, 256, 256, 2])
    """

    def __init__(
        self,
        motion_type: str = "piecewise_transient",
        angle: float = 0,
        translation: float = 10,
        center_percentage: float = 0.02,
        motion_percentage: Sequence[float] = (15, 20),
        num_segments: int = 8,
        random_num_segments: bool = False,
        non_uniform: bool = False,
        spatial_dims: Sequence[int] = (-2, -1),
    ):
        """Inits :class:`MotionSimulation`.

        Parameters
        ----------
        motion_type : str
            The motion_type of motion to simulate.
        angle : float
            The angle to rotate the k-space array by.
        translation : float
            The translation to apply to the k-space array.
        center_percentage : float
            The percentage of the k-space array to center the motion.
        motion_percentage : Sequence[float]
            The percentage of the k-space array to apply the motion.
        num_segments : int
            The number of segments to partition the k-space array into.
        random_num_segments : bool
            Whether to randomly generate the number of segments.
        non_uniform : bool
            Whether to use non-uniform sampling.
        spatial_dims : Sequence[int]
            The spatial dimensions to apply the motion to.
        """
        self.motion_type = motion_type
        self.angle, self.translation = angle, translation
        self.center_percentage = center_percentage

        if motion_percentage[1] == motion_percentage[0]:
            motion_percentage[1] += 1  # type: ignore
        elif motion_percentage[1] < motion_percentage[0]:
            raise ValueError("Uniform is not defined when low>= high.")

        self.motion_percentage = motion_percentage

        self.spatial_dims = spatial_dims
        self._spatial_dims = random.choice(spatial_dims)

        self.num_segments = num_segments
        self.random_num_segments = random_num_segments

        if non_uniform:
            raise NotImplementedError("NUFFT is not implemented. This is a feature to be added in the future.")

        self.trajectory = None
        self.params: Dict[Any, Any] = {}

    def _calc_dimensions(self, shape):
        """Calculate the dimensions to apply the motion to.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the image.

        Returns
        -------
        Sequence[int]
            The dimensions to apply the motion to.
        """
        pe_dims = [0, 1, 2]
        pe_dims.pop(self._spatial_dims)
        self.phase_encoding_dims = pe_dims
        shape = list(shape)
        if shape[-1] == 2:
            shape = shape[:-1]
        self.shape = shape.copy()
        shape.pop(self._spatial_dims)
        self.phase_encoding_shape = torch.tensor(shape)
        self.num_phase_encoding_steps = self.phase_encoding_shape[0] * self.phase_encoding_shape[1]
        self._spatial_dims = len(self.shape) - 1 if self._spatial_dims == -1 else self._spatial_dims

    def _generate_random_segments(self):
        """Generate random segments."""
        pct_corrupt = torch.distributions.Uniform(*[x / 100 for x in self.motion_percentage]).sample((1, 1))

        corrupt_matrix_shape = torch.tensor([int(x * math.sqrt(pct_corrupt)) for x in self.phase_encoding_shape])

        if torch.prod(corrupt_matrix_shape) == 0:
            corrupt_matrix_shape = [1, 1]

        if self.motion_type in {"gaussian"}:
            num_segments = torch.prod(corrupt_matrix_shape)
        else:
            if not self.random_num_segments:
                num_segments = self.num_segments
            else:
                num_segments = random.randint(1, self.num_segments)

        # segment a smaller vector occupying pct_corrupt percent of the space
        if self.motion_type in {"piecewise_transient", "piecewise_constant"}:
            seg_locs = create_rand_partition(torch.prod(corrupt_matrix_shape), num_segments=num_segments)
        else:
            seg_locs = list(range(num_segments))

        rand_segmentation = segment_array_by_locs(shape=torch.prod(corrupt_matrix_shape), locations=seg_locs)

        seg_lengths = [(rand_segmentation == seg_num).sum() for seg_num in torch.unique(rand_segmentation)]

        # assign segments to a vector with same number of elements as pe-steps
        if self.motion_type in {"piecewise_transient", "gaussian"}:
            seg_vector = segments_to_random_indices(torch.prod(self.phase_encoding_shape), seg_lengths)
        else:
            seg_vector = segments_to_random_blocks(torch.prod(self.phase_encoding_shape), seg_lengths)

        # reshape to phase encoding shape with a random order
        reshape_order = random.choice(["F", "C"])

        if reshape_order == "F":
            seg_array = utils.reshape_fortran(
                seg_vector, (self.phase_encoding_shape[0].item(), self.phase_encoding_shape[1].item())
            )
        else:
            seg_array = seg_vector.reshape((self.phase_encoding_shape[0].item(), self.phase_encoding_shape[1].item()))

        self.order = reshape_order

        # mask center k-space
        mask_not_including_center = (
            get_center_rect(
                self.phase_encoding_shape,
                center_percentage=self.center_percentage,
                dim=1 if reshape_order == "C" else 0,
            )
            == 0
        )

        self.seg_array = seg_array * mask_not_including_center
        self.num_segments = num_segments

    def _get_motion_trajectory(self, translation_rotation=None, random_segments=True):
        """Obtain a motion trajectory.

        Returns
        -------
        torch.tensor
            The random trajectory.
        """

        if random_segments:
            self._generate_random_segments()
        else:
            raise NotImplementedError("Custom segments (masks) not supported")

        if not translation_rotation:
            translations, rotations = self._simulate_random_trajectory()
        else:
            (translations, rotations) = translation_rotation
            if translations.shape[0] != self.num_segments:
                translations = torch.cat((torch.tensor([[0, 0, 0]]), translations), dim=0)
            if rotations.shape[0] != self.num_segments:
                rotations = torch.cat((torch.tensor([[0, 0, 0]]), rotations), dim=0)

        # if segment==0, then no motion
        translations[0, :] = 0
        rotations[0, :] = 0

        # lookup values for each segment
        translations_pe = [translations[:, i][self.seg_array.long()] for i in range(3)]
        rotations_pe = [rotations[:, i][self.seg_array.long()] for i in range(3)]

        # reshape and convert to radians
        translations = torch.stack(
            [torch.broadcast_to(x.unsqueeze(self._spatial_dims), self.shape) for x in translations_pe], 0
        )
        rotations = torch.stack(
            [torch.broadcast_to(x.unsqueeze(self._spatial_dims), self.shape) for x in rotations_pe], 0
        )

        rotations = rotations * (math.pi / 180.0)  # convert to radians

        self.translations = translations.reshape(3, -1)
        self.rotations = rotations.reshape(3, -1).reshape(3, -1)

    def _simulate_random_trajectory(self):
        """Simulate a random trajectory."""
        # generate random translations and rotations
        rand_translations = torch.distributions.normal.Normal(loc=0, scale=self.translation).sample(
            (self.num_segments, 3)
        )
        rand_rotations = torch.distributions.normal.Normal(loc=0, scale=self.angle).sample((self.num_segments, 3))
        return rand_translations, rand_rotations

    def forward(
        self,
        kspace,
        translations_rotations: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        apply_backward_transform: bool = False,  # pylint: disable=unused-argument
        apply_forward_transform: bool = False,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass of :class:`MotionSimulation`.

        Parameters
        ----------
        kspace : torch.Tensor
            The kspace to apply the motion to.
        translations_rotations : Optional[Tuple[torch.Tensor, torch.Tensor]]
            The translations and rotations to apply to the kspace. If None, a random trajectory is generated.
        apply_backward_transform : bool
            Placeholder for the backward transform. Generalizes the Composer, but not used.
        apply_forward_transform : bool
            Placeholder for the forward transform. Generalizes the Composer, but not used.

        Returns
        -------
        torch.Tensor
            The kspace with the motion applied.
        """
        self._calc_dimensions(kspace.shape)
        self._get_motion_trajectory(translations_rotations)

        motion_kspace = translate_kspace(freq_domain=kspace, translations=self.translations)

        return motion_kspace

    def __call__(self, *args, **kwargs):
        """Call :class:`MotionSimulation`."""
        return self.forward(*args, **kwargs)
