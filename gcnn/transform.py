"""
Apply D6 transformations to convolution kernels on the hexagonal lattice.
"""

import torch

from pac.matrix_manipulation import reflect, rotate


def hexagon_axial(ksize: int) -> torch.Tensor:
    """
    Construct a hexagonal support mask within a square kernel array.

    The mask includes a kernel weight when ``max(|x|, |y|, |x - y|)`` is at most ``ksize // 2`` in axial coordinates.

    Parameters
    ----------
    ksize : int
        Side length of the square kernel array. The value must be odd.

    Returns
    -------
    torch.Tensor of shape (ksize, ksize)
        Floating-point mask equal to 1.0 inside the hexagonal support and 0.0 elsewhere.
    """
    assert ksize % 2 == 1
    if ksize == 1:
        return torch.ones((1, 1))

    mask = torch.zeros((ksize, ksize))
    center = ksize // 2
    for i in range(ksize):
        for j in range(ksize):
            y = i - center
            x = j - center
            if max(abs(x), abs(y), abs(x - y)) <= center:
                mask[i, j] = 1.0
    return mask.float()


def d6_all_transforms(weight: torch.Tensor) -> torch.Tensor:
    """
    Apply all 12 D6 transformations to a batch of spatial kernels.

    The 12 outputs are in the order of `pac.geometry.d6_orbit`.

    Parameters
    ----------
    weight : torch.Tensor of shape (num_kernels, kernel_size, kernel_size)
        Spatial kernels to transform.

    Returns
    -------
    torch.Tensor of shape (12, num_kernels, kernel_size, kernel_size)
        Kernels transformed under each D6 group element.
    """
    center = (weight.shape[-2] // 2, weight.shape[-1] // 2)
    transforms = []

    # ======[ First 6: Rotations ]======
    rotated = weight.clone()
    transforms.append(rotated)
    for k in range(1, 6):
        rotated = rotate(
            rotated.clone(), center=center
        )  # Assumes PyTorch differentiable rotate
        transforms.append(rotated)

    # ======[ Next 6: Reflect + Rotations ]======
    reflected = reflect(weight.clone(), center=center)
    rotated_reflected = reflected.clone()
    transforms.append(rotated_reflected)
    for k in range(1, 6):
        rotated_reflected = rotate(rotated_reflected.clone(), center=center)
        transforms.append(rotated_reflected)

    return torch.stack(transforms, dim=0)


def precompute_d6_indices(
    kernel_size: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """
    Precompute flattened indices for all D6 kernel transformations.

    Each row contains the source index for every position in a transformed kernel.

    Parameters
    ----------
    kernel_size : int
        Side length of the square kernel array. The value must be odd.
    device : str or torch.device, default 'cpu'
        Device on which the index tensor is created.

    Returns
    -------
    torch.Tensor of shape (12, kernel_size * kernel_size)
        One flattened permutation for each D6 group element.
    """
    indices = torch.arange(
        kernel_size * kernel_size, dtype=torch.float32, device=device
    ).view(1, kernel_size, kernel_size)

    transformed_indices = d6_all_transforms(indices)  # Shape: (12, 1, K, K)

    return transformed_indices.view(12, kernel_size * kernel_size).long()


def fast_d6_all_transforms(
    weight: torch.Tensor, perm_indices: torch.Tensor
) -> torch.Tensor:
    """
    Apply all D6 kernel transformations from precomputed indices.

    The result is equivalent to `d6_all_transforms`, but the transformed values are obtained by indexing the flattened
    kernels.

    Parameters
    ----------
    weight : torch.Tensor of shape (num_kernels, kernel_size, kernel_size)
        Spatial kernels to transform.
    perm_indices : torch.Tensor of shape (12, kernel_size * kernel_size)
        Permutation indices returned by `precompute_d6_indices`.

    Returns
    -------
    torch.Tensor of shape (12, num_kernels, kernel_size, kernel_size)
        Kernels transformed under each D6 group element.
    """
    N, K, _ = weight.shape
    flat_weight = weight.view(N, K * K)

    transformed = flat_weight[:, perm_indices].permute(1, 0, 2).view(12, N, K, K)

    return transformed
