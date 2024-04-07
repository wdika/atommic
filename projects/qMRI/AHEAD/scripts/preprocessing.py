# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import os
from pathlib import Path
from typing import Tuple

import h5py
import ismrmrd
import numpy as np
from tqdm import tqdm


def preprocess_ahead_raw_data(raw_data_file: str) -> np.ndarray:
    """
    Preprocess the raw data of the AHEAD dataset.

    Parameters
    ----------
    raw_data_file : str
        Path to the raw data and coil sensitivities of the AHEAD dataset.

    Returns
    -------
    kspace: np.ndarray
        The k-space data.
    """
    dataset = ismrmrd.Dataset(raw_data_file, "dataset", create_if_needed=False)
    number_of_acquisitions = dataset.number_of_acquisitions()

    # find the first no noise scan
    first_scan = 0
    for i in tqdm(range(number_of_acquisitions)):
        head = dataset.read_acquisition(i).getHead()
        if head.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            first_scan = i
            break

    meas = []
    for i in tqdm(range(first_scan, number_of_acquisitions)):
        meas.append(dataset.read_acquisition(i))

    hdr = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())

    # Matrix size
    enc = hdr.encoding[0]
    enc_Nx = enc.encodedSpace.matrixSize.x
    enc_Ny = enc.encodedSpace.matrixSize.y
    enc_Nz = enc.encodedSpace.matrixSize.z

    nCoils = hdr.acquisitionSystemInformation.receiverChannels

    nslices = enc.encodingLimits.slice.maximum + 1 if enc.encodingLimits.slice is not None else 1
    nreps = enc.encodingLimits.repetition.maximum + 1 if enc.encodingLimits.repetition is not None else 1
    ncontrasts = enc.encodingLimits.contrast.maximum + 1 if enc.encodingLimits.contrast is not None else 1

    # initialize k-space array
    Kread = np.zeros((enc_Nx, enc_Ny, enc_Nz, nCoils), dtype=np.complex64)

    # Select the appropriate measurements from the data
    for acq in tqdm(meas):
        head = acq.getHead()
        if head.idx.contrast == ncontrasts - 1 and head.idx.repetition == nreps - 1 and head.idx.slice == nslices - 1:
            ky = head.idx.kspace_encode_step_1
            kz = head.idx.kspace_encode_step_2
            Kread[:, ky, kz, :] = np.transpose(acq.data, (1, 0))

    return Kread


def preprocess_ahead_coil_sensitivities(coil_sensitivities_file: str) -> np.ndarray:
    """
    Preprocess the coil sensitivities of the AHEAD dataset.

    Parameters
    ----------
    coil_sensitivities_file : str
        Path to the coil sensitivities of the AHEAD dataset.

    Returns
    -------
    coil_sensitivities: np.ndarray
        The coil sensitivities.
    """
    # load the coil sensitivities
    coil_sensitivities = h5py.File(coil_sensitivities_file, "r")

    # get the coil sensitivities
    coil_sensitivities_real = np.array(coil_sensitivities["0real"])
    coil_sensitivities_imag = np.array(coil_sensitivities["1imag"])
    coil_sensitivities = coil_sensitivities_real + 1j * coil_sensitivities_imag

    # transpose to get the correct shape, i.e. (x, y, z, coils)
    coil_sensitivities = np.transpose(coil_sensitivities, (3, 2, 1, 0))

    return coil_sensitivities


def get_plane(data: np.ndarray, data_on_kspace: bool = True, plane: str = "sagittal") -> np.ndarray:
    """
    Get the given plane from the data.

    Parameters
    ----------
    data : np.ndarray
        The data to get the plane from.
    data_on_kspace : bool, optional
        Whether the data is on the kspace or not. The default is True.
    plane : str, optional
        The plane to get the kspace and coil sensitivities from. The default is "sagittal".

    Returns
    -------
    data: np.ndarray
        The data of the given plane.
    """
    if not data_on_kspace:
        data = np.fft.fftn(data, axes=(0, 1, 2))

    if plane == "axial":
        data = np.transpose(data, (2, 0, 1, 3))
    elif plane == "coronal":
        data = np.transpose(data, (1, 0, 2, 3))

    # all planes need to be rotated by 90 degrees in x-y to get the correct orientation
    data = np.rot90(data, k=1, axes=(1, 2))

    if not data_on_kspace:
        data = np.fft.ifftn(data, axes=(0, 1, 2))

    return data


def compute_targets(
    kspace: np.ndarray, coil_sensitivities: np.ndarray, coil_dim: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the target images from the kspace and coil sensitivities.

    Parameters
    ----------
    kspace : np.ndarray
        The kspace.
    coil_sensitivities : np.ndarray
        The coil sensitivities.
    coil_dim : int, optional
        The dimension of the coil sensitivities. The default is -1.

    Returns
    -------
    image_space : np.ndarray
        The image space.
    target_image : np.ndarray
        The target image.
    """
    # get the image space
    image_space = np.fft.fftshift(
        np.fft.ifftn(np.fft.fftshift(kspace, axes=(0, 1, 2)), axes=(0, 1, 2)), axes=(0, 1, 2)
    )

    # compute the target
    target = np.sum(image_space * np.conj(coil_sensitivities), axis=coil_dim)

    return image_space, target


def save_data(
    image_space: np.ndarray, coil_sensitivities: np.ndarray, target: np.ndarray, output_path: Path, filename: str
):
    """
    Save the data.

    Parameters
    ----------
    image_space : np.ndarray
        The image space.
    coil_sensitivities : np.ndarray
        The coil sensitivities.
    target : np.ndarray
        The target image.
    output_path : Path
        The output path.
    filename : str
        The filename.
    """
    # we need to move the coils dimension to dimension 2 and get kspace
    image_space = np.moveaxis(image_space, -1, 2)
    # we need to move the coils dimension to dimension 1 and get coil sensitivities
    coil_sensitivities = np.moveaxis(coil_sensitivities, -1, 1)

    # get kspace
    kspace = np.fft.fftn(image_space, axes=(-2, -1))
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))

    if not os.path.exists(output_path):
        output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path / f"{filename}.h5", "w") as f:
        f.create_dataset("kspace", data=kspace.astype(np.complex64))
        f.create_dataset("sensitivity_map", data=coil_sensitivities.astype(np.complex64))
        f.create_dataset("target", data=target.astype(np.complex64))


def main(args):
    # get all files
    files = list(Path(args.data_path).iterdir())
    # get the fnames
    fnames = [
        str(file).rsplit("/", maxsplit=1)[-1].split("_")[1].split(".")[0] for file in files if "coilsens" in file.name
    ]

    plane = args.plane

    # iterate over all subjects
    for fname in fnames:
        print(f"Processing subject {fname}...")

        # get all files for this subject from files
        subject_files = [file for file in files if fname in file.name]
        raw_data_files = [file for file in subject_files if "coilsens" not in file.name and "inv1" not in file.name]
        raw_data_files.sort()

        # preprocess the raw data
        kspaces = [preprocess_ahead_raw_data(str(x)) for x in raw_data_files]
        kspaces = [get_plane(x, data_on_kspace=True, plane=plane) for x in kspaces]

        # preprocess the coil sensitivities
        coil_sensitivities_file = [file for file in subject_files if "coilsens" in file.name][0]
        coil_sensitivities = preprocess_ahead_coil_sensitivities(str(coil_sensitivities_file))
        coil_sensitivities = get_plane(coil_sensitivities, data_on_kspace=False, plane=plane)

        # compute the image spaces and targets
        image_spaces = []
        targets = []
        for x in kspaces:
            image_space, target = compute_targets(x, coil_sensitivities, coil_dim=-1)
            image_spaces.append(image_space)
            targets.append(target)
        image_space = np.stack(image_spaces, axis=1)
        target = np.stack(targets, axis=1)

        slice_range = args.slice_range
        if slice_range is not None:
            image_space = image_space[slice_range[0] : slice_range[1]]
            coil_sensitivities = coil_sensitivities[slice_range[0] : slice_range[1]]
            target = target[slice_range[0] : slice_range[1]]

        # save the data to disk
        save_data(
            image_space,
            coil_sensitivities,
            target,
            Path(args.output_path),
            f"mp2rageme_{fname}_{plane}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default="data/ahead_data")
    parser.add_argument("--output_path", type=Path, default="data/ahead_data_preprocessed")
    parser.add_argument("--plane", type=str, default="axial")
    parser.add_argument("--slice_range", default=None, type=int, nargs="+")
    args = parser.parse_args()
    main(args)
