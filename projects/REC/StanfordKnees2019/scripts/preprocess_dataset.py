# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import os
from pathlib import Path

import h5py
import ismrmrd
import numpy as np
from tqdm import tqdm


def ismrmrd_to_np(filename):
    """
    Read ISMRMRD data file to numpy array.

    Taken from https://github.com/iasonsky/meddlr/blob/main/datasets/format_mridata_org.py

    Parameters
    ----------
    filename : str
        The path to the ISMRMRD file.

    Returns
    -------
    kspace : np.ndarray
        The k-space data.
    """
    dataset = ismrmrd.Dataset(filename, create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dataset.read_xml_header())
    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_channels = header.acquisitionSystemInformation.receiverChannels

    try:
        rec_std = dataset.read_array("rec_std", 0)
        rec_weight = 1.0 / (rec_std**2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
        print("Using rec std...")
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)
    kspace = np.zeros([num_channels, num_slices, num_ky, num_kx], dtype=np.complex64)
    num_acq = dataset.number_of_acquisitions()

    for i in tqdm(range(num_acq)):
        acq = dataset.read_acquisition(i)
        i_ky = acq.idx.kspace_encode_step_1  # pylint: disable=no-member
        i_slice = acq.idx.slice  # pylint: disable=no-member
        data = np.matmul(opt_mat.T, acq.data)
        kspace[:, i_slice, i_ky, :] = data * ((-1) ** i_slice)

    dataset.close()

    return kspace.astype(np.complex64)


def main(args):
    output_dir = Path(args.output_path)
    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    files = list(Path(args.data_path).iterdir())

    for fname in tqdm(files):
        kspace = ismrmrd_to_np(fname)
        kspace = np.moveaxis(kspace, 0, 1)

        # save the kspace as h5py file
        with h5py.File(output_dir / f"{fname.stem}.h5", "w") as f:
            f.create_dataset("kspace", data=kspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    main(args)
