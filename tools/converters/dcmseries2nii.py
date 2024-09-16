# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse
import sys
from pathlib import Path

import dicom2nifti
import SimpleITK as sitk
from tqdm import tqdm


def main(args):
    subjects = list(Path(args.inputdir).iterdir())

    # find the last dir inside each dir in the input dir
    for subject in tqdm(subjects):
        series = subject
        while series.is_dir():
            series = list(series.iterdir())[-1]
        series = series.parent

        outdir = Path(args.outdir) / f'{subject.name}'
        outdir.mkdir(parents=True, exist_ok=True)

        dicom2nifti.convert_directory(series, outdir, compression=True, reorient=True)

        # Get the nifti file, rename it to the subject name and move it to the outdir
        nifti_file = list(outdir.glob('*.nii.gz'))[0]
        outname = Path(args.outdir) / f'{subject.name}.nii.gz'
        nifti_file.rename(outname)

        if args.flip_ud:
            # flip up-down
            sitk.WriteImage(sitk.Flip(nifti_file, [False, True, False]), str(outname))

        if args.flip_lr:
            # flip left-right
            sitk.WriteImage(sitk.Flip(nifti_file, [True, False, False]), str(outname))

        # Remove the outdir
        for f in outdir.iterdir():
            f.unlink()
        outdir.rmdir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a series of DICOM files to a single DICOM file')
    parser.add_argument('inputdir', help='Input directory containing the DICOM series')
    parser.add_argument('outdir', help='Output directory for the single DICOM file')
    parser.add_argument('--flip-ud', action='store_true', help='Flip the output image up-down')
    parser.add_argument('--flip-lr', action='store_true', help='Flip the output image left-right')
    args = parser.parse_args()

    sys.exit(main(args))
