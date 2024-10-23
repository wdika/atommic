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
        subject_name = subject.name
        series = list(subject.iterdir())
        for se in series:
            series_subject_name = subject_name + f"_{se.name}"

            outdir = Path(args.outdir) / f'{series_subject_name}'
            outdir.mkdir(parents=True, exist_ok=True)

            # Get the nifti file, rename it to the subject name and move it to the outdir
            outname = Path(args.outdir) / f'{series_subject_name}.nii.gz'

            # if file exists skip
            if outname.exists():
                print(f'{outname} exists, skipping...')
                continue

            dicom2nifti.convert_directory(se, outdir, compression=True, reorient=True)

            nifti_file = list(outdir.glob('*.nii.gz'))[0]

            if args.flip_ud or args.flip_lr:
                # Read the NIFTI file
                image = sitk.ReadImage(str(nifti_file))
                if args.flip_ud:
                    # Flip the image up-down (along Y-axis)
                    image = sitk.Flip(image, [False, True, False])
                if args.flip_lr:
                    # Flip the image left-right (along X-axis)
                    image = sitk.Flip(image, [True, False, False])
                # Write the flipped image directly to the output filename
                sitk.WriteImage(image, str(outname))
            else:
                # If no flipping is needed, simply rename the file
                nifti_file.rename(outname)

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
