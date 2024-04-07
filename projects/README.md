# Advanced Toolbox for Multitask Medical Imaging Consistency (ATOMMIC)

## **Datasets**

**ATOMMIC** supports several public datasets for accelerated MRI reconstruction, MRI segmentation, and quantitative
imaging, as well as multitasking, i.e. training a model to perform reconstruction and segmentation simultaneously.

Private datasets can also be used with this repo, but the data must be converted to the appropriate format.
The preferred format is HDF5, but NIfTI is also supported for segmentation. Data can be stored either as 3D volumes or
2D slices. You can check the preprocessing scripts for each dataset to see how the data should be formatted.

- For reconstruction, it's best to store the data should with the following dimensions:
[num_slices, num_coils, height, width].
- For segmentation, the data should be stored with the following dimensions: [num_slices, height, width]. Labels can be
stored as either one-hot or categorical.

You can extend the dataloaders for the corresponding task to support your own dataset.

On each of the following project folders, you can find the corresponding preprocessing script, which can be used to
convert the data to the appropriate format.

### **Quantitative Imaging**

For quantitative imaging, the following public datasets are supported:
- [AHEAD](qMRI/AHEAD).

### **Reconstruction**

For accelerated MRI reconstruction, the following public datasets are supported:
- [CC359](REC/CC359),
- [fastMRI Brains Multicoil](REC/fastMRIBrainsMulticoil),
- [fastMRI Knees Multicoil](REC/fastMRIKneesMulticoil),
- [fastMRI Knees Singlecoil](REC/fastMRIKneesSinglecoil),
- [SKM-TEA](REC/SKMTEA),
- [Stanford Knees](REC/StanfordKnees2019).

### **Segmentation**

For MRI segmentation, the following public datasets are supported:
- [BraTS2023AdultGlioma](SEG/BraTS2023AdultGlioma),
- [ISLES2022SubAcuteStroke](SEG/ISLES2022SubAcuteStroke),
- [SKM-TEA](SEG/SKMTEA).


## **Models**

**ATOMMIC** supports several models for accelerated MRI reconstruction, MRI segmentation, and quantitative imaging, as
well as multitasking. Please check [here](../README.md) the list of supported models.

On each project folder, you can find the corresponding model configuration file, which can be used to train and test
the model. You only need to change the `data paths` and `output paths` to the appropriate paths on your system. You can
also change the `model parameters` to change the model architecture and hyperparameters.

## **Training/Testing**

To train/test a model, you can use the following command:

```bash
atommic run -c path-to-config-file
```

## **Reproducing the ATOMMIC paper**
ATOMMIC paper is fully reproducible. Please check [here](ATOMMIC_paper/README.md) for more information.
