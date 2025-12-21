MRI Segmentation (SEG)
======================
The following publicly available datasets are natively supported within atommic for segmentation.

.. toctree::
   :maxdepth: 8

   brats2023adultglioma
   isles2022subacutestroke
   ../reconstruction/skmtea

Segmentation can be performed in an multiclass or a multilabel manner.
The `segmentation_mode` needs to be set either to `multiclass` or `multilabel` to ensure the chosen configurations
aligns with the segmentation approach. Make sure to add the background_class when `multiclass` is selected.

Multiclass vs. Multilabel Segmentation
--------------------------------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Feature
     - Multiclass
     - Multilabel

   * - **Pixel Constraint**
     - | **Each class is dependent**
       | (Only one class can be assigned per voxel)
     - | **Each class is independent**
       | (Multiple classes can be assigned per voxel)

   * - **Channel Output** (:math:`C`)
     - | :math:`N + 1`
       | (Includes explicit Background class)
     - | :math:`N`
       | (Background is implicit)

   * - **Activation Function**
     - | **Softmax**
       | :math:`\frac{e^{z_i}}{\sum e^{z_j}}` (Coupled probabilities)
     - | **Sigmoid**
       | :math:`\frac{1}{1 + e^{-z_i}}` (Independent probabilities)

   * - **Background Logic**
     - | **Explicit Class**
       | :math:`P(BG) = 1 - \sum P(Foreground)`
     - | **Implicit Absence**
       | All channels :math:`\approx 0`

   * - **Inference Decision**
     - | **Argmax**
       | (Select index with highest probability)
     - | **Thresholding**
       | (Select all indices where :math:`P > 0.5`)

   * - **Target Label Format**
     - | **Flat Integer Map** :math:`(H, W)`
       | *(or One-Hot)*
     - | **Stacked Binary Masks** :math:`(H, W, N)`

   * - **Loss Function**
     - | **Categorical Cross-Entropy**
       | (Penalizes the target class vs. all others)
     - | **Binary Cross-Entropy**
       | (Penalizes each channel independently)

   * - **Probability Space**
     - Joint Distribution (Sum = 1)
     - Independent Bernoulli Distributions
