Metrics
=======

``ATOMMIC`` provides a number of metrics for each task to evaluate the performance of the models. The metrics are
implemented as classes that can be instantiated and called with the desired inputs. Depending on the chosen task, the
corresponding metrics will be also logged on the selected logger.

In `tools <https://github.com/wdika/atommic/tree/main/tools/evaluation>`_, you can find scripts that allows you to
evaluate the performance of a model on a dataset. The scripts take as input the ground truth and the predictions of the
model and compute the metrics for each task.

The metrics are implemented in the following modules:

* :func:`~atommic.collections.reconstruction.metrics.reconstruction_metrics.mse`:
    Mean Squared Error (MSE) metric for ``reconstruction``, ``quantitative``, and ``multitask`` tasks.

* :func:`~atommic.collections.reconstruction.metrics.reconstruction_metrics.nmse`:
    Normalized Mean Squared Error (NMSE) metric for ``reconstruction``, ``quantitative``, and ``multitask`` tasks.

* :func:`~atommic.collections.reconstruction.metrics.reconstruction_metrics.psnr`:
    Peak Signal-to-Noise Ratio (PSNR) metric for ``reconstruction``, ``quantitative``, and ``multitask`` tasks.

* :func:`~atommic.collections.reconstruction.metrics.reconstruction_metrics.ssim`:
    Structural Similarity Index (SSIM) metric for ``reconstruction``, ``quantitative``, and ``multitask`` tasks.

* :func:`~atommic.collections.reconstruction.metrics.reconstruction_metrics.haarpsi`:
    Structural Similarity Index (SSIM) metric for ``reconstruction``, ``quantitative``, and ``multitask`` tasks.

* :class:`~atommic.collections.reconstruction.metrics.reconstruction_metrics.ReconstructionMetrics`:
    Class that wraps all the metrics for ``reconstruction``, ``quantitative``, and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.asd`:
    Average Surface Distance (ASD) metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.binary_cross_entropy_with_logits_metric`:
    Binary Cross Entropy with Logits (BCE) metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.dice_metric`:
    Dice metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.f1_per_class_metric`:
    F1 per class metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.hausdorff_distance_metric`:
    Hausdorff Distance (HD) metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.hausdorff_distance_95_metric`:
    95th percentile of the Hausdorff Distance (HD95) metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.iou_metric`:
    Intersection over Union (IoU) metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.precision_metric`:
    Precision metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.recall_metric`:
    Recall metric for ``segmentation`` and ``multitask`` tasks.

* :func:`~atommic.collections.segmentation.metrics.segmentation_metrics.surface_distances`:
    Surface Distances (SD) metric for ``segmentation`` and ``multitask`` tasks.

* :class:`~atommic.collections.segmentation.metrics.segmentation_metrics.SegmentationMetrics`:
    Class that wraps all the metrics for ``segmentation`` and ``multitask`` tasks.
