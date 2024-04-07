# coding=utf-8
__author__ = "Dimitris Karkalousos"

from typing import Dict, List, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from atommic.collections.common.parts.fft import fft2, ifft2
from atommic.collections.common.parts.utils import check_stacked_complex, coil_combination_method, expand_op
from atommic.collections.reconstruction.nn.base import BaseMRIReconstructionModel
from atommic.collections.reconstruction.nn.crnn_base.crnn_block import GRUConv2d, RecurrentConvolutionalNetBlock
from atommic.core.classes.common import typecheck

__all__ = ["CRNNet"]


class CRNNet(BaseMRIReconstructionModel):
    """Implementation of the Convolutional Recurrent Neural Network, as presented in [Qin2019]_.

    References
    ----------
    .. [Qin2019] C. Qin, J. Schlemper, J. Caballero, A. N. Price, J. V. Hajnal and D. Rueckert, "Convolutional
        Recurrent Neural Networks for Dynamic MR Image Reconstruction," in IEEE Transactions on Medical Imaging, vol.
        38, no. 1, pp. 280-290, Jan. 2019, doi: 10.1109/TMI.2018.2863670.

    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Inits :class:`CRNNet`.

        Parameters
        ----------
        cfg : DictConfig
            Configuration.
        trainer : Trainer, optional
            PyTorch Lightning trainer. Default is ``None``.
        """
        super().__init__(cfg=cfg, trainer=trainer)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        self.no_dc = cfg_dict.get("no_dc")
        self.num_iterations = cfg_dict.get("num_iterations")

        self.reconstruction_module = RecurrentConvolutionalNetBlock(
            GRUConv2d(
                in_channels=2,
                out_channels=2,
                hidden_channels=cfg_dict.get("hidden_channels"),
                n_convs=cfg_dict.get("n_convs"),
                batchnorm=cfg_dict.get("batchnorm"),
            ),
            num_iterations=self.num_iterations,
            fft_centered=self.fft_centered,
            fft_normalization=self.fft_normalization,
            spatial_dims=self.spatial_dims,
            coil_dim=self.coil_dim,
            no_dc=self.no_dc,
        )

    # pylint: disable=arguments-differ
    @typecheck()
    def forward(
        self,
        y: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        initial_prediction: torch.Tensor,  # pylint: disable=unused-argument
        sigma: float = 1.0,  # pylint: disable=unused-argument
    ) -> List[torch.Tensor]:
        """Forward pass of :class:`CRNNet`.

        Parameters
        ----------
        y : torch.Tensor
            Subsampled k-space data. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]
        mask : torch.Tensor
            Subsampling mask. Shape [1, 1, n_x, n_y, 1]
        initial_prediction : torch.Tensor
            Initial prediction. Shape [batch_size, n_x, n_y, 2]
        sigma : float
            Noise level. Default is ``1.0``.

        Returns
        -------
        List of torch.Tensor
            List of the intermediate predictions for each cascade. Shape [batch_size, n_x, n_y].
        """
        predictions = self.reconstruction_module(y, sensitivity_maps, mask)
        return [self.process_intermediate_pred(x, sensitivity_maps) for x in predictions]

    def process_intermediate_pred(
        self, prediction: Union[List, torch.Tensor], sensitivity_maps: torch.Tensor
    ) -> torch.Tensor:
        """Process the intermediate prediction.

        Parameters
        ----------
        prediction : torch.Tensor
            Intermediate prediction. Shape [batch_size, n_coils, n_x, n_y, 2]
        sensitivity_maps : torch.Tensor
            Coil sensitivity maps. Shape [batch_size, n_coils, n_x, n_y, 2]

        Returns
        -------
        torch.Tensor
            Processed prediction. Shape [batch_size, n_x, n_y].
        """
        return check_stacked_complex(
            coil_combination_method(
                ifft2(
                    prediction,
                    centered=self.fft_centered,
                    normalization=self.fft_normalization,
                    spatial_dims=self.spatial_dims,
                ),
                sensitivity_maps,
                self.coil_combination_method,
                self.coil_dim,
            )
        )

    def process_reconstruction_loss(
        self,
        target: torch.Tensor,
        prediction: Union[list, torch.Tensor],
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
        attrs: Dict,
        r: int,
        loss_func: torch.nn.Module,
    ) -> torch.Tensor:
        """Processes the reconstruction loss for the CRNNet model. It differs from the base class in that it uses the
        intermediate predictions to compute the loss.

        Parameters
        ----------
        target : torch.Tensor
            Target data of shape [batch_size, n_x, n_y, 2].
        prediction : Union[list, torch.Tensor]
            Prediction(s) of shape [batch_size, n_x, n_y, 2].
        sensitivity_maps : torch.Tensor
            Sensitivity maps of shape [batch_size, n_coils, n_x, n_y, 2]. It will be used if self.ssdu is True, to
            expand the target and prediction to multiple coils.
        mask : torch.Tensor
            Mask of shape [batch_size, n_x, n_y, 2]. It will be used if self.ssdu is True, to enforce data consistency
            on the prediction.
        attrs : Dict
            Attributes of the data with pre normalization values.
        r : int
            The selected acceleration factor.
        loss_func : torch.nn.Module
            Loss function. Must be one of {torch.nn.L1Loss(), torch.nn.MSELoss(),
            atommic.collections.reconstruction.losses.ssim.SSIMLoss()}. Default is ``torch.nn.L1Loss()``.

        Returns
        -------
        loss: torch.FloatTensor
            If self.accumulate_loss is True, returns an accumulative result of all intermediate losses.
            Otherwise, returns the loss of the last intermediate loss.
        """
        # If kspace reconstruction loss is used, the target needs to be transformed to k-space.
        if self.kspace_reconstruction_loss:
            # If inputs are complex, then they need to be viewed as real.
            if target.shape[-1] != 2 and torch.is_complex(target):
                target = torch.view_as_real(target)
            # If SSDU is used, then the coil-combined inputs need to be expanded to multiple coils using the
            # sensitivity maps.
            if self.ssdu:
                target = expand_op(target, sensitivity_maps, self.coil_dim)
            # Transform to k-space.
            target = fft2(target, self.fft_centered, self.fft_normalization, self.spatial_dims)
            # Ensure loss inputs are both viewed in the same way.
            target = self.__abs_output__(target)
        elif not self.unnormalize_loss_inputs:
            # Ensure loss inputs are both viewed in the same way.
            target = self.__abs_output__(target)
            # Normalize inputs to [0, 1]
            target = torch.abs(target / torch.max(torch.abs(target)))

        def compute_reconstruction_loss(t, p, s):
            if self.unnormalize_loss_inputs:
                # we do the unnormalization here to avoid explicitly iterating through list of predictions, which
                # might be a list of lists.
                t, p, s = self.__unnormalize_for_loss_or_log__(t, p, s, attrs, r)

            # If kspace reconstruction loss is used, the target needs to be transformed to k-space.
            if self.kspace_reconstruction_loss:
                # If inputs are complex, then they need to be viewed as real.
                if p.shape[-1] != 2 and torch.is_complex(p):
                    p = torch.view_as_real(p)
                # If SSDU is used, then the coil-combined inputs need to be expanded to multiple coils using the
                # sensitivity maps.
                if self.ssdu:
                    p = expand_op(p, s, self.coil_dim)
                # Transform to k-space.
                p = fft2(p, self.fft_centered, self.fft_normalization, self.spatial_dims)
                # If SSDU is used, then apply the mask to the prediction to enforce data consistency.
                if self.ssdu:
                    p = p * mask
                # Ensure loss inputs are both viewed in the same way.
                p = self.__abs_output__(p)
            elif not self.unnormalize_loss_inputs:
                p = self.__abs_output__(p)
                # Normalize inputs to [0, 1]
                p = torch.abs(p / torch.max(torch.abs(p)))

            if "ssim" in str(loss_func).lower():
                return loss_func(
                    t.unsqueeze(dim=self.coil_dim),
                    p.unsqueeze(dim=self.coil_dim),
                    data_range=torch.tensor(
                        [max(torch.max(t).item(), torch.max(p).item()) - min(torch.min(t).item(), torch.min(p).item())]
                    )
                    .unsqueeze(dim=0)
                    .to(t.device),
                )
            return loss_func(t, p)

        iterations_weights = torch.logspace(-1, 0, steps=self.num_iterations).to(target.device)
        if self.num_echoes > 0:
            iterations_loss = [
                torch.mean(
                    torch.stack(
                        [
                            compute_reconstruction_loss(
                                target[echo].unsqueeze(0), iteration_pred[echo].unsqueeze(0), sensitivity_maps
                            )
                            for echo in range(target.shape[0])
                        ]
                    )
                ).to(target)
                for iteration_pred in prediction
            ]
        else:
            iterations_loss = [
                compute_reconstruction_loss(target, iteration_pred, sensitivity_maps) for iteration_pred in prediction
            ]
        loss = sum(x * w for x, w in zip(iterations_loss, iterations_weights)) / self.num_iterations
        return loss
