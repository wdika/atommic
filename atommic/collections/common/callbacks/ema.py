# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py

import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class EMA(Callback):
    """Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    .. note::
        Extends :class:`pytorch_lightning.callbacks.Callback`.

    Examples
    --------
    >>> from atommic.collections.common.callbacks.ema import EMA
    >>> ema = EMA(decay=0.9999, validate_original_weights=False, every_n_steps=1, cpu_offload=False)
    >>> trainer = Trainer(callbacks=[ema])
    >>> trainer.fit(model)
    >>> trainer.test(model)
    """

    def __init__(
        self,
        decay: float,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ):
        """Inits :class:`EMA`.

        Parameters
        ----------
        decay : float
            The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights : bool
            Validate the original weights, as apposed to the EMA weights. Default is ``False``.
        every_n_steps : int
            Apply EMA every N steps. Default is ``1``.
        cpu_offload : bool
            Offload weights to CPU. Default is ``False``.
        """
        super().__init__()
        if not 0 <= decay <= 1:
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Initialize the EMA weights.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        pl_module : pl.LightningModule
            PyTorch Lightning Module.
        """
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Swap the model weights.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        pl_module : pl.LightningModule
            PyTorch Lightning Module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Swap back the model weights.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        pl_module : pl.LightningModule
            PyTorch Lightning Module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Swap the model weights.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        pl_module : pl.LightningModule
            PyTorch Lightning Module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Swap back the model weights.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        pl_module : pl.LightningModule
            PyTorch Lightning Module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: "pl.Trainer") -> bool:
        """Check if we should validate the EMA weights.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        """
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: "pl.Trainer") -> bool:
        """Check if EMA has been initialized.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        """
        return any(isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers)

    def swap_model_weights(self, trainer: "pl.Trainer", saving_ema_model: bool = False):
        """Swaps the model weights with the EMA weights.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        saving_ema_model : bool, optional
            Whether we are saving the EMA model. Default is ``False``.
        """
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer"):
        """Saves an EMA copy of the model + EMA optimizer states for resume.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: "pl.Trainer"):
        """Saves the original optimizer states for resume.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        """
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """Restore EMA weights when loading a checkpoint.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning Trainer.
        pl_module : pl.LightningModule
            PyTorch Lightning Module.
        checkpoint : Dict[str, Any]
            Checkpoint dictionary.
        """
        checkpoint_callback = trainer.checkpoint_callback

        # Replace connector._ckpt_path with below to avoid calling into lightning's protected API
        ckpt_path = trainer.ckpt_path

        if ckpt_path and checkpoint_callback is not None and "atommic" in type(checkpoint_callback).__name__:
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f"-EMA{ext}"):
                rank_zero_info(
                    "loading EMA based weights. The callback will treat the loaded EMA weights "
                    "as the main weights and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))

                checkpoint["optimizer_states"] = ema_state_dict["optimizer_states"]
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    "Unable to find the associated EMA weights when re-loading, training "
                    f"will start with new EMA weights. Expected them to be at: {ema_path}",
                )


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    """Update EMA parameters.

    Parameters
    ----------
    ema_model_tuple : tuple
        EMA model parameters.
    current_model_tuple : tuple
        Current model parameters.
    decay : float
        Decay factor.
    """
    torch._foreach_mul_(ema_model_tuple, decay)  # pylint: disable=protected-access
    torch._foreach_add_(ema_model_tuple, current_model_tuple, alpha=(1.0 - decay))  # pylint: disable=protected-access


def run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    """Run EMA update on CPU.

    Parameters
    ----------
    ema_model_tuple : tuple
        EMA model parameters.
    current_model_tuple : tuple
        Current model parameters.
    decay : float
        Decay factor.
    pre_sync_stream : torch.cuda.Stream, optional
        CUDA stream. Default is ``None``.
    """
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()
    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""EMAOptimizer is a wrapper for torch.optim.Optimizer that computes Exponential Moving Average of parameters
    registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer with the following formula:
    $$ ema\_weight = ema\_weight + (1 - decay) * (training\_weight - ema\_weight) $$

    To access EMA parameters, use ``swap_ema_weights()`` context manager to perform a temporary in-place swap of
    regular parameters with EMA parameters.

    .. note::
        EMAOptimizer is not compatible with APEX AMP O2.
        Extends :class:`torch.optim.Optimizer`.

    Returns
    -------
    torch.optim.Optimizer
        EMAOptimizer instance.

    Examples
    --------
    >>> model = Model().to(device)
    >>> opt = EMAOptimizer(opt, device, 0.9999)
    >>> for epoch in range(n_epochs):
    >>>     training_loop(model, opt)
    >>>     regular_eval_accuracy = evaluate(model)
    >>>     with opt.swap_ema_weights():
    >>>         ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Inits :class:`EMAOptimizer`.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to wrap.
        device : torch.device
            Device for EMA parameters.
        decay : float
            Decay factor. Default is ``0.9999``.
        every_n_steps : int
            Apply EMA every N steps. Default is ``1``.
        current_step : int
            Current step. Default is ``0``.
        stream : torch.cuda.Stream, optional
            CUDA stream. Default is ``None``.
        """
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = stream
        self.thread = None

        self.ema_params: tuple = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        """Returns an iterator over all parameters."""
        return (param for group in self.param_groups for param in group["params"])

    def step(self, closure=None, **kwargs):  # pylint: disable=unused-argument
        """Performs a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss, by default None.
        **kwargs
            Additional parameters.

        Returns
        -------
        float
            Loss.
        """
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params = self.ema_params + tuple(
                copy.deepcopy(param.data.detach()).to(self.device) for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step = self.current_step + 1
        return loss

    def _should_update_at_step(self) -> bool:
        """Checks if EMA parameters should be updated at current step."""
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        """Updates EMA parameters"""
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True) for param in self.all_parameters()
            )

            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    @staticmethod
    def swap_tensors(tensor1, tensor2):
        """Swaps the values of two tensors in-place."""
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        """Switches the main parameter weights with the EMA weights."""
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        """A context manager to in-place swap regular parameters with EMA parameters. It swaps back to the original
        regular parameters on context manager exit.

        Parameters
        ----------
        enabled : bool, optional
            whether the swap should be performed, by default True
        """
        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        """Wait for the EMA update to finish."""
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        """Return the optimizer state."""
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # if we are in the context of saving an EMA model, the EMA weights are in the modules' actual weights
        ema_params = self.ema_params if not self.in_saving_ema_model_context else list(self.all_parameters())
        state_dict = {
            "opt": self.optimizer.state_dict(),
            "ema": ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the optimizer state."""
        self.join()
        self.optimizer.load_state_dict(state_dict["opt"])
        self.ema_params = tuple(param.to(self.device) for param in copy.deepcopy(state_dict["ema"]))
        self.current_step = state_dict["current_step"]
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        """Add a param group to the :class:`Optimizer` s `param_groups`."""
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True
