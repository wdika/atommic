# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/callbacks/nemo_model_checkpoint.py

import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Union

import pytorch_lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info

from atommic.collections.common.callbacks import EMA
from atommic.utils import logging, model_utils
from atommic.utils.app_state import AppState
from atommic.utils.get_rank import is_global_rank_zero


class ATOMMICModelCheckpoint(ModelCheckpoint):
    """Light wrapper around Lightning's ModelCheckpoint to force a saved checkpoint on train_end.
    Extends Lightning's on_save_checkpoint func to save the .atommic file. Saves the .atommic file based
    on the best checkpoint saved (according to the monitor value).
    Also contains func to save the EMA copy of the model.
    """

    def __init__(
        self,
        always_save_atommic: bool = False,
        save_atommic_on_train_end: bool = True,
        save_best_model: bool = False,
        postfix: str = ".atommic",
        n_resume: bool = False,
        model_parallel_size: int = None,
        **kwargs,
    ):
        """Inits :class:`ATOMMICModelCheckpoint`

        Parameters
        ----------
        always_save_atommic : bool, optional
            Whether to always save the .atommic file. Default is ``False``.
        save_atommic_on_train_end : bool, optional
            Whether to save the .atommic file on train end. Default is ``True``.
        save_best_model : bool, optional
            Whether to save the best model. Default is ``False``.
        postfix : str, optional
            The postfix of the .atommic file. Default is ``.atommic``.
        n_resume : bool, optional
            Whether to resume from a previous run. Default is ``False``.
        model_parallel_size : int, optional
            The model parallel size. Default is ``None``.
        """
        # Parse and store "extended" parameters: save_best model and postfix.
        self.always_save_atommic = always_save_atommic
        self.save_atommic_on_train_end = save_atommic_on_train_end
        self.save_best_model = save_best_model
        if self.save_best_model and not self.save_atommic_on_train_end:
            logging.warning(
                (
                    "Found save_best_model is True and save_atommic_on_train_end is False. "
                    "Set save_atommic_on_train_end to True to automatically save the best model."
                )
            )
        self.postfix = postfix
        self.previous_best_path = ""
        self.model_parallel_size = model_parallel_size

        # `prefix` is deprecated
        if "prefix" in kwargs:
            self.prefix = kwargs.pop("prefix")
        else:
            self.prefix = ""

        # Call the parent class constructor with the remaining kwargs.
        super().__init__(**kwargs)

        if self.save_top_k != -1 and n_resume:
            logging.debug("Checking previous runs")
            self.atommic_topk_check_previous_run()

    def atommic_topk_check_previous_run(self):
        """Checks if there are any previous runs and if so, loads the best model from the previous run."""
        try:
            self.best_k_models  # pylint: disable=pointless-statement
            self.kth_best_model_path  # pylint: disable=pointless-statement
            self.best_model_score  # pylint: disable=pointless-statement
            self.best_model_path  # pylint: disable=pointless-statement
        except AttributeError as e:
            raise AttributeError(
                "Lightning's ModelCheckpoint was updated. ATOMMICModelCheckpoint needs to update."
            ) from e
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""

        checkpoints = list(path for path in self._saved_checkpoint_paths if not self._is_ema_filepath(path))
        for checkpoint in checkpoints:
            if "mp_rank" in str(checkpoint) or "tp_rank" in str(checkpoint):
                checkpoint = model_utils.uninject_model_parallel_rank(checkpoint)
            checkpoint = str(checkpoint)
            # second case is for distributed checkpoints, since they are a directory there's no extension
            if checkpoint[-10:] == '-last.ckpt' or checkpoint[-5:] == '-last':
                continue
            index = checkpoint.find(self.monitor) + len(self.monitor) + 1  # Find monitor in str + 1 for '='
            if index != len(self.monitor):
                match = re.search("[A-z]", checkpoint[index:])
                if match:
                    value = checkpoint[index : index + match.start() - 1]  # -1 due to separator hypen
                    self.best_k_models[checkpoint] = float(value)
        if len(self.best_k_models) < 1:
            return  # No saved checkpoints yet

        _reverse = bool(self.mode == "min")

        best_k_models = sorted(self.best_k_models, key=self.best_k_models.get, reverse=_reverse)

        # This section should be ok as rank zero will delete all excess checkpoints, since all other ranks are
        # instantiated after rank zero. models_to_delete should be 0 for all other ranks.
        if self.model_parallel_size is not None:
            # check for distributed checkpoint
            if checkpoints[0].is_dir():
                models_to_delete = len(best_k_models) - self.save_top_k
            else:
                models_to_delete = len(best_k_models) - self.model_parallel_size * self.save_top_k
        else:
            models_to_delete = len(best_k_models) - self.save_top_k

        models_to_delete = max(0, models_to_delete)
        logging.debug(f'Number of models to delete: {models_to_delete}')

        # If EMA enabled, delete the additional EMA weights
        ema_enabled = self._has_ema_ckpts(self._saved_checkpoint_paths)

        for _ in range(models_to_delete):
            model = best_k_models.pop(-1)
            self.best_k_models.pop(model)
            self._del_model_without_trainer(model)
            if ema_enabled and self._fs.exists(self._ema_format_filepath(model)):
                self._del_model_without_trainer(self._ema_format_filepath(model))
            logging.debug(f"Removed checkpoint: {model}")

        self.kth_best_model_path = best_k_models[-1]
        self.best_model_path = best_k_models[0]
        self.best_model_score = self.best_k_models[self.best_model_path]

    # pylint: disable=inconsistent-return-statements
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Saves the .atommic file based on the best checkpoint saved (according to the monitor value)."""
        output = super().on_save_checkpoint(  # pylint: disable=assignment-from-no-return
            trainer, pl_module, checkpoint
        )
        if not self.always_save_atommic:
            return output
        # Load the best model and then re-save it
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            logging.warning("always_save_atommic will slow down training for model_parallel > 1.")
        # since we are creating tarfile artifacts we need to update .atommic path
        app_state.model_restore_path = os.path.abspath(
            os.path.expanduser(os.path.join(self.dirpath, self.prefix + self.postfix))
        )
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            maybe_injected_best_model_path = model_utils.inject_model_parallel_rank(self.best_model_path)
        else:
            maybe_injected_best_model_path = self.best_model_path

        if self.save_best_model:
            if not os.path.exists(maybe_injected_best_model_path):
                return

            if self.best_model_path == self.previous_best_path:
                return output

            old_state_dict = deepcopy(pl_module.state_dict())
            checkpoint = torch.load(maybe_injected_best_model_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            # get a new instanace of the model
            pl_module.load_state_dict(checkpoint, strict=True)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            pl_module.save_to(save_path=app_state.model_restore_path)
            logging.info(f"New best .atommic model saved to: {app_state.model_restore_path}")
            pl_module.load_state_dict(old_state_dict, strict=True)
        else:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            pl_module.save_to(save_path=app_state.model_restore_path)
            logging.info(f"New .atommic model saved to: {app_state.model_restore_path}")
        return output

    def on_train_end(self, trainer, pl_module):
        """Saves the .atommic file based on the best checkpoint saved (according to the monitor value)."""
        if trainer.fast_dev_run:
            return None

        # check if we need to save a last checkpoint manually as validation isn't always run based on the interval
        if self.save_last and trainer.val_check_interval != 0:
            should_save_last_checkpoint = False
            if isinstance(trainer.val_check_interval, float) and trainer.val_check_interval % trainer.global_step != 0:
                should_save_last_checkpoint = True
            if isinstance(trainer.val_check_interval, int) and trainer.global_step % trainer.val_check_interval != 0:
                should_save_last_checkpoint = True
            if should_save_last_checkpoint:
                monitor_candidates = self._monitor_candidates(trainer)
                super()._save_last_checkpoint(trainer, monitor_candidates)
        # Call parent on_train_end() to save the -last checkpoint
        super().on_train_end(trainer, pl_module)

        # Load the best model and then re-save it
        if self.save_best_model:
            # wait for all processes
            trainer.strategy.barrier("SaveBestCheckpointConnector.resume_end")
            if self.best_model_path == "":
                logging.warning(
                    f"{self} was told to save the best checkpoint at the end of training, but no saved checkpoints "
                    "were found. Saving latest model instead."
                )
            else:
                self.best_model_path = trainer.strategy.broadcast(self.best_model_path)
                trainer._checkpoint_connector.restore(self.best_model_path)  # pylint: disable=protected-access

        if self.save_atommic_on_train_end:
            pl_module.save_to(save_path=os.path.join(self.dirpath, self.prefix + self.postfix))

    def _del_model_without_trainer(self, filepath: str) -> None:
        """Deletes the checkpoint file without instantiating the model."""
        filepath = Path(filepath)  # type: ignore

        # check if filepath is a distributed a checkpoint
        if model_utils.ckpt_to_dir(filepath).is_dir():
            if is_global_rank_zero():
                try:
                    dist_ckpt = model_utils.ckpt_to_dir(filepath)
                    shutil.rmtree(dist_ckpt)
                    logging.info(f"Removed distributed checkpoint: {dist_ckpt}")
                except Exception:
                    logging.info(f"Tried to remove distributed checkpoint: {dist_ckpt} but failed.")

        else:
            app_state = AppState()

            # legacy model parallel checkpoint
            if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                # filepath needs to be updated to include mp_rank
                filepath = model_utils.inject_model_parallel_rank(filepath)

            # each model parallel rank needs to remove its model
            if is_global_rank_zero() or (
                app_state.model_parallel_size is not None and app_state.data_parallel_rank == 0
            ):
                try:
                    self._fs.rm(filepath)
                    logging.info(f"Removed checkpoint: {filepath}")
                except Exception:
                    logging.info(f"Tried to remove checkpoint: {filepath} but failed.")

    def _ema_callback(self, trainer: "pytorch_lightning.Trainer") -> Optional[EMA]:
        """Returns the EMA callback if it exists."""
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _save_checkpoint(self, trainer: "pytorch_lightning.Trainer", filepath: str) -> None:
        """Saves the checkpoint and the EMA copy of the model if EMA is enabled."""
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            with ema_callback.save_original_optimizer_state(trainer):
                super()._save_checkpoint(trainer, filepath)

            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                filepath = self._ema_format_filepath(filepath)
                if self.verbose:
                    rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                super()._save_checkpoint(trainer, filepath)
        else:
            super()._save_checkpoint(trainer, filepath)

    def _remove_checkpoint(self, trainer: "pytorch_lightning.Trainer", filepath: str) -> None:
        """Removes the checkpoint and the EMA copy of the model if EMA is enabled."""
        super()._remove_checkpoint(trainer, filepath)
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # remove EMA copy of the state dict as well.
            filepath = self._ema_format_filepath(filepath)
            super()._remove_checkpoint(trainer, filepath)

    def _ema_format_filepath(self, filepath: str) -> str:
        """Returns the filepath of the EMA copy of the model."""
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    def _has_ema_ckpts(self, checkpoints: Iterable[Path]) -> bool:
        """Returns True if any of the checkpoints are EMA checkpoints."""
        return any(self._is_ema_filepath(checkpoint_path) for checkpoint_path in checkpoints)

    def _is_ema_filepath(self, filepath: Union[Path, str]) -> bool:
        """Returns True if the filepath is an EMA checkpoint."""
        return str(filepath).endswith(f"-EMA{self.FILE_EXTENSION}")

    @property
    def _saved_checkpoint_paths(self) -> Iterable[Path]:
        """Returns the saved checkpoint paths."""
        # distributed checkpoints are directories so we check for them here
        dist_checkpoints = [d for d in list(Path(self.dirpath).glob("*")) if d.is_dir()]
        if dist_checkpoints:
            return dist_checkpoints
        return Path(self.dirpath).rglob("*.ckpt")
