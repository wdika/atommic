# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/exp_manager.py

import glob
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from shutil import copy, move
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loops import _TrainingEpochLoop
from pytorch_lightning.strategies.ddp import DDPStrategy

import atommic.utils
from atommic.collections.common.callbacks import EMA
from atommic.constants import ATOMMIC_ENV_VARNAME_TESTING, ATOMMIC_ENV_VARNAME_VERSION
from atommic.utils import logging, timers
from atommic.utils.app_state import AppState
from atommic.utils.callbacks import ATOMMICModelCheckpoint, PreemptionCallback
from atommic.utils.env_var_parsing import get_envbool
from atommic.utils.exceptions import ATOMMICBaseException
from atommic.utils.get_rank import is_global_rank_zero
from atommic.utils.lightning_logger_patch import add_filehandlers_to_pl_logger


class NotFoundError(ATOMMICBaseException):
    """Raised when a file or folder is not found"""


class LoggerMisconfigurationError(ATOMMICBaseException):
    """Raised when a mismatch between trainer.logger and exp_manager occurs"""

    def __init__(self, message):
        """Inits :class:`LoggerMisconfigurationError`.

        Parameters
        ----------
        message : str
            The message to display.
        """
        message = (
            message + "You can disable lightning's trainer from creating a logger by passing logger=False to its "
            "constructor. "
        )
        super().__init__(message)


class CheckpointMisconfigurationError(ATOMMICBaseException):
    """Raised when a mismatch between trainer.callbacks and exp_manager occurs"""


@dataclass
class EarlyStoppingParams:
    """Parameters for the early stopping callback."""

    monitor: str = "val_loss"  # The metric that early stopping should consider.
    mode: str = "min"  # inform early stopping whether to look for increase or decrease in monitor.
    min_delta: float = 0.001  # smallest change to consider as improvement.
    patience: int = 10  # how many (continuous) validation cycles to wait with no improvement and stopping training.
    verbose: bool = True
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: Optional[bool] = None
    log_rank_zero_only: bool = False


@dataclass
class CallbackParams:
    """Parameters for a callback"""

    filepath: Optional[str] = None  # Deprecated
    # If None, exp_manager will attempt to handle the filepath
    dirpath: Optional[str] = None
    # If None, exp_manager will attempt to handle the filepath
    filename: Optional[str] = None
    monitor: Optional[str] = "val_loss"
    verbose: Optional[bool] = True
    save_last: Optional[bool] = True
    save_top_k: Optional[int] = 3
    save_weights_only: Optional[bool] = False
    mode: Optional[str] = "min"
    auto_insert_metric_name: bool = True
    every_n_epochs: Optional[int] = 1
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    # If None, exp_manager will attempt to handle the filepath
    prefix: Optional[str] = None
    postfix: str = ".atommic"
    save_best_model: bool = False
    always_save_atommic: bool = False
    # Automatically save .atommic file during on_train_end hook
    save_atommic_on_train_end: Optional[bool] = True
    # tensor parallel size * pipeline parallel size
    model_parallel_size: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = False  # Save after training, not after validation


@dataclass
class StepTimingParams:
    """Parameters for the step timing callback."""

    reduction: Optional[str] = "mean"
    # if True torch.cuda.synchronize() is called on start/stop
    sync_cuda: Optional[bool] = False
    # if positive, defines the size of a sliding window for computing mean
    buffer_size: Optional[int] = 1


@dataclass
class EMAParams:
    """Parameters for the EMA callback."""

    enable: Optional[bool] = False
    decay: Optional[float] = 0.999
    cpu_offload: Optional[bool] = False
    validate_original_weights: Optional[bool] = False
    every_n_steps: int = 1


@dataclass
class ExpManagerConfig:
    """Configuration for the experiment manager."""

    # Log dir creation parameters
    explicit_log_dir: Optional[str] = None
    exp_dir: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    use_datetime_version: Optional[bool] = True
    resume_if_exists: Optional[bool] = False
    resume_past_end: Optional[bool] = False
    resume_ignore_no_checkpoint: Optional[bool] = False
    resume_from_checkpoint: Optional[str] = None
    # Logging parameters
    create_tensorboard_logger: Optional[bool] = True
    summary_writer_kwargs: Optional[Dict[Any, Any]] = None
    create_wandb_logger: Optional[bool] = False
    wandb_logger_kwargs: Optional[Dict[Any, Any]] = None
    # Checkpointing parameters
    create_checkpoint_callback: Optional[bool] = True
    checkpoint_callback_params: Optional[CallbackParams] = CallbackParams()
    create_early_stopping_callback: Optional[bool] = False
    early_stopping_callback_params: Optional[EarlyStoppingParams] = EarlyStoppingParams()
    create_preemption_callback: Optional[bool] = True
    # Additional exp_manager arguments
    files_to_copy: Optional[List[str]] = None
    # logs timing of train/val/test steps
    log_step_timing: Optional[bool] = True
    step_timing_kwargs: Optional[StepTimingParams] = StepTimingParams()
    # Configures creation of log files for different ranks
    log_local_rank_0_only: Optional[bool] = False
    log_global_rank_0_only: Optional[bool] = False
    # disable initial validation when resuming from a checkpoint saved during validation
    disable_validation_on_resume: Optional[bool] = True
    ema: Optional[EMAParams] = EMAParams()
    # Wall clock time limit
    max_time_per_run: Optional[str] = None
    # time to sleep non 0 ranks during initialization
    seconds_to_sleep: float = 5


class TimingCallback(Callback):
    """Logs execution time of train/val/test steps"""

    def __init__(self, timer_kwargs=None):
        """Inits :class:`TimingCallback`."""
        if timer_kwargs is None:
            timer_kwargs = {}
        self.timer = timers.NamedTimer(**timer_kwargs)

    def _on_batch_start(self, name):
        """Called at the beginning of each batch"""
        # reset only if we do not return mean of a sliding window
        if self.timer.buffer_size <= 0:
            self.timer.reset(name)

        self.timer.start(name)

    def _on_batch_end(self, name, pl_module):
        """Called at the end of each batch"""
        self.timer.stop(name)
        # Set the `batch_size=1` as WAR for `dataloader_iter`, which is not used for any metric
        pl_module.log(
            name + ' in s',
            self.timer[name],
            on_step=True,
            on_epoch=False,
            batch_size=1,
            prog_bar=(name == "train_step_timing"),
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):  # pylint: disable=unused-argument
        """Called at the beginning of each training batch"""
        self._on_batch_start("train_step_timing")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs  # pylint: disable=unused-argument
    ):
        """Logs the time taken by the training batch"""
        self._on_batch_end("train_step_timing", pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Logs the time taken by the validation batch"""
        self._on_batch_start("validation_step_timing")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Logs the time taken by the validation step"""
        self._on_batch_end("validation_step_timing", pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Logs execution time of test steps"""
        self._on_batch_start("test_step_timing")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Logs execution time of test steps"""
        self._on_batch_end("test_step_timing", pl_module)

    def on_before_backward(self, trainer, pl_module, loss):
        """Logs the time taken for backward pass"""
        self._on_batch_start("train_backward_timing")

    def on_after_backward(self, trainer, pl_module):
        """Note: this is called after the optimizer step"""
        self._on_batch_end("train_backward_timing", pl_module)


def exp_manager(trainer: Trainer, cfg: Optional[Union[DictConfig, Dict]] = None) -> Optional[Path]:  # noqa: MC0001
    r"""exp_manager is a helper function used to manage folders for experiments. It follows the pytorch lightning
    paradigm of exp_dir/model_or_experiment_name/version. If the lightning trainer has a logger, exp_manager will get
    exp_dir, name, and version from the logger. Otherwise, it will use the exp_dir and name arguments to create the
    logging directory. exp_manager also allows for explicit folder creation via explicit_log_dir.

    The version can be a datetime string or an integer. Date time version can be disabled if use_datetime_version is
    set to False. It optionally creates TensorBoardLogger, WandBLogger, ModelCheckpoint objects from pytorch lightning.
    It copies sys.argv, and git information if available to the logging directory. It creates a log file for each
    process to log their output into.

    exp_manager additionally has a resume feature (resume_if_exists) which can be used to continuing training from the
    constructed log_dir. When you need to continue the training repeatedly (like on a cluster which you need multiple
    consecutive jobs), you need to avoid creating the version folders. Therefore, from v1.0.0, when resume_if_exists
    is set to True, creating the version folders is ignored.

    Parameters
    ----------
    trainer : pytorch_lightning.Trainer
        The lightning trainer object.
    cfg : DictConfig or Dict, optional
        Can have the following keys:
            - explicit_log_dir : str
                Can be used to override exp_dir/name/version folder creation. Defaults to ``None``, which will use
                    exp_dir, name, and version to construct the logging directory.
            - exp_dir : str
                The base directory to create the logging directory. Defaults to ``None``, which logs to
                    ./atommic_experiments.
            - name : str
                The name of the experiment. Defaults to ``None`` which turns into "default" via name = name or
                    "default".
            - version : str
                The version of the experiment. Defaults to None which uses either a datetime string or lightning's
                    TensorboardLogger system of using version_{int}.
            - use_datetime_version : bool
                Whether to use a datetime string for version. Default is ``True``.
            - resume_if_exists : bool
                Whether this experiment is resuming from a previous run. If True, it sets
                    trainer._checkpoint_connector._ckpt_path so that the trainer should auto-resume. exp_manager will
                    move files under log_dir to log_dir/run_{int}. Default is ``False``. When resume_if_exists is
                    True, we would not create version folders to make it easier to find the log folder for next runs.
            - resume_past_end : bool
                exp_manager errors out if resume_if_exists is True and a checkpoint matching '\'*end.ckpt indicating a
                    previous training run fully completed. This behaviour can be disabled, in which case the
                    '\'*end.ckpt will be loaded by setting resume_past_end to True. Default is ``False``.
            - resume_ignore_no_checkpoint : bool
                exp_manager errors out if resume_if_exists is True and no checkpoint could be found. This behaviour
                    can be disabled, in which case exp_manager will print a message and continue without restoring, by
                    setting resume_ignore_no_checkpoint to True. Default is ``False``.
            - resume_from_checkpoint : str
                Can be used to specify a path to a specific checkpoint file to load from. This will override any
                    checkpoint found when resume_if_exists is True. Default is ``None``.
            - create_tensorboard_logger : bool
                Whether to create a tensorboard logger and attach it to the pytorch lightning trainer.
                    Default is ``True``.
            - summary_writer_kwargs : dict
                A dictionary of kwargs that can be passed to lightning's TensorboardLogger class. Note that log_dir is
                    passed by exp_manager and cannot exist in this dict. Default is ``None``.
            - create_wandb_logger : bool
                Whether to create a Weights and Biases logger and attach it to the pytorch lightning  trainer.
                    Default is ``False``.
            - wandb_logger_kwargs : dict
                A dictionary of kwargs that can be passed to lightning's WandBLogger class. Note that name and project
                    are required parameters if create_wandb_logger is True. Default is ``None``..
            - create_checkpoint_callback : bool
                Whether to create a ModelCheckpoint callback and attach it to the pytorch lightning trainer. The
                    ModelCheckpoint saves the top 3 models with the best "val_loss", the most recent checkpoint under
                    '\'*last.ckpt, and the final checkpoint after training completes under '\'*end.ckpt.
                    Default is ``True``.
            - create_early_stopping_callback : bool
                Whether to create an EarlyStopping callback and attach it to the pytorch lightning trainer. The
                    EarlyStopping callback stops training if the "val_loss" does not improve for 3 epochs.
                    Default is ``True``.
            - files_to_copy : list
                A list of files to copy to the experiment logging directory. Defaults to None which copies no files.
            - log_local_rank_0_only : bool
                Whether to only create log files for local rank 0. Default is ``False``. Set this to True if you are
                    using DDP with many GPUs and do not want many log files in your exp dir.
            - log_global_rank_0_only : bool
                Whether to only create log files for global rank 0. Defaults to False. Set this to True if you are
                    using DDP with many GPUs and do not want many log files in your exp dir.
            - max_time : str
                The maximum wall clock time *per run*. This is intended to be used on clusters where you want a
                    checkpoint to be saved after this specified time and be able to resume from that checkpoint.
                    Default is ``None``.
            - seconds_to_sleep : float
                Seconds to sleep non rank 0 processes for. Used to give enough time for rank 0 to initialize.

    Returns
    -------
    log_dir : Path
        The final logging directory where logging files are saved. Usually the concatenation of exp_dir, name, and
        version.
    """
    # Add rank information to logger
    # Note: trainer.global_rank and trainer.is_global_zero are not set until trainer.fit, so have to hack around it
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = trainer.node_rank * trainer.num_devices + local_rank
    logging.rank = global_rank

    if cfg is None:
        logging.error("exp_manager did not receive a cfg argument. It will be disabled.")
        return None

    if trainer.fast_dev_run:
        logging.info("Trainer was called with fast_dev_run. exp_manager will return without any functionality.")
        return None

    # Ensure passed cfg is compliant with ExpManagerConfig
    schema = OmegaConf.structured(ExpManagerConfig)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.merge(schema, cfg)

    # Ensures that trainer options are compliant with atommic and exp_manager arguments
    error_checks(trainer, cfg)

    log_dir, exp_dir, name, version = get_log_dir(
        trainer=trainer,
        exp_dir=cfg.exp_dir,
        name=cfg.name,
        version=cfg.version,
        explicit_log_dir=cfg.explicit_log_dir,
        use_datetime_version=cfg.use_datetime_version,
        resume_if_exists=cfg.resume_if_exists,
    )

    check_resume(
        trainer,
        log_dir,
        cfg.resume_if_exists,
        cfg.resume_past_end,
        cfg.resume_ignore_no_checkpoint,
        cfg.checkpoint_callback_params.dirpath,
        cfg.resume_from_checkpoint,
    )

    checkpoint_name = name
    # If name returned from get_log_dir is "", use cfg.name for checkpointing
    if checkpoint_name is None or checkpoint_name == "":
        checkpoint_name = cfg.name or "default"

    cfg.name = name  # Used for configure_loggers so that the log_dir is properly set even if name is ""
    cfg.version = version

    # update app_state with log_dir, exp_dir, etc
    app_state = AppState()
    app_state.log_dir = log_dir
    app_state.exp_dir = exp_dir
    app_state.name = name
    app_state.version = version
    app_state.checkpoint_name = checkpoint_name
    app_state.create_checkpoint_callback = cfg.create_checkpoint_callback
    app_state.checkpoint_callback_params = cfg.checkpoint_callback_params

    # Create the logging directory if it does not exist. Cannot limit creation to global zero as all ranks write to own
    # log file.
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Experiments will be logged at {log_dir}")
    trainer._default_root_dir = log_dir  # pylint: disable=protected-access

    if cfg.log_local_rank_0_only is True and cfg.log_global_rank_0_only is True:
        raise ValueError(
            "Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. "
            "Please set either one or neither."
        )

    # This is set if the env var atommic_TESTING is set to True.
    atommic_testing = get_envbool(ATOMMIC_ENV_VARNAME_TESTING, False)

    # Handle logging to file
    log_file = log_dir / f"atommic_log_globalrank-{global_rank}_localrank-{local_rank}.txt"
    if cfg.log_local_rank_0_only is True and not atommic_testing:
        if local_rank == 0:
            logging.add_file_handler(log_file)
    elif cfg.log_global_rank_0_only is True and not atommic_testing:
        if global_rank == 0:
            logging.add_file_handler(log_file)
    else:
        # Logs on all ranks.
        logging.add_file_handler(log_file)

    # For some reason, LearningRateLogger requires trainer to have a logger. Safer to create logger on all ranks
    # not just global rank 0.
    if cfg.create_tensorboard_logger or cfg.create_wandb_logger:
        configure_loggers(
            trainer,
            [Path(exp_dir)],
            [Path(log_dir)],
            cfg.name,
            cfg.version,
            cfg.checkpoint_callback_params,
            cfg.create_tensorboard_logger,
            cfg.summary_writer_kwargs,
            cfg.create_wandb_logger,
            cfg.wandb_logger_kwargs,
        )

    # add loggers timing callbacks
    if cfg.log_step_timing:
        timing_callback = TimingCallback(timer_kwargs=cfg.step_timing_kwargs or {})
        trainer.callbacks.insert(0, timing_callback)

    if cfg.ema.enable:
        ema_callback = EMA(
            decay=cfg.ema.decay,
            validate_original_weights=cfg.ema.validate_original_weights,
            cpu_offload=cfg.ema.cpu_offload,
            every_n_steps=cfg.ema.every_n_steps,
        )
        trainer.callbacks.append(ema_callback)

    if cfg.create_early_stopping_callback:
        early_stop_callback = EarlyStopping(**cfg.early_stopping_callback_params)
        trainer.callbacks.append(early_stop_callback)

    if cfg.create_checkpoint_callback:
        configure_checkpointing(
            trainer,
            log_dir,
            checkpoint_name,
            cfg.resume_if_exists,
            cfg.checkpoint_callback_params,
            cfg.create_preemption_callback,
        )

    if cfg.disable_validation_on_resume:
        # extend training loop to skip initial validation when resuming from checkpoint
        configure_no_restart_validation_training_loop(trainer)

    # Setup a stateless timer for use on clusters.
    if cfg.max_time_per_run is not None:
        found_ptl_timer = False
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, Timer):
                # NOTE: PTL does not expose a `trainer.max_time`. By the time we are in this function, PTL has already
                # set up a timer if the user specifies `trainer.max_time` so best we can do is replace that.
                # Working: If only `trainer.max_time` is set - it behaves as a normal PTL timer.
                # If only `exp_manager.max_time_per_run` is set - it behaves as a StateLessTimer.
                # If both are set, it also behaves as a StateLessTimer.
                logging.warning(
                    "Found a PTL Timer callback, replacing with a StatelessTimer callback. "
                    "This will happen if you set trainer.max_time as well as exp_manager.max_time_per_run."
                )
                trainer.callbacks[idx] = StatelessTimer(cfg.max_time_per_run)
                found_ptl_timer = True
                break

        if not found_ptl_timer:
            trainer.max_time = cfg.max_time_per_run
            trainer.callbacks.append(StatelessTimer(cfg.max_time_per_run))

    if is_global_rank_zero():
        # Move files_to_copy to folder and add git information if present
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                copy(Path(_file), log_dir)

        # Create files for cmd args and git info
        with open(log_dir / "cmd-args.log", "w", encoding="utf-8") as _file:
            _file.write(" ".join(sys.argv))

        # Try to get git hash
        git_repo, git_hash = get_git_hash()
        if git_repo:
            with open(log_dir / "git-info.log", "w", encoding="utf-8") as _file:
                _file.write(f"commit hash: {git_hash}")
                _file.write(get_git_diff())

        # Add err_file logging to global_rank zero
        logging.add_err_file_handler(log_dir / "atommic_error_log.txt")

        # Add lightning file logging to global_rank zero
        add_filehandlers_to_pl_logger(log_dir / "lightning_logs.txt", log_dir / "atommic_error_log.txt")

    elif trainer.num_devices * trainer.num_devices > 1:
        # sleep other ranks so rank 0 can finish
        # doing the initialization such as moving files
        time.sleep(cfg.seconds_to_sleep)

    return log_dir


def error_checks(trainer: Trainer, cfg: Optional[Union[DictConfig, Dict]] = None):
    """Checks that the passed trainer is compliant with atommic and exp_manager's passed configuration. Checks that:
    - Throws error when hydra has changed the working directory. This causes issues with lightning's DDP
    - Throws error when trainer has loggers defined but create_tensorboard_logger or create_wandB_logger is True
    - Prints error messages when 1) run on multi-node and not Slurm, and 2) run on multi-gpu without DDP
    """
    if HydraConfig.initialized() and get_original_cwd() != os.getcwd():
        raise ValueError(
            "Hydra changed the working directory. This interferes with ExpManger's functionality. Please pass "
            "hydra.run.dir=. to your python script."
        )

    if trainer.logger is not None and (cfg.create_tensorboard_logger or cfg.create_wandb_logger):  # type: ignore
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger, and either "
            f"create_tensorboard_logger: {cfg.create_tensorboard_logger} or create_wandb_logger: "  # type: ignore
            f"was set to True. These can only be used if trainer does not already have a logger."
        )

    if trainer.num_nodes > 1 and not check_slurm(trainer):
        logging.error(
            "You are running multi-node training without SLURM handling the processes."
            " Please note that this is not tested in atommic and could result in errors."
        )

    if trainer.num_devices > 1 and not isinstance(trainer.strategy, DDPStrategy):
        logging.error(
            "You are running multi-gpu without ddp.Please note that this is not tested in atommic and could result in "
            "errors."
        )


def check_resume(  # noqa: MC0001
    trainer: Trainer,
    log_dir: Union[str, Path],
    resume_if_exists: bool = False,
    resume_past_end: bool = False,
    resume_ignore_no_checkpoint: bool = False,
    dirpath: str = None,
    resume_from_checkpoint: str = None,
):
    """Checks that resume=True was used correctly with the arguments pass to exp_manager. Sets
    trainer._checkpoint_connector._ckpt_path as necessary.

    Parameters
    ----------
    trainer : pytorch_lightning.Trainer
        The trainer that is being used.
    log_dir : Union[str, Path]
        The directory where the logs are being saved.
    resume_if_exists : bool
        Whether to resume if the experiment directory already exists.
    resume_past_end : bool
        Whether to resume from the end of the experiment.
    resume_ignore_no_checkpoint : bool
        Whether to ignore if there is no checkpoint to resume from.
    dirpath : str
        The directory to resume from. If None, will resume from the latest checkpoint.
    resume_from_checkpoint : str
        The checkpoint to resume from. If None, will resume from the latest checkpoint.

    Returns
    -------
    NotFoundError : bool
        If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
    ValueError : bool
        If resume is True, and there were more than 1 checkpoint could be found.
    """
    if not log_dir:
        raise ValueError(f"Resuming requires the log_dir {log_dir} to be passed to exp_manager")

    checkpoint = None
    if resume_from_checkpoint:
        checkpoint = resume_from_checkpoint
    if resume_if_exists:
        # Use <log_dir>/checkpoints/ unless `dirpath` is set
        checkpoint_dir = Path(dirpath) if dirpath else Path(Path(log_dir) / "checkpoints")

        # when using distributed checkpointing, checkpoint_dir is a directory of directories
        # we check for this here
        dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
        end_dist_checkpoints = [d for d in dist_checkpoints if d.match("*end")]
        last_dist_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

        end_checkpoints = end_dist_checkpoints if end_dist_checkpoints else list(checkpoint_dir.rglob("*end.ckpt"))
        last_checkpoints = last_dist_checkpoints if last_dist_checkpoints else list(checkpoint_dir.rglob("*last.ckpt"))

        if not checkpoint_dir.exists() or (not len(end_checkpoints) > 0 and not len(last_checkpoints) > 0):
            if resume_ignore_no_checkpoint:
                warn = (
                    "There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir "
                    f":{checkpoint_dir}. "
                )
                if checkpoint is None:
                    warn += "Training from scratch."
                elif checkpoint == resume_from_checkpoint:
                    warn += f"Training from {resume_from_checkpoint}."
                logging.warning(warn)
            else:
                raise NotFoundError(
                    "There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir "
                    f":{checkpoint_dir}. Cannot resume."
                )
        elif len(end_checkpoints) > 0:
            if resume_past_end:
                if len(end_checkpoints) > 1:
                    if 'mp_rank' in str(end_checkpoints[0]):
                        checkpoint = end_checkpoints[0]  # type: ignore
                    else:
                        raise ValueError(f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
            else:
                raise ValueError(
                    f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
                )
        elif len(last_checkpoints) > 1:
            if 'mp_rank' in str(last_checkpoints[0]) or 'tp_rank' in str(last_checkpoints[0]):
                checkpoint = last_checkpoints[0]  # type: ignore
                checkpoint = atommic.utils.model_utils.uninject_model_parallel_rank(checkpoint)
            else:
                raise ValueError(f"Multiple checkpoints {last_checkpoints} that matches *last.ckpt.")
        else:
            checkpoint = last_checkpoints[0]  # type: ignore

    # PTL 2.0 supports ckpt_path instead of resume_from_checkpoint as the trainer flag
    if checkpoint is not None:
        trainer.ckpt_path = str(checkpoint)
        logging.info(f'Resuming training from checkpoint: {trainer.ckpt_path}')

    if is_global_rank_zero():
        # Check to see if any files exist that need to be moved
        files_to_move = []
        if Path(log_dir).exists():
            for child in Path(log_dir).iterdir():
                if child.is_file():
                    files_to_move.append(child)

        if len(files_to_move) > 0:
            # Move old files to a new folder
            other_run_dirs = Path(log_dir).glob("run_*")
            run_count = 0
            for fold in other_run_dirs:
                if fold.is_dir():
                    run_count += 1
            new_run_dir = Path(Path(log_dir) / f"run_{run_count}")
            new_run_dir.mkdir()
            for _file in files_to_move:
                move(str(_file), str(new_run_dir))


def check_explicit_log_dir(
    trainer: Trainer,
    explicit_log_dir: List[Union[Path, str]],
    exp_dir: str,
    name: str,  # pylint: disable=unused-argument
    version: str,
) -> Tuple[Path, str, str, str]:
    """Checks that the passed arguments are compatible with explicit_log_dir.

    Parameters
    ----------
    trainer : pytorch_lightning.Trainer
        The trainer to check.
    explicit_log_dir : str
        The explicit log dir to check.
    exp_dir : str
        The experiment directory to check.
    name : str
        The experiment name to check.
    version : str
        The experiment version to check.

    Returns
    -------
    tuple
        The log_dir, exp_dir, name, and version that should be used.

    Raises
    ------
    LoggerMisconfigurationError
        If the trainer already has a logger.
    """
    if trainer.logger is not None:
        raise LoggerMisconfigurationError(
            "The pytorch lightning trainer that was passed to exp_manager contained a logger and explicit_log_dir: "
            f"{explicit_log_dir} was pass to exp_manager. Please remove the logger from the lightning trainer."
        )
    # Checking only (explicit_log_dir) vs (exp_dir and version).
    # The `name` will be used as the actual name of checkpoint/archive.
    if exp_dir or version:
        logging.error(
            f"exp_manager received explicit_log_dir: {explicit_log_dir} and at least one of exp_dir: {exp_dir}, "
            f"or version: {version}. Please note that exp_dir, name, and version will be ignored."
        )
    if is_global_rank_zero() and Path(str(explicit_log_dir)).exists():
        logging.warning(f"Exp_manager is logging to {explicit_log_dir}, but it already exists.")
    return Path(str(explicit_log_dir)), str(explicit_log_dir), "", ""


def get_log_dir(
    trainer: Trainer,
    exp_dir: str = None,
    name: str = None,
    version: str = None,
    explicit_log_dir: str = None,
    use_datetime_version: bool = True,
    resume_if_exists: bool = False,
) -> Tuple[Path, str, str, str]:
    """Obtains the log_dir used for exp_manager.

    Parameters
    ----------
    trainer : pytorch_lightning.Trainer
        The trainer to check.
    exp_dir : str
        The experiment directory to check.
    name : str
        The experiment name to check.
    version : str
        The experiment version to check.
    explicit_log_dir : str
        The explicit log dir to check.
    use_datetime_version : bool
        Whether to use datetime versioning.
    resume_if_exists : bool
        Whether to resume if the log_dir already exists.

    Raises
    -------
    LoggerMisconfigurationError : bool
        If trainer is incompatible with arguments.
    NotFoundError : bool
        If resume is True, resume_ignore_no_checkpoint is False, and checkpoints could not be found.
    ValueError : bool
        If resume is True, and there were more than 1 checkpoint could be found.
    """
    if explicit_log_dir:  # If explicit log_dir was passed, short circuit
        return check_explicit_log_dir(trainer, [Path(explicit_log_dir)], exp_dir, name, version)  # type: ignore

    # Default exp_dir to ./atommic_experiments if None was passed
    _exp_dir = exp_dir
    if exp_dir is None:
        _exp_dir = str(Path.cwd() / "atommic_experiments")

    # If the user has already defined a logger for the trainer, use the logger defaults for logging directory
    if trainer.logger is not None:
        if trainer.logger.save_dir:
            if exp_dir:
                raise LoggerMisconfigurationError(
                    "The pytorch lightning trainer that was passed to exp_manager contained a logger, the logger's "
                    f"save_dir was not None, and exp_dir ({exp_dir}) was not None. If trainer.logger.save_dir "
                    "exists, exp_manager will use trainer.logger.save_dir as the logging directory and exp_dir "
                    "must be None."
                )
            _exp_dir = trainer.logger.save_dir
        if name:
            raise LoggerMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a logger, and name: "
                f"{name} was also passed to exp_manager. If the trainer contains a "
                "logger, exp_manager will use trainer.logger.name, and name passed to exp_manager must be None."
            )
        name = trainer.logger.name
        version = f"version_{trainer.logger.version}"
    # Use user-defined exp_dir, project_name, exp_name, and versioning options
    else:
        name = name or "default"
        version = version or os.environ.get(ATOMMIC_ENV_VARNAME_VERSION)

        if not version:
            if resume_if_exists:
                logging.warning(
                    "No version folders would be created under the log folder as 'resume_if_exists' is enabled."
                )
                version = None
            elif is_global_rank_zero():
                if use_datetime_version:
                    version = time.strftime("%Y-%m-%d_%H-%M-%S")
                else:
                    tensorboard_logger = TensorBoardLogger(save_dir=_exp_dir, name=name, version=version)
                    version = f"version_{tensorboard_logger.version}"
                os.environ[ATOMMIC_ENV_VARNAME_VERSION] = "" if version is None else version

    log_dir = Path(str(_exp_dir)) / Path(str(name)) / Path("" if version is None else str(version))
    return log_dir, str(_exp_dir), str(name), str(version)


def get_git_hash():
    """Helper function that tries to get the commit hash if running inside a git folder.

    Returns
    -------
    Bool: Whether the git subprocess ran without error.
    String: git subprocess output or error message
    """
    try:
        return True, subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as err:
        return False, f'{err.output.decode("utf-8")}\n'


def get_git_diff():
    """Helper function that tries to get the git diff if running inside a git folder.

    Returns
    -------
    bool
        Whether the git subprocess ran without error.
    str
        If git subprocess output or error message.
    """
    try:
        return subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as err:
        return f'{err.output.decode("utf-8")}\n'


def configure_loggers(
    trainer: Trainer,
    exp_dir: List[Union[Path, str]],
    log_dir: List[Union[Path, str]],  # pylint: disable=unused-argument
    name: str,
    version: str,
    checkpoint_callback_params: dict,  # pylint: disable=unused-argument
    create_tensorboard_logger: bool,
    summary_writer_kwargs: dict,
    create_wandb_logger: bool,
    wandb_kwargs: dict,
):
    """Creates TensorboardLogger and/or WandBLogger and attach them to trainer. Raises ValueError if
    summary_writer_kwargs or wandb_kwargs are miss configured.

    Parameters
    ----------
    trainer : pytorch_lightning.Trainer
        The trainer to attach the loggers to.
    exp_dir : str
        The experiment directory.
    log_dir : str
        The logging directory.
    name : str
        The name of the experiment.
    version : str
        The version of the experiment.
    checkpoint_callback_params : dict
        The checkpoint callback parameters.
    create_tensorboard_logger : bool
        Whether to create a TensorboardLogger.
    summary_writer_kwargs : dict
        The kwargs to pass to the TensorboardLogger.
    create_wandb_logger : bool
        Whether to create a Weights & Biases logger.
    wandb_kwargs : dict
        The kwargs to pass to the Weights & Biases logger.

    Returns
    -------
    LoggerList
        A list of loggers.
    """
    # Potentially create tensorboard logger and/or WandBLogger
    logger_list = []
    if create_tensorboard_logger:
        if summary_writer_kwargs is None:
            summary_writer_kwargs = {}
        elif "log_dir" in summary_writer_kwargs:
            raise ValueError(
                "You cannot pass `log_dir` as part of `summary_writer_kwargs`. `log_dir` is handled by lightning's "
                "TensorBoardLogger logger."
            )
        tensorboard_logger = TensorBoardLogger(
            save_dir=exp_dir[0], name=name, version=version, **summary_writer_kwargs
        )
        logger_list.append(tensorboard_logger)
        logging.info("TensorboardLogger has been set up")

    if create_wandb_logger:
        if wandb_kwargs is None:
            wandb_kwargs = {}
        if "name" not in wandb_kwargs and "project" not in wandb_kwargs:
            raise ValueError("name and project are required for wandb_logger")
        wandb_logger = WandbLogger(save_dir=str(exp_dir[0]), version=version, **wandb_kwargs)

        logger_list.append(wandb_logger)
        logging.info("WandBLogger has been set up")

    trainer._logger_connector.configure_logger(logger_list)  # pylint: disable=protected-access


def configure_checkpointing(  # noqa: MC0001
    trainer: Trainer,
    log_dir: Path,
    name: str,
    resume: bool,
    params: "DictConfig",
    create_preemption_callback: bool,
):
    """Adds ModelCheckpoint to trainer. Raises CheckpointMisconfigurationError if trainer already has a ModelCheckpoint
    callback or if trainer.weights_save_path was passed to Trainer.
    """
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            raise CheckpointMisconfigurationError(
                "The pytorch lightning trainer that was passed to exp_manager contained a ModelCheckpoint "
                "and create_checkpoint_callback was set to True. Please either set create_checkpoint_callback "
                "to False, or remove ModelCheckpoint from the lightning trainer"
            )

    # Create the callback and attach it to trainer
    if "filepath" in params:
        if params.filepath is not None:
            logging.warning("filepath is deprecated. Please switch to dirpath and filename instead")
            if params.dirpath is None:
                params.dirpath = Path(params.filepath).parent
            if params.filename is None:
                params.filename = Path(params.filepath).name
        with open_dict(params):
            del params["filepath"]
    if params.dirpath is None:
        params.dirpath = Path(log_dir / "checkpoints")
    if params.filename is None:
        params.filename = f"{name}--{{{params.monitor}:.4f}}-{{epoch}}"
    if params.prefix is None:
        params.prefix = name
    ATOMMICModelCheckpoint.CHECKPOINT_NAME_LAST = f"{params.filename}-last"

    logging.debug(params.dirpath)
    logging.debug(params.filename)
    logging.debug(params.prefix)

    if "val" in params.monitor:
        if (
            trainer.max_epochs is not None
            and trainer.max_epochs != -1
            and trainer.max_epochs < trainer.check_val_every_n_epoch
        ):
            logging.error(
                "The checkpoint callback was told to monitor a validation value but trainer.max_epochs("
                f"{trainer.max_epochs}) was less than trainer.check_val_every_n_epoch("
                f"{trainer.check_val_every_n_epoch}). It is very likely this run will fail with "
                f"ModelCheckpoint(monitor='{params.monitor}') not found in the returned metrics. Please ensure that "
                "validation is run within trainer.max_epochs."
            )
        elif trainer.max_steps is not None and trainer.max_steps != -1:
            logging.warning(
                "The checkpoint callback was told to monitor a validation value and trainer's max_steps was set to "
                f"{trainer.max_steps}. Please ensure that max_steps will run for at least "
                f"{trainer.check_val_every_n_epoch} epochs to ensure that checkpointing will not error out."
            )

    checkpoint_callback = ATOMMICModelCheckpoint(n_resume=resume, **params)
    checkpoint_callback.last_model_path = trainer.ckpt_path or ""
    if "mp_rank" in checkpoint_callback.last_model_path or "tp_rank" in checkpoint_callback.last_model_path:
        checkpoint_callback.last_model_path = atommic.utils.model_utils.uninject_model_parallel_rank(
            checkpoint_callback.last_model_path
        )
    trainer.callbacks.append(checkpoint_callback)
    if create_preemption_callback:
        # Check if cuda is available as preemption is supported only on GPUs
        if torch.cuda.is_available():
            # By default, PreemptionCallback handles SIGTERM.
            # To handle other signals pass the signal in the call as below:
            # PreemptionCallback(checkpoint_callback, signal.SIGCHLD)
            preemption_callback = PreemptionCallback(checkpoint_callback)
            trainer.callbacks.append(preemption_callback)
        else:
            logging.info("Preemption is supported only on GPUs, disabling preemption")


def check_slurm(trainer):
    """Checks if the trainer is running on a slurm cluster. If so, it will check if the trainer is running on the
    master node. If it is not, it will exit.

    Parameters
    ----------
    trainer : pytorch_lightning.Trainer
        The trainer to check.

    Returns
    -------
    bool
        True if the trainer is running on the master node, False otherwise.
    """
    try:
        return trainer.accelerator_connector.is_slurm_managing_tasks
    except AttributeError:
        return False


class StatelessTimer(Timer):
    """Extension of PTL timers to be per run."""

    # pylint: disable=arguments-differ
    @staticmethod
    def state_dict() -> Dict[str, Any]:
        """Saves the state of the timer."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the timer."""


def configure_no_restart_validation_training_loop(trainer: pytorch_lightning.Trainer) -> None:
    """Configure the training loop to skip validation when resuming from a checkpoint."""
    if not isinstance(trainer.fit_loop.epoch_loop, _TrainingEpochLoop):
        warnings.warn("Detected custom epoch loop. Skipping no validation on restart support.", UserWarning)
        return
    # Pass trainer object to avoid trainer getting overwritten as None
    loop = SkipResumeTrainingValidationLoop(trainer, trainer.min_steps, trainer.max_steps)
    trainer.fit_loop.epoch_loop = loop


class SkipResumeTrainingValidationLoop(_TrainingEpochLoop):
    """Extend the PTL Epoch loop to skip validating when resuming. This happens when resuming a checkpoint that has
    already run validation, but loading restores the training state before validation has run.
    """

    def _should_check_val_fx(self) -> bool:
        """Skip validation if we are resuming from a checkpoint and the global step is a multiple of the validation."""
        if self.restarting and self.global_step % self.trainer.val_check_batch == 0:
            return False
        return super()._should_check_val_fx()


def clean_exp_ckpt(exp_log_dir: Union[str, Path], remove_ckpt: bool = True, remove_atommic: bool = False):
    """Helper method that removes Pytorch Lightning .ckpt files or atommic .atommic files from the checkpoint
    directory.

    Parameters
    ----------
    exp_log_dir : str or Path
        Path to the root directory of the current experiment.
    remove_ckpt : bool, optional
        Whether to remove all *.ckpt files in the checkpoints directory. Default is True.
    remove_atommic : bool, optional
        Whether to remove all *.atommic files in the checkpoints directory. Default is False.
    """
    exp_log_dir = str(exp_log_dir)

    if remove_ckpt:
        logging.info("Deleting *.ckpt files ...")
        ckpt_files = glob.glob(os.path.join(exp_log_dir, "checkpoints", "*.ckpt"))
        for filepath in ckpt_files:
            os.remove(filepath)
            logging.info(f"Deleted file : {filepath}")

    if remove_atommic:
        logging.info("Deleting *.atommic files ...")
        atommic_files = glob.glob(os.path.join(exp_log_dir, "checkpoints", "*.atommic"))
        for filepath in atommic_files:
            os.remove(filepath)
            logging.info(f"Deleted file : {filepath}")
