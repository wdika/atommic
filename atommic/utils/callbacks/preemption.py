# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/callbacks/preemption.py

import signal
import sys
import warnings

import torch
from pytorch_lightning.callbacks import Callback


class PreemptionCallback(Callback):
    """PreemptionCallback class creates a callback that checks for preemption during training at the end of every step.
    Upon preemption the callback provides a function to gracefully exit the training immediately and also saves the
    current state in a checkpoint as *last.ckpt. (to be able to start from the same step without wasting any compute
    while resuming the next time).

    PreemptionCallback is always enabled by default via the arg create_preemption_callback under ExpManagerConfig.
    To disable please pass create_preemption_callback: False in your config file.
    """

    def __init__(self, checkpoint_callback, sig=None):
        """Inits :class:`PreemptionCallback`.

        Parameters
        ----------
        checkpoint_callback : pytorch_lightning.callbacks.ModelCheckpoint
            The checkpoint callback
        sig : int, optional
            The signal to be used for preemption, by default None
        """
        self.sig = sig
        if self.sig is None:
            self.sig = signal.SIGTERM
        self.checkpoint_callback = checkpoint_callback
        self.preemption_enabled = False

    @property
    def interrupted(self):
        """Checks if the job was preempted by broadcasting the preemption signal to all ranks."""
        interrupted = torch.tensor(self._interrupted, device=torch.cuda.current_device(), dtype=torch.int32)
        torch.distributed.broadcast(interrupted, 0)
        interrupted = bool(interrupted.item())
        return interrupted

    def on_train_start(self, trainer, pl_module):
        """
        Defines custom handlers at the beginning of training to be executed when the preemption signal is received.
        """
        # Check if torch distributed is initialised, as It's needed for broadcasting the preemption signal to all ranks
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            warnings.warn("Preemption requires torch distributed to be initialized, disabling preemption callback")
        else:
            self.preemption_enabled = True
            # Bool var that's initialized to false and made True upon receving the preemption signal
            self._interrupted = False
            self.released = False
            self.original_handler = signal.getsignal(self.sig)

            # Master handler executed only by rank 0 when the preemption siganal is received,
            # to avoid deadlock conditions
            def master_handler(signum, frame):  # pylint: disable=unused-argument
                """Handler executed by rank 0 when the preemption signal is received."""
                self.release()
                self._interrupted = True

            # Handler executed by the non zero ranks
            def ignoring_handler(signum, frame):  # pylint: disable=unused-argument
                """Handler executed by non zero ranks when the preemption signal is received."""
                self.release()

            self.private_rank = torch.distributed.get_rank()
            if self.private_rank == 0:
                signal.signal(self.sig, master_handler)
            else:
                signal.signal(self.sig, ignoring_handler)

        return self

    def on_train_end(self, trainer, pl_module):
        """Defines custom handlers at the end of training to be executed when the preemption signal is received.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer object
        pl_module : pytorch_lightning.LightningModule
            The lightning module
        """
        if self.preemption_enabled:
            self.release()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        """Defines custom handlers at the end of every training step to be executed when the preemption signal is
        received.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer object
        pl_module : pytorch_lightning.LightningModule
            The lightning module
        outputs : list
            The outputs of the training step
        batch : list
            The batch of data
        batch_idx : int
            The index of the batch
        """
        if self.preemption_enabled:
            # check if the job was preempted at the end of every training step/iteration
            # NOTE: "self.interrupted" is a property which triggers a distributed broadcast of "_interrupted" flag from
            # rank 0 to all other ranks, to avoid performance overheads it's best to store the result in a regular
            # local variable
            interrupted = self.interrupted
            if interrupted:
                warnings.warn("Received SIGTERM, saving checkpoint and exiting")
                monitor_candidates = self.checkpoint_callback._monitor_candidates(  # pylint: disable=protected-access
                    trainer
                )
                self.checkpoint_callback._save_last_checkpoint(  # pylint: disable=protected-access
                    trainer, monitor_candidates
                )
                sys.exit(0)

    def release(self):
        """Releases the preemption callback."""
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
