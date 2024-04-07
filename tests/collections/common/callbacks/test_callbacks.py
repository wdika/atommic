# coding=utf-8
# __author__ = "Dimitris Karkalousos"

import time
from unittest.mock import Mock
from atommic.collections.common.callbacks import LogEpochTimeCallback


class TestLogEpochTimeCallback:
    # Tests that the LogEpochTimeCallback initializes without any errors
    def test_log_epoch_time_callback_initialize(self):
        callback = LogEpochTimeCallback()
        assert callback is not None

    # Tests that the on_train_epoch_start method of LogEpochTimeCallback is called without any errors
    def test_log_epoch_time_callback_on_train_epoch_start(self):
        callback = LogEpochTimeCallback()
        trainer = None
        pl_module = None
        callback.on_train_epoch_start(trainer, pl_module)
        assert True

    # Tests that the on_validation_epoch_start method of LogEpochTimeCallback is called without any errors
    def test_log_epoch_time_callback_on_validation_epoch_start(self):
        callback = LogEpochTimeCallback()
        trainer = None
        pl_module = None
        callback.on_validation_epoch_start(trainer, pl_module)
        assert True

    # Tests that the on_validation_epoch_end method of LogEpochTimeCallback is called without any errors
    def test_log_epoch_time_callback_on_validation_epoch_end(self):
        callback = LogEpochTimeCallback()
        trainer = None
        pl_module = None
        callback.on_validation_epoch_end(trainer, pl_module)
        assert True

    # Tests that the on_test_epoch_start method of LogEpochTimeCallback is called without any errors
    def test_log_epoch_time_callback_on_test_epoch_start(self):
        callback = LogEpochTimeCallback()
        trainer = None
        pl_module = None
        callback.on_test_epoch_start(trainer, pl_module)
        assert True
