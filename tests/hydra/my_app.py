# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/tests/hydra/my_app.py

from dataclasses import dataclass

from omegaconf import MISSING, OmegaConf

from atommic.core.conf.hydra_runner import hydra_runner


@dataclass
class DefaultConfig:
    """This is structured config for this application. It provides the schema used for validation of user-written \
    spec file as well as default values of the selected parameters."""

    dataset_name: str = MISSING


@hydra_runner(config_name="DefaultConfig", schema=DefaultConfig)
def my_app(cfg):
    """
    This is the main application entry point. It is decorated with hydra_runner which takes care of parsing the
    command line arguments, instantiating the config object and running the application.
    """
    print(OmegaConf.to_yaml(cfg))
    # Get dataset_name.
    dataset_name = cfg.dataset_name


if __name__ == "__main__":
    my_app()
