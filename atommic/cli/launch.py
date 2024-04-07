# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from atommic.collections.multitask.rs.nn.idslr import IDSLR
from atommic.collections.multitask.rs.nn.idslr_unet import IDSLRUNet
from atommic.collections.multitask.rs.nn.mtlrs import MTLRS
from atommic.collections.multitask.rs.nn.recseg_unet import RecSegUNet
from atommic.collections.multitask.rs.nn.segnet import SegNet
from atommic.collections.multitask.rs.nn.seranet import SERANet
from atommic.collections.quantitative.nn.qcirim import qCIRIM
from atommic.collections.quantitative.nn.qvarnet import qVarNet
from atommic.collections.quantitative.nn.qzf import qZF
from atommic.collections.reconstruction.nn.ccnn import CascadeNet
from atommic.collections.reconstruction.nn.cirim import CIRIM
from atommic.collections.reconstruction.nn.crnn import CRNNet
from atommic.collections.reconstruction.nn.dunet import DUNet
from atommic.collections.reconstruction.nn.jointicnet import JointICNet
from atommic.collections.reconstruction.nn.kikinet import KIKINet
from atommic.collections.reconstruction.nn.lpd import LPDNet
from atommic.collections.reconstruction.nn.modl import MoDL
from atommic.collections.reconstruction.nn.multidomainnet import MultiDomainNet
from atommic.collections.reconstruction.nn.proximal_gradient import ProximalGradient
from atommic.collections.reconstruction.nn.recurrentvarnet import RecurrentVarNet
from atommic.collections.reconstruction.nn.unet import UNet
from atommic.collections.reconstruction.nn.varnet import VarNet
from atommic.collections.reconstruction.nn.vsnet import VSNet
from atommic.collections.reconstruction.nn.xpdnet import XPDNet
from atommic.collections.reconstruction.nn.zf import ZF
from atommic.collections.segmentation.nn.attentionunet import SegmentationAttentionUNet
from atommic.collections.segmentation.nn.dynunet import SegmentationDYNUNet
from atommic.collections.segmentation.nn.lambdaunet import SegmentationLambdaUNet
from atommic.collections.segmentation.nn.unet import SegmentationUNet
from atommic.collections.segmentation.nn.unet3d import Segmentation3DUNet
from atommic.collections.segmentation.nn.unetr import SegmentationUNetR
from atommic.collections.segmentation.nn.vnet import SegmentationVNet
from atommic.core.conf.hydra_runner import hydra_runner
from atommic.utils import logging
from atommic.utils.exp_manager import exp_manager


def register_cli_subcommand(parser: argparse._SubParsersAction):
    """Register parser for the launch command."""
    parser_launch = parser.add_parser(
        "run",
        help="Launch atommic through cli given a configuration (yaml) file, e.g. atommic run -c /path/to/config.yaml",
    )
    parser_launch.add_argument(
        "-c",
        "--config-path",
        required=True,
        type=str,
        help="Path to the configuration file.",
    )
    parser_launch.add_argument(
        "-m",
        "--multi-run",
        action="store_true",
        help="Hydra Multi-Run for hyperparameter optimization.",
    )
    parser_launch.set_defaults(func=main)


@hydra_runner(config_path="../src", config_name="config")
def main(cfg: DictConfig):  # noqa: MC0001
    """
    Main function for training and running a model

    Parameters
    ----------
    cfg : Configuration (yaml) file.
        DictConfig
    """
    cfg = OmegaConf.load(f"{cfg.config_path}")

    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model_name = (cfg.model["model_name"]).upper()

    if model_name == "CASCADENET":
        model = CascadeNet(cfg.model, trainer=trainer)
    elif model_name == "CIRIM":
        model = CIRIM(cfg.model, trainer=trainer)
    elif model_name == "CRNNET":
        model = CRNNet(cfg.model, trainer=trainer)
    elif model_name == "DUNET":
        model = DUNet(cfg.model, trainer=trainer)
    elif model_name in ("E2EVN", "VN"):
        model = VarNet(cfg.model, trainer=trainer)
    elif model_name == "IDSLR":
        model = IDSLR(cfg.model, trainer=trainer)
    elif model_name == "IDSLRUNET":
        model = IDSLRUNet(cfg.model, trainer=trainer)
    elif model_name == "JOINTICNET":
        model = JointICNet(cfg.model, trainer=trainer)
    elif model_name == "KIKINET":
        model = KIKINet(cfg.model, trainer=trainer)
    elif model_name == "LPDNET":
        model = LPDNet(cfg.model, trainer=trainer)
    elif model_name == "MODL":
        model = MoDL(cfg.model, trainer=trainer)
    elif model_name == "MTLRS":
        model = MTLRS(cfg.model, trainer=trainer)
    elif model_name == "MULTIDOMAINNET":
        model = MultiDomainNet(cfg.model, trainer=trainer)
    elif model_name == "PROXIMALGRADIENT":
        model = ProximalGradient(cfg.model, trainer=trainer)
    elif model_name == "QCIRIM":
        model = qCIRIM(cfg.model, trainer=trainer)
    elif model_name == "QVN":
        model = qVarNet(cfg.model, trainer=trainer)
    elif model_name == "QZF":
        model = qZF(cfg.model, trainer=trainer)
    elif model_name == "RECSEGNET":
        model = RecSegUNet(cfg.model, trainer=trainer)
    elif model_name == "RVN":
        model = RecurrentVarNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONATTENTIONUNET":
        model = SegmentationAttentionUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONDYNUNET":
        model = SegmentationDYNUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONLAMBDAUNET":
        model = SegmentationLambdaUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONUNET":
        model = SegmentationUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONUNETR":
        model = SegmentationUNetR(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATION3DUNET":
        model = Segmentation3DUNet(cfg.model, trainer=trainer)
    elif model_name == "SEGMENTATIONVNET":
        model = SegmentationVNet(cfg.model, trainer=trainer)
    elif model_name == "SEGNET":
        model = SegNet(cfg.model, trainer=trainer)
    elif model_name == "SERANET":
        model = SERANet(cfg.model, trainer=trainer)
    elif model_name == "UNET":
        model = UNet(cfg.model, trainer=trainer)
    elif model_name == "VSNET":
        model = VSNet(cfg.model, trainer=trainer)
    elif model_name == "XPDNET":
        model = XPDNet(cfg.model, trainer=trainer)
    elif model_name == "ZF":
        model = ZF(cfg.model, trainer=trainer)
    else:
        raise NotImplementedError(f"{model_name} is not implemented in atommic.")

    if cfg.get("pretrained", None):
        checkpoint = cfg.get("checkpoint", None)
        logging.info(f"Loading pretrained model from {checkpoint}")

        # instantiate model
        if checkpoint.endswith(".atommic"):
            if "huggingface" in checkpoint:
                _, state_dict = model.from_pretrained(checkpoint)
            else:
                _, state_dict = model.restore_from(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]

        model.load_state_dict(state_dict)

    if cfg.get("mode", None) == "train":
        logging.info("Validating")
        trainer.validate(model)
        logging.info("Training")
        trainer.fit(model)
    else:
        logging.info("Testing")
        trainer.test(model)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
