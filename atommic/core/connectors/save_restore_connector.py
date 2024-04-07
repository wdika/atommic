# coding=utf-8
from __future__ import annotations  # necessary for lazy types evaluation

__author__ = "Dimitris Karkalousos"
# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/connectors/save_restore_connector.py

import os
import shutil
import tarfile
import tempfile
import uuid
from typing import Optional, Set, Union

import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

# to avoid circular import do not import ModelPT directly
from atommic.core import classes as atommic_classes
from atommic.utils import logging, model_utils
from atommic.utils.app_state import AppState
from atommic.utils.get_rank import is_global_rank_zero


class SaveRestoreConnector:
    """This class is used to save and restore the model state."""

    def __init__(self) -> None:
        """Inits :class:`SaveRestoreConnector`."""
        self._model_config_yaml = "model_config.yaml"
        self._model_weights_ckpt = "model_weights.ckpt"
        self._model_extracted_dir = None

    def save_to(self, model: "atommic.ModelPT", save_path: str):  # type: ignore  # noqa: F821
        """Saves model instance (weights and configuration) into .atommic file.

        You can use "restore_from" method to fully restore instance from .atommic file.

        .atommic file is an archive (tar.gz) with the following:
        - model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for
        model's constructor
        - model_weights.ckpt - model checkpoint

        Parameters
        ----------
        model : ModelPT
            ModelPT object to be saved.
        save_path : str
            Path to .atommic file where model instance should be saved
        """
        if is_global_rank_zero():
            with tempfile.TemporaryDirectory() as tmpdir:
                config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                model.to_config_file(path2yaml_file=config_yaml)
                # update subconfigs, if there are child model, since child model can change its config
                self._update_subconfigs(model, path2yaml_file=config_yaml)
                if model.has_native_or_submodules_artifacts():
                    self._handle_artifacts(model, atommic_file_folder=tmpdir)
                    # We should not update self._cfg here - the model can still be in use
                    self._update_artifact_paths(model, path2yaml_file=config_yaml)
                self._save_state_dict_to_disk(model.state_dict(), model_weights)
                self._make_atommic_file_from_folder(filename=save_path, source_dir=tmpdir)
        else:
            return

    def load_config_and_state_dict(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,  # pylint: disable=unused-argument
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """Restores model instance (weights and configuration) into .atommic file

        Parameters
        ----------
        calling_cls : class
            Class of the model to be restored.
        restore_path : str
            Path to .atommic file from which model should be instantiated
        override_config_path : Optional[Union[OmegaConf, str]]
            Path to a yaml config that will override the internal config file or an OmegaConf/DictConfig object
            representing the model config.
        map_location : Optional[torch.device]
            Optional torch.device() to map the instantiated model to a device. Default is ``None``, it will select a
            GPU if available, falling back to CPU otherwise.
        strict : bool
            Passed to load_state_dict. When set to False, the model will be able to load a checkpoint that has
            more parameters than the model itself. Default is ``True``.
        return_config : bool
            If set to true, will return just the underlying config of the restored model as an OmegaConf DictConfig
            object without instantiating the model.
        trainer : Trainer
            Optional trainer object to be used for model parallelism.

        Example
        -------
            ```
            model = atommic.collections.asr.models.EncDecCTCModel.restore_from('asr.atommic')
            assert isinstance(model, atommic.collections.asr.models.EncDecCTCModel)
            ```

        Returns
        -------
            An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there (original .atommic behavior)
        cwd = os.getcwd()

        if map_location is None:
            if torch.cuda.is_available():
                map_location = torch.device("cuda")
            else:
                map_location = torch.device("cpu")

        app_state = AppState()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Check if self.model_extracted_dir is set, and is a valid path
                if self.model_extracted_dir is not None and os.path.isdir(self.model_extracted_dir):
                    # Log that atommic will use the provided `model_extracted_dir`
                    logging.info(
                        "Restoration will occur within pre-extracted directory : " f"`{self.model_extracted_dir}`."
                    )
                    # Override `tmpdir` above with the pre-extracted `model_extracted_dir`
                    tmpdir = self.model_extracted_dir
                else:
                    # Extract the atommic file into the temporary directory
                    self._unpack_atommic_file(
                        path2file=restore_path, out_folder=tmpdir, extract_config_only=return_config is True
                    )

                # Change current working directory to the temporary directory
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = self.model_config_yaml
                else:
                    # can be str path or OmegaConf / DictConfig object
                    config_yaml = override_config_path
                if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                    conf = OmegaConf.load(config_yaml)
                else:
                    conf = config_yaml
                    if override_config_path is not None:
                        # Resolve the override config
                        conf = OmegaConf.to_container(conf, resolve=True)
                        conf = OmegaConf.create(conf)
                # If override is top level config, extract just `model` from it
                if "model" in conf:
                    conf = conf.model

                if return_config:
                    instance = conf
                    return instance

                if app_state.model_parallel_rank is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                else:
                    model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                OmegaConf.set_struct(conf, True)
                os.chdir(cwd)
                # get the class
                calling_cls._set_model_restore_state(  # pylint: disable=protected-access
                    is_being_restored=True, folder=tmpdir
                )
                instance = calling_cls.from_config_dict(config=conf, trainer=trainer)
                instance = instance.to(map_location)
                # add load_state_dict override
                if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                state_dict = self._load_state_dict_from_disk(model_weights, map_location=map_location)
            finally:
                os.chdir(cwd)

        return (conf, instance, state_dict)

    @staticmethod
    def load_instance_with_state_dict(instance, state_dict, strict):
        """Loads the state dict into the instance."""
        instance.load_state_dict(state_dict, strict=strict)
        instance._set_model_restore_state(is_being_restored=False)  # pylint: disable=protected-access

    def restore_from(
        self,
        calling_cls,
        restore_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
    ):
        """Restores model instance (weights and configuration) into .atommic file

        Parameters
        ----------
        calling_cls : class
            The class of the model to be restored.
        restore_path : str
            Path to .atommic file from which model should be instantiated.
        override_config_path : str or OmegaConf/DictConfig object, optional
            Path to a yaml config that will override the internal config file or an OmegaConf/DictConfig object
            representing the model config.
        map_location : torch.device, optional
            Optional torch.device() to map the instantiated model to a device. By default (None), it will select a
            GPU if available, falling back to CPU otherwise.
        strict : bool, optional
            Passed to load_state_dict. Default is ``True``.
        return_config : bool, optional
            If set to true, will return just the underlying config of the restored model as an OmegaConf/DictConfig
            object without instantiating the model.
        trainer : Trainer, optional
            Optional trainer object to be used for restoring the model.

        Returns
        -------
        An instance of type cls or its underlying config (if return_config is set).
        """
        # Get path where the command is executed - the artifacts will be "retrieved" there (original .atommic behavior)
        loaded_params = self.load_config_and_state_dict(
            calling_cls, restore_path, override_config_path, map_location, strict, return_config, trainer
        )
        if not isinstance(loaded_params, tuple) or return_config is True:
            return loaded_params
        _, instance, state_dict = loaded_params
        self.load_instance_with_state_dict(instance, state_dict, strict)
        logging.info(f"Model {instance.__class__.__name__} was successfully restored from {restore_path}.")
        return instance, state_dict

    def extract_state_dict_from(self, restore_path: str, save_dir: str, split_by_module: bool = False):
        """Extract the state dict(s) from a provided .atommic tarfile and save it to a directory.

        Parameters
        ----------
        restore_path : str
            Path to .atommic file from which state dict(s) should be extracted.
        save_dir : str
            Directory in which the saved state dict(s) should be stored.
        split_by_module : bool, optional
            Bool flag, which determines whether the output checkpoint should be for the entire Model, or
            the individual module's that comprise the Model. Default is ``False``.

        Example
        -------
        To convert the .atommic tarfile into a single Model level PyTorch checkpoint
        ::
        state_dict = atommic.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.atommic',
        './asr_ckpts')
        To restore a model from a Model level checkpoint
        ::
        model = atommic.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
        model.load_state_dict(torch.load("./asr_ckpts/model_weights.ckpt"))
        To convert the .atommic tarfile into multiple Module level PyTorch checkpoints
        ::
        state_dict = atommic.collections.asr.models.EncDecCTCModel.extract_state_dict_from('asr.atommic',
        './asr_ckpts', split_by_module=True). To restore a module from a Module level checkpoint
        ::
        model = atommic.collections.asr.models.EncDecCTCModel(cfg)  # or any other method of restoration
        # load the individual components
        model.preprocessor.load_state_dict(torch.load("./asr_ckpts/preprocessor.ckpt"))
        model.encoder.load_state_dict(torch.load("./asr_ckpts/encoder.ckpt"))
        model.decoder.load_state_dict(torch.load("./asr_ckpts/decoder.ckpt"))

        Returns
        -------
        The state dict that was loaded from the original .atommic checkpoint.
        """
        cwd = os.getcwd()

        save_dir = os.path.abspath(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                self._unpack_atommic_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                state_dict = self._load_state_dict_from_disk(model_weights)

                if not split_by_module:
                    filepath = os.path.join(save_dir, self.model_weights_ckpt)
                    self._save_state_dict_to_disk(state_dict, filepath)

                else:
                    key_set = {key.split(".")[0] for key in state_dict.keys()}
                    for primary_key in key_set:
                        inner_keys = [key for key in state_dict.keys() if key.split(".")[0] == primary_key]
                        state_dict_subset = {
                            ".".join(inner_key.split(".")[1:]): state_dict[inner_key] for inner_key in inner_keys
                        }
                        filepath = os.path.join(save_dir, f"{primary_key}.ckpt")
                        self._save_state_dict_to_disk(state_dict_subset, filepath)

                logging.info(f"Checkpoints from {restore_path} were successfully extracted into {save_dir}.")
            finally:
                os.chdir(cwd)

        return state_dict

    @staticmethod
    def register_artifact(model, config_path: str, src: str, verify_src_exists: bool = True):
        """Register model artifacts with this function. These artifacts (files) will be included inside .atommic file
        when model.save_to("mymodel.atommic") is called.

        How it works:
        1. It always returns existing absolute path which can be used during Model constructor call. EXCEPTION: src is
        None or "" in which case nothing will be done and src will be returned
        2. It will add (config_path, model_utils.ArtifactItem()) pair to self.artifacts. If "src" is local existing
        path, then it will be returned in absolute path form. elif "src" starts with
        "atommic_file:unique_artifact_name": .atommic will be untarred to a temporary folder location and an actual
        existing path will be returned else an error will be raised.

            .. code-block::

              If "src" is local existing path:
                  then it will be returned in absolute path form
              elif "src" starts with "atommic_file:unique_artifact_name":
                  .atommic will be untarred to a temporary folder location and an actual existing path will be returned
              else:
                  an error will be raised.

        WARNING: use .register_artifact calls in your models' constructors.
        The returned path is not guaranteed to exist after you have exited your model's constructor.

        Parameters
        ----------
        model : ModelPT
            ModelPT object to register artifact for.
        config_path : str
            Artifact key. Usually corresponds to the model config.
        src : str
            Path to artifact.
        verify_src_exists : bool, optional
            If set to False, then the artifact is optional and register_artifact will return None even if src is not
            found. Default is ``True``.

        Returns
        --------
        If src is not None or empty it always returns absolute path which is guaranteed to exist during model instance
         life.
        """
        app_state = AppState()

        artifact_item = model_utils.ArtifactItem()

        # This is for backward compatibility, if the src objects exists simply inside the tarfile
        # without its key having been overridden, this pathway will be used.
        src_obj_name = os.path.basename(src)
        if app_state.atommic_file_folder is not None:
            src_obj_path = os.path.abspath(os.path.join(app_state.atommic_file_folder, src_obj_name))
        else:
            src_obj_path = src_obj_name

        # src is a local existing path - register artifact and return exact same path for usage by the model
        if os.path.exists(os.path.abspath(src)):
            return_path = os.path.abspath(src)
            artifact_item.path_type = model_utils.ArtifactPathType.LOCAL_PATH

        elif src.startswith("atommic:"):
            return_path = os.path.abspath(os.path.join(app_state.atommic_file_folder, src[5:]))
            artifact_item.path_type = model_utils.ArtifactPathType.TAR_PATH

        elif os.path.exists(src_obj_path):
            return_path = src_obj_path
            artifact_item.path_type = model_utils.ArtifactPathType.TAR_PATH
        elif verify_src_exists:
            raise FileNotFoundError(
                f"src path does not exist or it is not a path in atommic file. src value I got was: {src}. "
                f"Absolute: {os.path.abspath(src)}"
            )
        else:
            # artifact is optional and we simply return None
            return None

        if not os.path.exists(return_path):
            raise AssertionError

        artifact_item.path = os.path.abspath(src)
        model.artifacts[config_path] = artifact_item
        # we were called by ModelPT
        if hasattr(model, "cfg"):
            with open_dict(model._cfg):  # pylint: disable=protected-access
                OmegaConf.update(model.cfg, config_path, return_path)
        return return_path

    def _handle_artifacts(self, model, atommic_file_folder):  # noqa: MC0001
        """This method is called by ModelPT.save_to() and ModelPT.load_from(). It will handle all artifacts and save
        them to the atommic file.

        Parameters
        ----------
        model : ModelPT
            ModelPT object to handle artifacts for.
        atommic_file_folder : str
            Path to temporary folder where atommic file will be untarred.
        """
        tarfile_artifacts = []
        app_state = AppState()

        # aggregate artifacts from self and all children recursively
        artifacts_containers = []
        for _, config_path, module in model.named_atommic_modules():
            if module.has_artifacts():  # atommic model with artifacts
                artifacts_containers.append((config_path, module.artifacts))

        if len(artifacts_containers) > 0 and (not hasattr(model, "artifacts") or model.artifacts is None):
            # model has no artifacts, but submodules have some
            model.artifacts = {}

        for config_path, artifacts in artifacts_containers:
            for subconf_path, artiitem in artifacts.items():
                conf_path = f"{config_path}.{subconf_path}" if config_path else f"{subconf_path}"
                if artiitem.path_type == model_utils.ArtifactPathType.LOCAL_PATH:
                    if not os.path.exists(artiitem.path):
                        raise FileNotFoundError(f"Artifact {conf_path} not found at location: {artiitem.path}")

                    # Generate new uniq artifact name and copy it to atommic_file_folder
                    # Note uuid.uuid4().hex is guaranteed to be 32 character long
                    artifact_base_name = os.path.basename(artiitem.path)
                    artifact_uniq_name = f"{uuid.uuid4().hex}_{artifact_base_name}"
                    shutil.copy2(artiitem.path, os.path.join(atommic_file_folder, artifact_uniq_name))

                    # Update artifacts registry
                    artiitem.hashed_path = "atommic:" + artifact_uniq_name
                    model.artifacts[conf_path] = artiitem

                elif artiitem.path_type == model_utils.ArtifactPathType.TAR_PATH:
                    # process all tarfile artifacts in one go, so preserve key-value pair
                    tarfile_artifacts.append((conf_path, artiitem))
                    if subconf_path:  # artifact from submodule
                        model.artifacts[conf_path] = artiitem

                else:
                    raise ValueError("Directly referencing artifacts from other atommic files isn't supported yet")

        # Process current tarfile artifacts by unpacking the previous tarfile and extract the artifacts
        # that are currently required.
        # artifacts can be native (from the model itself) and from submodules
        # model + submodules restoration paths, handle only unique paths
        restoration_paths: Set[str] = set()
        model_metadata = app_state.get_model_metadata_from_guid(model.model_guid)
        if model_metadata.restoration_path is not None:
            restoration_paths.add(model_metadata.restoration_path)
        # aggregate restoration paths for all submodules recursively
        for module in model.modules():
            if isinstance(module, atommic_classes.modelPT.ModelPT):  # if atommic model
                submodule_restoration_path = app_state.get_model_metadata_from_guid(module.model_guid).restoration_path
                if submodule_restoration_path is not None:
                    restoration_paths.add(submodule_restoration_path)
        if len(tarfile_artifacts) > 0 and len(restoration_paths) == 0:
            # TODO: see cases when this can occur, and if we can fix them
            logging.warning("Model contains registered artifacts, but no restoration paths found")
        if len(tarfile_artifacts) > 0 and len(restoration_paths) > 0:
            # Need to step into atommic archive to extract file
            # Get path where the command is executed - the artifacts will be "retrieved" there
            # (original .atommic behavior)
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as archive_dir:
                # Step into the atommic archive to try and find the file
                try:
                    # unpack all restorations paths (atommic checkpoints)
                    # in atommic checkpoints all resources contain hash in name, so there should be no collisions
                    for path in restoration_paths:
                        if self.model_extracted_dir:
                            shutil.copytree(src=path, dst=archive_dir, dirs_exist_ok=True)
                        else:
                            self._unpack_atommic_file(path2file=path, out_folder=archive_dir)
                    os.chdir(archive_dir)
                    for conf_path, artiitem in tarfile_artifacts:
                        # Get basename and copy it to atommic_file_folder
                        if "atommic:" in artiitem.path:
                            artifact_base_name = artiitem.path.split("atommic:")[1]
                        else:
                            artifact_base_name = os.path.basename(artiitem.path)
                        # no need to hash here as we are in tarfile_artifacts which are already hashed
                        artifact_uniq_name = artifact_base_name
                        shutil.copy2(artifact_base_name, os.path.join(atommic_file_folder, artifact_uniq_name))

                        # Update artifacts registry
                        new_artiitem = model_utils.ArtifactItem()
                        new_artiitem.path = "atommic:" + artifact_uniq_name
                        new_artiitem.path_type = model_utils.ArtifactPathType.TAR_PATH
                        model.artifacts[conf_path] = new_artiitem
                finally:
                    # change back working directory
                    os.chdir(cwd)

    @staticmethod
    def _update_subconfigs(model: "atommic_classes.ModelPT", path2yaml_file):  # type: ignore
        """Update subconfigs if ModelPT has submodules. Should be called before updating artifacts paths."""
        if not model.has_atommic_submodules():
            # no submodules => nothing to update
            return
        conf = OmegaConf.load(path2yaml_file)
        # update subconfigs for all children recursively, parent configs updated before children
        for _, conf_path, submodule in model.named_atommic_modules():
            if not conf_path:  # self
                continue
            OmegaConf.update(conf, conf_path, submodule.cfg)
        with open(path2yaml_file, "w", encoding="utf-8") as fout:
            OmegaConf.save(config=conf, f=fout, resolve=True)

    @staticmethod
    def _update_artifact_paths(model, path2yaml_file):
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to update the artifact paths in the
        model.
        """
        if hasattr(model, "artifacts") and model.artifacts is not None and len(model.artifacts) > 0:
            conf = OmegaConf.load(path2yaml_file)
            for conf_path, item in model.artifacts.items():
                if item.hashed_path is None:
                    OmegaConf.update(conf, conf_path, item.path)
                else:
                    OmegaConf.update(conf, conf_path, item.hashed_path)
            with open(path2yaml_file, "w", encoding="utf-8") as fout:
                OmegaConf.save(config=conf, f=fout, resolve=True)

    @staticmethod
    def _inject_model_parallel_rank_for_ckpt(dirname, basename):
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to inject the parallel rank of the
        process into the checkpoint file name.
        """
        model_weights = os.path.join(dirname, basename)
        model_weights = model_utils.inject_model_parallel_rank(model_weights)
        return model_weights

    @staticmethod
    def _make_atommic_file_from_folder(filename, source_dir):
        """The method is called by ModelPT.save_to() and ModelPT.load_from() to create a atommic file from a folder."""
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        with tarfile.open(filename, "w") as tar:
            tar.add(source_dir, arcname=".")

    @staticmethod
    def _unpack_atommic_file(path2file: str, out_folder: str, extract_config_only: bool = False) -> str:
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to unpack a atommic file."""
        if not os.path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        # we start with an assumption of uncompressed tar, which should be true for versions 1.7.0 and above
        tar_header = "r:"
        try:
            tar_test = tarfile.open(path2file, tar_header)  # pylint: disable=consider-using-with
            tar_test.close()
        except tarfile.ReadError:
            # can be older checkpoint => try compressed tar
            tar_header = "r:gz"
        tar = tarfile.open(path2file, tar_header)  # pylint: disable=consider-using-with
        if not extract_config_only:
            tar.extractall(path=out_folder)
        else:
            members = [x for x in tar.getmembers() if ".yaml" in x.name]
            tar.extractall(path=out_folder, members=members)
        tar.close()
        return out_folder

    @staticmethod
    def _save_state_dict_to_disk(state_dict, filepath):
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to save the state dict to disk."""
        torch.save(state_dict, filepath)

    @staticmethod
    def _load_state_dict_from_disk(model_weights, map_location="cpu"):
        """This method is called by ModelPT.save_to() and ModelPT.load_from() to load the state dict from disk."""
        return torch.load(model_weights, map_location=map_location)

    @property
    def model_config_yaml(self) -> str:
        """This property is used to get the path to the model config yaml file."""
        return self._model_config_yaml

    @model_config_yaml.setter
    def model_config_yaml(self, path: str):
        """This property is used to set the path to the model config yaml file."""
        self._model_config_yaml = path

    @property
    def model_weights_ckpt(self) -> str:
        """This property is used to get the path to the model weights ckpt file."""
        return self._model_weights_ckpt

    @model_weights_ckpt.setter
    def model_weights_ckpt(self, path: str):
        """This property is used to set the path to the model weights ckpt file."""
        self._model_weights_ckpt = path

    @property
    def model_extracted_dir(self) -> Optional[str]:
        return self._model_extracted_dir

    @model_extracted_dir.setter
    def model_extracted_dir(self, path: None):
        self._model_extracted_dir = path
