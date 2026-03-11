#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train KenLM fusion layer on top of a frozen acoustic model.

Usage:
    python train_fusion.py fusion_module.acoustic_checkpoint=path/to/checkpoint.ckpt
    python train_fusion.py fusion_module.acoustic_checkpoint=gs://... fusion_module.n_beams=10
"""

import logging
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.fusion import KenLMFusionModule
from emg2qwerty.transforms import Transform

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="fusion_base")
def main(config: DictConfig) -> None:
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    pl.seed_everything(config.seed, workers=True)

    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate fusion module
    log.info("Instantiating KenLMFusionModule")
    module = KenLMFusionModule(**config.fusion_module)

    # Instantiate data module (same as acoustic model training)
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    callbacks = [instantiate(cfg) for cfg in config.get("callbacks", [])]

    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)

    trainer.fit(module, datamodule)

    # Load best checkpoint and evaluate
    module = KenLMFusionModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    val_metrics = trainer.validate(module, datamodule)
    test_metrics = trainer.test(module, datamodule)

    pprint.pprint(
        {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": trainer.checkpoint_callback.best_model_path,
        },
        sort_dicts=False,
    )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
