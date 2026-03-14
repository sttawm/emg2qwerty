# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset, Seq2SeqVocab
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import *
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

class AbstractCTCModule(pl.LightningModule):
    def __init__(self, 
                 optimizer: DictConfig,
                 lr_scheduler: DictConfig,
                 decoder: DictConfig,
    ):
        super().__init__()
        
        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        if phase == 'train':
            return
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer, # type: ignore
            lr_scheduler_config=self.hparams.lr_scheduler, # type: ignore
        )

class TDSConvCTCModule(AbstractCTCModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__(optimizer, lr_scheduler, decoder)

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )
        
        self.save_hyperparameters()


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

class TransformerCTCModule(AbstractCTCModule):
    def __init__(
        self,
        in_channels,
        block_channels,
        kernel_size,
        d_model,
        d_mlp,
        nhead,
        n_encode_blocks,
        dropout,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        ):
        
        super().__init__(optimizer, lr_scheduler, decoder)
        
        self.embedding = FrameWiseEncoder(
            in_channels,
            block_channels,
            kernel_size,
            d_model
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_mlp, dropout=dropout),
            num_layers=n_encode_blocks,
        )
        
        self.to_output = nn.Sequential(
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        ) 
        
        self.save_hyperparameters()
        
    def forward(self, inputs: torch.Tensor, src_pad_mask=None) -> torch.Tensor:
        inputs = self.embedding(inputs)
        inputs = self.transformer_encoder(inputs, src_key_padding_mask=src_pad_mask)
        return self.to_output(inputs)
    
    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size
        
        T = inputs.size(0)

        time_indices = torch.arange(T, device=inputs.device)  
        src_key_padding_mask = time_indices.unsqueeze(0) >= input_lengths.unsqueeze(1)

        emissions = self.forward(inputs, src_key_padding_mask)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss


class TransformerCEModule(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        block_channels,
        kernel_size,
        d_model,
        d_mlp,
        nhead,
        n_encode_blocks,
        n_decode_blocks,
        dropout,
        max_decode_steps,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ):
        super().__init__()
        
        self.vocab = Seq2SeqVocab()
        self.vocab_size = self.vocab.vocab_size
        self.max_decode_steps = max_decode_steps
        
        self.d_model = d_model
        
        self.input_embedding = FrameWiseEncoder(
            in_channels,
            block_channels,
            kernel_size,
            d_model
        )
        
        self.output_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=self.vocab.pad_id)
        self.output_pos_enc = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=n_decode_blocks,
            num_encoder_layers=n_encode_blocks,
            dim_feedforward=d_mlp,
            dropout=dropout,
        )
        
        self.fc_out = nn.Linear(d_model, self.vocab_size)
        
        self.loss = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_id)
        
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
        
        self.save_hyperparameters()
        
    def target_embedding(self, tgt):
        return self.output_pos_enc(math.sqrt(self.d_model) * self.output_embedding(tgt))
        
    def forward(self, src, tgt, src_key_pad_mask, tgt_key_pad_mask):
        src_emb = self.input_embedding(src)
        
        tgt_emb = self.target_embedding(tgt)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[0], device=tgt.device)
        
        decoding = self.transformer(
            src_emb, 
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_pad_mask,
            memory_key_padding_mask=src_key_pad_mask,
            tgt_key_padding_mask=tgt_key_pad_mask,
            tgt_is_causal=True, # prevent attention on future decoded outputs
        )
        
        return self.fc_out(decoding)
    
    @torch.no_grad()
    def generate(self, src, src_key_pad_mask, max_len=None):
        if max_len is None:
            max_len = self.max_decode_steps
            
        src_emb = self.input_embedding(src)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_pad_mask)
        
        ys = torch.full((1, src.shape[1]), self.vocab.bos_id, dtype=torch.long, device=src.device) # 1, N
        
        unfinished = torch.ones(src.shape[1], dtype=torch.bool, device=src.device)
        
        for _ in range(max_len):
            seq_len = ys.shape[0]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=src.device)
            tgt_emb = self.target_embedding(ys)
            output = self.transformer.decoder(
                tgt_emb, 
                memory, 
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_pad_mask
            )
            
            logits = self.fc_out(output[-1, :, :])      # only care about the last word
            next_chars = torch.argmax(logits, dim=-1)
            next_chars = torch.where(unfinished, next_chars, torch.tensor(self.vocab.pad_id, device=src.device))
            ys = torch.cat((ys, next_chars.unsqueeze(0)), dim=0)
            
            unfinished = unfinished & (next_chars != self.vocab.eos_id)
            
            if unfinished.max() == 0:
                break
 
        return ys
    
    def _step(self, phase : str, batch: dict[str, torch.Tensor], *args, **kwargs):
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size
        
        targets = targets.long()
        tgt_seq_len = targets.shape[0] + 1

        tgt_in = torch.full(
            (tgt_seq_len, N),
            self.vocab.pad_id,
            dtype=torch.long,
            device=targets.device,
        )
        tgt_label = torch.full_like(tgt_in, self.vocab.pad_id)

        tgt_in[0, :] = self.vocab.bos_id

        for idx in range(N):
            length = int(target_lengths[idx].item())
            if length > 0:
                tgt_in[1 : length + 1, idx] = targets[:length, idx]
                tgt_label[:length, idx] = targets[:length, idx]
            tgt_label[length, idx] = self.vocab.eos_id
        
        

        src_key_pad_mask = torch.arange(inputs.shape[0], device=inputs.device).unsqueeze(0) >= input_lengths.unsqueeze(1)   # 1, S >= N, 1
        tgt_key_pad_mask = torch.arange(targets.shape[0]+1, device=targets.device).unsqueeze(0) >= \
            (target_lengths+1).unsqueeze(1) # 1, T >= N, 1, +1 due to <bos>
        
        logits = self(inputs, tgt_in, src_key_pad_mask, tgt_key_pad_mask)
        
        loss = self.loss(
            logits.reshape(-1, self.vocab_size),
            tgt_label.reshape(-1),
        )
        
        if phase != 'train':
            gen_preds = self.generate(inputs, src_key_pad_mask, int(target_lengths.max().item())+1)

            gen_preds = gen_preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            target_lengths_np = target_lengths.cpu().numpy()
            
            metrics = self.metrics[f"{phase}_metrics"]
                
            for i in range(N):
                pred_seq = self.vocab.strip_special_tokens(gen_preds[:, i].tolist())
                pred_label = LabelData.from_labels(pred_seq, _charset=self.vocab.base_charset)
                
                target_seq = targets_np[: target_lengths_np[i], i].tolist() # full target without dummy padding
                target_label = LabelData.from_labels(target_seq, _charset=self.vocab.base_charset)
                
                # Update the custom CharacterErrorRates metric
                metrics.update(prediction=pred_label, target=target_label)
        
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss 
        
    def training_step(self, batch: dict[str, torch.Tensor], *args, **kwargs)-> torch.Tensor:
        return self._step("train", batch, *args, **kwargs)
    
    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs)-> torch.Tensor:
        return self._step("val", batch, *args, **kwargs)
    
    def test_step(self, batch: dict[str, torch.Tensor], *args, **kwargs)-> torch.Tensor:
        return self._step("test", batch, *args, **kwargs)
    
    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()
        
    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")
        
    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer, # type: ignore
            lr_scheduler_config=self.hparams.lr_scheduler, # type: ignore
        )


class TransformerCTCFromConv(AbstractCTCModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features,
        mlp_features,
        conv_channels,
        kernel_width,
        d_model,
        d_mlp,
        nhead,
        n_encode_blocks,
        dropout,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ):
        super().__init__(optimizer, lr_scheduler, decoder)

        num_features = self.NUM_BANDS * mlp_features[-1]

        self.embedding = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=conv_channels,
                kernel_width=kernel_width,
            ),
            nn.Linear(num_features, d_model),
            PositionalEncoding(d_model, dropout=dropout),
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_mlp,
                dropout=dropout,
            ),
            num_layers=n_encode_blocks,
        )

        self.to_output = nn.Sequential(
            nn.Linear(d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.save_hyperparameters()

    def forward(self, inputs: torch.Tensor, src_pad_mask=None) -> torch.Tensor:
        inputs = self.embedding(inputs)
        inputs = self.transformer_encoder(inputs, src_key_padding_mask=src_pad_mask)
        return self.to_output(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        T = inputs.size(0)
        time_indices = torch.arange(T, device=inputs.device)
        src_key_padding_mask = time_indices.unsqueeze(0) >= input_lengths.unsqueeze(1)

        emissions = self.forward(inputs, src_key_padding_mask)

        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss
