from __future__ import annotations

import sys
sys.path.append("../")
import os
from math import log
from math import pi as PI
import numpy as np 
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from data import DatasetKmers
from torch.optim import Adam


BATCH_SIZE = 256 
ENCODER_WARMUP_STEPS = 100
DECODER_WARMUP_STEPS = 100
AGGRESSIVE_STEPS = 5 


def encoder_lr_sched(step):
    # Use Linear warmup
    return min(step / ENCODER_WARMUP_STEPS, 1.0)


def decoder_lr_sched(step):
    if step < ENCODER_WARMUP_STEPS:
        return 0.0
    else:
        if (step - ENCODER_WARMUP_STEPS + 1) % AGGRESSIVE_STEPS == 0:
            return min(
                (step - ENCODER_WARMUP_STEPS)
                / (DECODER_WARMUP_STEPS * AGGRESSIVE_STEPS),
                1.0,
            )
        else:
            return 0.0


def rbf_kernel(x, y, sigma=1.0):
    assert x.ndim == y.ndim == 2
    assert x.shape[1] == y.shape[1]

    nx, dim = x.shape
    ny, dim = y.shape

    x = x.unsqueeze(1).expand(nx, ny, dim)
    y = y.unsqueeze(0).expand(nx, ny, dim)
    return (-(x - y).pow(2) / (2 * sigma ** 2)).mean(dim=2).exp()


def polynomial_kernel(x, y, c=0.0, d=4.0):
    assert x.ndim == y.ndim == 2
    assert x.shape[1] == y.shape[1]

    nx, dim = x.shape
    ny, dim = y.shape

    x = x.unsqueeze(1).expand(nx, ny, dim)
    y = y.unsqueeze(0).expand(nx, ny, dim)

    return ((x * y).mean(dim=2) + c).pow(d)


def gaussian_nll(x, mu, sigma):
    return sigma.log() + 0.5 * (log(2 * PI) + ((x - mu) / sigma).pow(2))


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    dim: int = -1,
    return_randoms: bool = False,
    randoms: Tensor = None,
) -> Tensor:
    """
    Mostly from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    """
    if randoms is None:
        randoms = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
    gumbels = (logits + randoms) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if return_randoms:
        return ret, randoms
    else:
        return ret


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)



class InfoTransformerVAE(pl.LightningModule):
    def __init__(
        self,
        # dataset: ProtiensDataset,
        dataset: DatasetKmers, 
        d_model: int = 8,
        is_autoencoder: bool = False,
        kl_factor: float = 0.0001,
        min_posterior_std: float = 1e-4,
        n_samples_mmd: int = 2,
        encoder_nhead: int = 12,
        encoder_dim_feedforward: int = 256,
        encoder_dropout: float = 0.05,
        encoder_num_layers: int = 6,
    ):
        super().__init__() 

        self.max_string_length = 1024 # by default 

        self.dataset = dataset
        self.vocab_size = len(self.dataset.vocab)

        self.d_model = d_model
        self.is_autoencoder = is_autoencoder

        self.kl_factor = kl_factor

        self.min_posterior_std = min_posterior_std
        self.n_samples_mmd = n_samples_mmd
        encoder_embedding_dim = d_model * 2
        decoder_embedding_dim = d_model

        self.encoder_token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=encoder_embedding_dim
        )
        self.encoder_position_encoding = PositionalEncoding(
            encoder_embedding_dim, dropout=encoder_dropout, max_len=5_000
        )
        
        self.decoder_token_unembedding = nn.Parameter(
            torch.randn(d_model, self.vocab_size)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_embedding_dim,
                nhead=encoder_nhead,
                dim_feedforward=encoder_dim_feedforward,
                dropout=encoder_dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=encoder_num_layers,
        )

        

        # also encoder layers, but doing decoder stuff
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decoder_embedding_dim,
                nhead=encoder_nhead,
                dim_feedforward=encoder_dim_feedforward,
                dropout=encoder_dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=encoder_num_layers,
        )

    def sample_prior(self, n):

        sequence_length = self.sequence_length

        return torch.randn(n, sequence_length, 8).to(self.device)

    def sample_posterior(self, mu, sigma, n=None):
        if n is not None:
            mu = mu.unsqueeze(0).expand(n, -1, -1, -1)

        return mu + torch.randn_like(mu) * sigma

    def generate_pad_mask(self, tokens):
        """Generate mask that tells encoder to ignore all but first stop token"""
        mask = tokens == 1
        inds = mask.float().argmax(
            dim=-1
        )  # Returns first index along axis when multiple present
        mask[torch.arange(0, tokens.shape[0]), inds] = False
        return mask

    def encode(self, tokens, as_probs=False):
        if as_probs:
            embed = tokens @ self.encoder_token_embedding.weight
        else:
            embed = self.encoder_token_embedding(tokens)

        embed = self.encoder_position_encoding(embed)

        pad_mask = self.generate_pad_mask(tokens)
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)
        mu = encoding[..., : self.d_model]
        sigma = F.softplus(encoding[..., self.d_model :]) + self.min_posterior_std

        return mu, sigma

    def decode(self, z, tokens):
        
        pad_mask = self.generate_pad_mask(tokens)
        
        decoding = self.decoder(z, src_key_padding_mask=pad_mask)

        logits = decoding @ self.decoder_token_unembedding

        return logits

    # @torch.no_grad()
    # def sample(
    #     self,
    #     n: int = -1,
    #     z: Tensor = None,
    #     differentiable: bool = False,
    #     return_logits: bool = False,
    # ):
    #     model_state = self.training
    #     self.eval()
    #     if z is None:
    #         z = self.sample_prior(n)
    #     else:
    #         n = z.shape[0]

    #     tokens = torch.zeros(
    #         n, 1, device=self.device
    #     ).long()  # Start token is 0, stop token is 1
    #     random_gumbels = torch.zeros(n, 0, self.vocab_size, device=self.device)
    #     while True:  # Loop until every molecule hits a stop token
    #         tgt = self.decoder_token_embedding(tokens)
    #         tgt = self.decoder_position_encoding(tgt)
    #         tgt_mask = nn.Transformer.generate_square_subsequent_mask(
    #             sz=tokens.shape[-1]
    #         ).to(self.device)

    #         decoding = self.decoder(tgt=tgt, memory=z, tgt_mask=tgt_mask)
    #         logits = decoding @ self.decoder_token_unembedding
    #         sample, randoms = gumbel_softmax(
    #             logits, dim=-1, hard=True, return_randoms=True
    #         )

    #         tokens = torch.cat(
    #             [tokens, sample[:, -1, :].argmax(dim=-1)[:, None]], dim=-1
    #         )
    #         random_gumbels = torch.cat([random_gumbels, randoms], dim=1)

    #         # 1 is the stop token. Check if all molecules have a stop token in them
    #         if (
    #             torch.all((tokens == 1).sum(dim=-1) > 0).item()
    #             or tokens.shape[-1] > self.max_string_length
    #         ):  # no longer break at 1024, instead variable max string length 
    #             break

    #     self.train(model_state)

    #     # TODO: Put this back in
    #     if not differentiable:
    #         sample = tokens

    #     if return_logits:
    #         return sample, logits
    #     else:
    #         return sample

    @staticmethod
    def _flatten_z(z):
        sh = z.shape
        if len(sh) == 3:
            return z.reshape(sh[0], sh[1] * sh[2])
        elif len(sh) == 4:
            return z.reshape(sh[0] * sh[1], sh[2] * sh[3])
        else:
            raise ValueError

    def sample_encoding(self, tokens):
        mu, sigma = self.encode(tokens)

        if self.is_autoencoder:
            z = mu
        else:
            z = self.sample_posterior(mu, sigma)

        return mu, sigma, z 

    def forward(self, tokens):
        mu, sigma, z = self.sample_encoding(tokens)
        z_norm = torch.sum(z ** 2, dim=-1).mean() 

        logits = self.decode(z, tokens)

        recon_loss = F.cross_entropy(
            logits.permute(0, 2, 1), tokens, reduction="none"
        ).mean()  # .sum(1).mean(0) 

        # No need for KL divergence when \alpha = 1 
        # see https://ojs.aaai.org//index.php/AAAI/article/view/4538 Eq. 6
        # Equation from the original "Auto-Encoding Variational Bayes" paper: https://arxiv.org/pdf/1312.6114.pdf
        sigma2 = sigma.pow(2)
        kldiv = (
            0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1).mean()
        )  # .sum(dim=(1, 2)).mean(0)


        primary_loss = recon_loss
        if self.kl_factor != 0:
            primary_loss = primary_loss + self.kl_factor * kldiv

        loss = primary_loss
        

        return dict(
            loss=loss,
            z=z,
            z_norm=z_norm,
            recon_loss=recon_loss,
            kldiv=kldiv,
            recon_token_acc=(logits.argmax(dim=-1) == tokens).float().mean(),
            recon_string_acc=(logits.argmax(dim=-1) == tokens)
            .all(dim=1)
            .float()
            .mean(dim=0),
            sigma_mean=sigma.mean(),
        )


class VAEModule(pl.LightningModule):
    def __init__(
        self,
        # dataset: ProtiensDataset, 
        dataset: DatasetKmers, 
        encoder_lr: float = 1e-3,
        decoder_lr: float = 1e-3,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.model = InfoTransformerVAE(dataset=dataset, *args, **kwargs)

    def training_step(self, data, batch_idx):
        batch = data

        def detach_return(d):
            return {k: (v.detach() if k != "loss" else v) for k, v in d.items()}

        outputs = detach_return(self.model(batch))

        for k, v in outputs.items():
            self.log("train/" + k, v.mean().item())

        return outputs

    def training_epoch_end(self, outputs):
        try:
            self.log("train/blosum_spearman", self.blosum.compute())
            self.blosum.reset()
        except:
            pass

    def validation_epoch_end(self, outputs):
        try:
            self.log("validation/blosum_spearman", self.blosum.compute())
            self.blosum.reset()
        except:
            pass
        return outputs

    def validation_step(self, data, batch_idx):
        batch = data
        outputs = self.model(batch)

        for k, v in outputs.items():
            self.log("validation/" + k, v.mean().item())

        return outputs

    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "encoder" in name:
                    encoder_params.append(param)
                elif "decoder" in name:
                    decoder_params.append(param)
                else:
                    raise ValueError(f"Unknown parameter {name}")

        optimizer = Adam(
            [
                dict(params=encoder_params, lr=self.encoder_lr),
                dict(params=decoder_params, lr=self.decoder_lr),
            ]
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [encoder_lr_sched, decoder_lr_sched]
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(scheduler=lr_scheduler, interval="step", frequency=1),
        )

config_dict = dict(
        loader_hint="Camelid_OAS-OAS_camel-sequences-public-3",
        max_epochs=300,
        gamma=0.1,
        batch_size=256,
        d_model=128,
        encoder_num_layers=6,
        # encoder_nhead=8,
        # encoder_dim_feedforward=512,
        decoder_num_layers=6,
        # decoder_nhead=8,
        decoder_dim_feedforward=512,
        encoder_dropout=0.05,
        decoder_dropout=0.05
    )


def load(dataset_path, checkpoint_path):
    # dataset = ProtiensDataset(dataset_path) 
    dataset = DatasetKmers(dataset_path )
    module = VAEModule.load_from_checkpoint(
        checkpoint_path, map_location=torch.device("cpu"), dataset=dataset,
        kl_factor=0.1 ) 

    return dict(dataset=dataset, module=module, model=module.model)
