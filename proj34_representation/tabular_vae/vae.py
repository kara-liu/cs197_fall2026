"""
vae.py
======

The model (TabularVAE) and the loss function (vae_loss).

How a VAE works, in one line:

    x --encoder--> (mu, logvar) --sample--> z --decoder--> x_hat

  * The encoder does not output one point z. It outputs a bell curve (a mean `mu`
    and a log-variance `logvar`), and we SAMPLE z from it.
  * Sampling has no gradient, so we use the "reparameterization trick":
        z = mu + sigma * eps,    eps ~ N(0, 1),    sigma = exp(0.5 * logvar)
  * The decoder tries to rebuild x from z.

Loss = reconstruction + beta * KL
  * reconstruction: how well does x_hat match x?
        categorical part -> cross-entropy (one classification problem per variable)
        continuous part  -> squared error
  * KL: how far is the encoder's bell curve from a standard normal N(0, 1)?
        This keeps the latent space tidy, so later we can draw z ~ N(0, 1)
        and decode it into brand-new synthetic rows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularVAE(nn.Module):
    def __init__(self, input_dim, cat_dims, hidden_dim=64, latent_dim=8):
        """
        input_dim   width of the preprocessed matrix (preprocessor.input_dim_)
        cat_dims    list with the number of categories of each categorical variable,
                    e.g. [4, 3, 2]. The categorical part comes FIRST in every row.
        hidden_dim  size of the hidden layers
        latent_dim  size of z
        """
        super().__init__()
        self.cat_dims = list(cat_dims)
        self.latent_dim = latent_dim

        # Encoder: x -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> hidden -> raw outputs (same width as the input)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        # 1) encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10, 10)   # clamp keeps training stable

        # 2) sample z with the reparameterization trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # 3) decode. For categorical blocks the outputs are "logits" (raw scores),
        #    NOT probabilities; cross_entropy applies the softmax for us.
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n):
        """Generate n synthetic rows (in the preprocessed layout)."""
        z = torch.randn(n, self.latent_dim)   # z ~ N(0, 1)
        out = self.decoder(z)

        pieces, start = [], 0
        for k in self.cat_dims:
            probs = F.softmax(out[:, start:start + k], dim=1)        # logits -> probabilities
            choice = torch.multinomial(probs, num_samples=1).squeeze(1)  # draw one category
            pieces.append(F.one_hot(choice, num_classes=k).float())
            start += k
        pieces.append(out[:, start:])                                # continuous part as is
        return torch.cat(pieces, dim=1)


def vae_loss(x, x_hat, mu, logvar, cat_dims, beta=1.0):
    """
    Returns (total_loss, parts). `parts` is a dict with the three separate terms
    as plain numbers. Everything is averaged per row of the batch.

    x, x_hat : (batch, input_dim). Categorical one-hot blocks first, continuous part last.
    cat_dims : list of widths of the categorical blocks
    beta     : weight of the KL term
    """
    batch_size = x.size(0)
    cat_total = sum(cat_dims)

    # 1) Categorical reconstruction: one cross-entropy per categorical variable
    recon_cat = torch.zeros(())
    start = 0
    for k in cat_dims:
        target = x[:, start:start + k].argmax(dim=1)     # one-hot -> class number
        logits = x_hat[:, start:start + k]
        recon_cat = recon_cat + F.cross_entropy(logits, target, reduction="sum")
        start += k
    recon_cat = recon_cat / batch_size

    # 2) Continuous reconstruction: squared error, summed over the continuous columns
    recon_cont = F.mse_loss(x_hat[:, cat_total:], x[:, cat_total:], reduction="sum") / batch_size

    # 3) KL divergence between N(mu, sigma^2) and N(0, 1), summed over latent dimensions
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    total = recon_cat + recon_cont + beta * kl
    parts = {"recon_cat": recon_cat.item(), "recon_cont": recon_cont.item(), "kl": kl.item()}
    return total, parts


## This was generated with help from Claude.
