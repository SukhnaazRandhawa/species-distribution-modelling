"""One neural network that predicts all species at once (SINR-style).

Follows the approach of Cole et al. (2023), "Spatial Implicit Neural
Representations for Global-Scale Species Mapping", simplified for this
dataset: a residual MLP over encoded coordinates, trained on presence-only
data with an "assume negative" loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.features import encode_coords


# ---------------------------------------------------------------- model
class ResidualBlock(nn.Module):
    """Two linear layers with a skip connection: out = x + f(x)."""

    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU())

    def forward(self, x):
        return x + self.net(x)


class SharedMLP(nn.Module):
    """Encoded location -> shared representation -> one logit per species."""

    def __init__(self, n_inputs, n_species, width=256, n_blocks=4, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, width), nn.ReLU(),
            *[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.classifier = nn.Linear(width, n_species)   # one output per species

    def forward(self, x):
        return self.classifier(self.encoder(x))         # logits, not probabilities


# ----------------------------------------------------------------- loss
def assume_negative_loss(logits_obs, species, logits_rand, pos_weight, full=True):
    """Presence-only loss.

    logits_obs  : (B, S) outputs at B observed sightings
    species     : (B,)   which species was seen at each sighting
    logits_rand : (B, S) outputs at B random locations on the globe
    pos_weight  : how much more a real sighting counts than an assumed absence
    full        : True  -> at a sighting, every OTHER species is assumed absent
                  False -> only the random locations provide negatives
    """
    B, S = logits_obs.shape
    rows = torch.arange(B, device=logits_obs.device)

    # log P(present) for the species actually seen
    loss_pos = -pos_weight * F.logsigmoid(logits_obs[rows, species])

    # Every species assumed absent at a random location
    loss_rand = -F.logsigmoid(-logits_rand).sum(dim=1)

    loss = loss_pos + loss_rand
    if full:
        # Other species assumed absent where this one was seen
        neg_obs = -F.logsigmoid(-logits_obs)
        neg_obs[rows, species] = 0.0                # not the observed species
        loss = loss + neg_obs.sum(dim=1)
    return (loss / S).mean()


# --------------------------------------------------------- train/predict
def pick_device():
    if torch.backends.mps.is_available():           # Apple Silicon GPU
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def random_locations(n, device):
    """n uniform random points on the sphere, as (lat, lon) in degrees."""
    lon = torch.rand(n, device=device) * 360 - 180
    lat = torch.rad2deg(torch.asin(torch.rand(n, device=device) * 2 - 1))
    return torch.stack([lat, lon], dim=1)


def encode_torch(locs, n_freq):
    """Same sine/cosine encoding as src.features.encode_coords, in torch."""
    lat = torch.deg2rad(locs[:, :1])
    lon = torch.deg2rad(locs[:, 1:])
    k = torch.arange(1, n_freq + 1, device=locs.device, dtype=locs.dtype)
    return torch.cat([torch.sin(k * lat), torch.cos(k * lat),
                      torch.sin(k * lon), torch.cos(k * lon)], dim=1)


def fit_predict_shared(train_locs, train_cols, query_locs, n_species,
                       pos_weight=2048.0, full_loss=True, n_freq=1,
                       width=256, n_blocks=4, dropout=0.3, epochs=10,
                       batch_size=2048, lr=5e-4, seed=0, verbose=True,
                       featurize=None, bg_pool_size=1_000_000):
    """Train the shared network and return (n_query, n_species) probabilities.

    featurize : optional function locs -> features (e.g. with climate).
                If given, random background locations are drawn from a
                pre-computed pool of bg_pool_size uniform points, because
                climate lookups cannot be done on the GPU each batch.
                If None, coordinates are encoded with n_freq frequencies and
                fresh random points are generated on the GPU every batch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()

    if featurize is None:
        featurize_np = lambda locs: encode_coords(locs, n_freq)
        sample_bg = lambda n: encode_torch(random_locations(n, device), n_freq)
    else:
        from src.pseudo_absences import uniform_sphere
        featurize_np = featurize
        pool = torch.tensor(featurize(uniform_sphere(
            bg_pool_size, np.random.default_rng(seed))), device=device)
        sample_bg = lambda n: pool[torch.randint(len(pool), (n,), device=device)]

    X = torch.tensor(featurize_np(train_locs), device=device)
    y = torch.tensor(train_cols, dtype=torch.long, device=device)
    model = SharedMLP(X.shape[1], n_species, width, n_blocks, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=device)
        total = 0.0
        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            x_rand = sample_bg(len(idx))
            loss = assume_negative_loss(model(X[idx]), y[idx], model(x_rand),
                                        pos_weight, full=full_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f'  epoch {epoch + 1}/{epochs}  loss {total / len(X):.4f}')

    # Predict in chunks to keep memory low.
    model.eval()
    Xq = featurize_np(query_locs)
    out = np.zeros((len(Xq), n_species), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(Xq), 20000):
            xb = torch.tensor(Xq[start:start + 20000], device=device)
            out[start:start + 20000] = torch.sigmoid(model(xb)).cpu().numpy()
    return out