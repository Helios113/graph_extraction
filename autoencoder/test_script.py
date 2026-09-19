import math
import torch
from batched_autoencoder import batched_manifold
# from autoencoder.autoencoder import manifold
import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mem(label):
    if device.type != "cuda":
        return
    alloc = torch.cuda.memory_allocated(device) / 2**30
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info(device))
    print(
        f"[mem] {label}: allocated={alloc:.2f}GiB peak={peak:.2f}GiB "
        f"reserved={reserved:.2f}GiB free={free:.2f}GiB total={total:.2f}GiB"
    )

print("pre datagen", flush=True)
a = time.time()
dummy_data = torch.empty(300, 50000, 786).normal_()
print(f"datagen {time.time()-a}", flush=True)

center = torch.mean(dummy_data, (1),keepdim=True)
scale = ((dummy_data - center) ** 2).sum(2, keepdim=True).mean(1, keepdim=True).sqrt()



@torch.no_grad()
def knn(x, k, query_set=None, chunk=4096, model_chunk=None, device=None, progress=False):
    """x, query_set: CPU tensors. Each (model_chunk, chunk) slice is moved to
    `device` only for the duration of its cdist/topk call."""

    self_q = query_set is None

    query_set = x if self_q else query_set

    data_size = x.shape[1]

    kk = min(k + int(self_q), data_size)
    batch, q_size = query_set.shape[0], query_set.shape[1]
    model_chunk = model_chunk or batch
    ds = torch.empty(batch, q_size, k, dtype=x.dtype)
    js = torch.empty(batch, q_size, k, dtype=torch.long)
    model_steps = range(0, batch, model_chunk)
    if progress:
        model_steps = tqdm.tqdm(model_steps)
    for m in model_steps:
        x_m = x[m:m + model_chunk].to(device)
        for s in range(0, q_size, chunk):
            d = torch.cdist(query_set[m:m + model_chunk, s:s + chunk].to(device), x_m)
            if self_q:
                b = torch.arange(d.shape[0], device=d.device)[:, None]
                r = torch.arange(d.shape[1], device=d.device)[None, :]
                d[b, r, s + r] = float("inf")
            dd, jj = d.topk(kk, largest=False, dim=-1)
            ds[m:m + model_chunk, s:s + chunk] = (dd[..., :k] if self_q else dd).cpu()
            js[m:m + model_chunk, s:s + chunk] = (jj[..., :k] if self_q else jj).cpu()
    return ds, js



mem("before knn")
dist, neighbor_idx = knn(dummy_data, 32, progress=True, chunk=512, model_chunk=8, device=device)
mem("after knn")
print(dist.shape)
print(neighbor_idx.shape)
data_size = dummy_data.shape[1]

ambient_spacing = dist[...,-1]

batch = dummy_data.shape[0]
offset = torch.arange(batch)[:, None] * data_size
counts = torch.bincount((neighbor_idx[:, :, 0] + offset).flatten(), minlength=batch * data_size)
counts = counts.reshape(batch, data_size)

hub_share = counts.sort(dim=-1, descending=True).values[:, :5].sum(dim=-1) / data_size


bmnfld = batched_manifold(center.to(device), scale.to(device), 300, network_dims=[786, 512,32]).to(device)
mem("after bmnfld construction")


# ---------------------------------------------------------------- training

epochs = 300
batch_size = 64
lr = 2e-3
lam_geom = 0.1

opt = torch.optim.Adam(bmnfld.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, lr, total_steps=epochs * math.ceil(data_size / batch_size), pct_start=0.05
)

model_arange = torch.arange(batch)[:, None, None]

for ep in tqdm.tqdm(range(epochs), desc="fit"):
    a = time.time()

    perm = torch.stack([torch.randperm(data_size) for _ in range(batch)])
    print("perm time:" ,time.time()-a)

    for s in tqdm.tqdm(range(0, data_size, batch_size)):
        t0 = time.time()

        idx = perm[:, s:s + batch_size]  # (batch, B)
        nbr_idx = neighbor_idx[model_arange[:, :, 0], idx]  # (batch, B, 32)

        x = dummy_data[model_arange[:, :, 0], idx].to(device)  # (batch, B, 786)
        neighbors = dummy_data[model_arange, nbr_idx].to(device)  # (batch, B, 32, 786)
        t1 = time.time()
        if ep == 0 and s == 0:
            mem("after first minibatch gather")

        z = bmnfld.encode(x)  # (batch, B, 32)
        z_neighbors = bmnfld.encode(
            neighbors.reshape(batch, -1, neighbors.shape[-1])
        ).reshape(batch, x.shape[1], bmnfld.n_neighbors, bmnfld.latent_dim)

        x_recon = bmnfld.decode(z)
        t2 = time.time()

        ambient_dist = (neighbors - x[:, :, None]).norm(dim=-1)
        latent_dist = (z_neighbors - z[:, :, None]).norm(dim=-1)

        recon_loss = ((x_recon - x) ** 2).sum(-1).mean()
        geom_loss = ((latent_dist - ambient_dist) ** 2).mean() / (ambient_dist ** 2).mean()

        loss = recon_loss + lam_geom * geom_loss
        t3 = time.time()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        t4 = time.time()

        print(f"gather: {t1 - t0:.4f}s  fwd: {t2 - t1:.4f}s  loss: {t3 - t2:.4f}s  bwd/step: {t4 - t3:.4f}s")
        mem(f"epoch {ep} step {s}")

    print("epoch time:", time.time() - a)
    mem(f"end of epoch {ep}")