# gmanifold

## What it's for

Take a bunch of real vectors — say, the embedding of every token in a
vocabulary, or the residual-stream activations of a Transformer at some
layer. Those points aren't scattered randomly through space: they sit near
some lower-dimensional surface (a *manifold*) folded into the high-dimensional
ambient space.

`gmanifold` does two things with that surface:

1. **Learns it.** Fits a smooth encoder/decoder pair that maps back and forth
   between the real, high-dimensional points and a small set of latent
   coordinates — a compressed, low-dimensional description of "where on the
   surface" each point sits.
2. **Samples new points from it.** Not just interpolating between two real
   points, but generating brand-new points that sit *plausibly on the same
   surface*, near the real data, distributed evenly across the surface's
   own area (not clumped wherever the real data happens to be dense).

The point of all this: you get synthetic-but-plausible activation states you
can feed into a real model and watch what happens — without those states
ever having occurred in real text.

## Why "volume-uniform" matters

Imagine the surface is a crumpled sheet of paper floating in 3D space. If you
just interpolated between two nearby points on it, you'd only ever sample
along the line between them — a 1D slice of a 2D surface. If you sampled
"uniformly in the flat coordinates" you use to describe the sheet, you'd
actually get *more* samples per unit area wherever the paper is more
crumpled, because a flat square of coordinate-space covers less actual sheet
surface where it's folded tightly.

`gmanifold` corrects for this: it samples so that every patch of the
*actual, crumpled* surface gets a fair share of samples, proportional to its
real surface area — not proportional to how the coordinates happen to be
laid out. That correction is what "volume-uniform" means here, and it's the
main piece of math the library carries.

## How it works, piece by piece

### 1. Learn the surface: `GlobalManifold`

- An **encoder** `E` squashes each real point (dimension `D`, e.g. 384) down
  to a small **latent** coordinate (dimension `m`, e.g. 16).
- A **decoder** `G` does the reverse: latent coordinate → point back in `D`
  dimensions.
- Both are small feed-forward networks, trained together so that:
  - decoding what you just encoded gets you back the original point
    (reconstruction loss),
  - nearby points in the real data stay nearby in latent space, and vice
    versa (a "don't distort local distances" loss) — this is what makes the
    decoder's local stretching factor a meaningful, trustworthy quantity
    later on.
- The learned surface is the decoder's image of small balls placed around
  every real point's latent coordinate — i.e. "the surface near where we
  actually have data," not a claim about the true manifold far from the
  observed points.

### 2. Measure how much the surface stretches: the Jacobian

At any latent coordinate, the decoder locally stretches or squashes space by
some factor as it maps `m` latent dimensions up into `D` ambient dimensions.
That stretching factor is exactly what "volume-uniform" sampling needs to
correct for — over-sample the parts that get stretched a lot, so the
resulting *ambient* coverage comes out even.

This stretching factor is computed from the decoder's **Jacobian** (its
matrix of partial derivatives at that point). In the original standalone
`gmanifold` library this was done directly in PyTorch with `torch.func.jacfwd`,
autodiff computing the exact derivative on the spot.

### 3. Sample: propose, weight, resample

To generate `n` new points:

1. Propose far more candidate latent points than needed, scattered in balls
   around the real data's latent coordinates.
2. For each candidate, compute the decoder's local stretch factor (the
   Jacobian-derived volume element) and use it as an importance weight —
   candidates in heavily-stretched regions get up-weighted so, after
   decoding, coverage is even across real surface area.
3. Resample `n` points from the candidates according to those weights, then
   decode them to get the final points.

There's a built-in sanity check here too: if a handful of candidates end up
dominating all the weight (a sign the fit or the sampling radius is
unreliable), the library warns loudly and can automatically temper the
weights to fix it.

### 4. Check the samples are trustworthy: validation tools

Because the learned surface is only a local approximation, `gmanifold` ships
independent ways to check how far you can trust it:

- **Reconstruction error** and **nearest-real distance** — how far a sample
  sits from the closest real point, in units of that region's natural
  spacing.
- **Kernel density score** — an independent density estimate of "how on-manifold"
  a point looks, unrelated to the decoder itself.
- **Tangent charts** — a second, much simpler manifold approximation (local
  PCA planes instead of a trained network), used purely as a cross-check
  against the trained decoder.
- **Calibration bands** — every diagnostic is reported next to two reference
  numbers: held-out real points (the best case) and real points nudged by a
  bit of noise (a known-bad case). Samples are judged relative to those
  bands, not in a vacuum.

### 5. Push samples through a real model

Once you have synthetic points near, say, the embedding manifold, the last
step is running them through an actual slice of a real Transformer (one
sub-layer, like a single FFN block) and checking where they land — validated
against an **independently fitted** manifold of the destination layer's real
activations, never one trained on the synthetic samples themselves. This is
the actual interpretability payoff: "if the model saw an activation state
that plausibly could have occurred but didn't, what would it do with it?"

## What changed when this was migrated into `new_jac`

The original `gmanifold` repo used live PyTorch + Hugging Face forward hooks
to read a Transformer's real activations and to run a sub-layer's forward
pass. That part has been swapped out for this repo's existing **ONNX**
pipeline:

- The autoencoder (`GlobalManifold`, steps 1–3 above) is **unchanged, pure
  PyTorch** — the Jacobian/volume-element math still uses `torch.func`, since
  that's not something this repo's ONNX approach needs to replace.
- Reading real activations and running a sub-layer forward pass
  ([`gmanifold/onnx_transformer.py`](onnx_transformer.py)) now goes through
  this repo's ONNX export + `onnxruntime` machinery instead of live
  HF forward hooks, using the same sublayer-boundary conventions
  (`configs/*.toml`) as the rest of the repo.
- One real constraint fell out of that switch: only **tokenwise** sub-layers
  (ones whose output at a position depends only on that position's own
  input, e.g. a feed-forward block) can be extracted as a self-contained ONNX
  subgraph and pushed through this way. A sub-layer that crosses attention
  (mixes information across sequence positions) can't be isolated the same
  way an ONNX subgraph, since ORT has no equivalent of a forward hook
  splicing a value into a running computation graph.

See [onnx_transformer.py](onnx_transformer.py) for the implementation and
[`../scripts/gmanifold/sweep.py`](../scripts/gmanifold/sweep.py) /
[`../examples/gmanifold/simple_stories.py`](../examples/gmanifold/simple_stories.py)
for worked examples end-to-end.
