SAMPLE TANGENT SPACE DIRECT:

RUAPEHU:
sbatch --job-name=tangent_sweep_direct_driver --partition=hopper --nodelist=ruapehu --time=12:00:00 \
  --cpus-per-task=1 --mem=2G --output=logs/run_layer_batched_tangent_sweep_direct_%j.out \
  --wrap="uv run python -m scripts.python.run_layer_batched_tangent_sweep_direct configs/gpt2.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --target-batch-gb 250 \
    --compute-node ruapehu \
    --storage-node mauao \
    --job-time-limit 12:00:00 \
    --knn-k 160 \
    --basis-method dual \
    --n 1000 \
    --radius 0.01 0.05 0.1 0.5 1.0"


MAUAO:

sbatch --job-name=tangent_sweep_direct_driver --partition=ampere --nodelist=mauao --time=12:00:00 \
  --cpus-per-task=1 --mem=2G --output=logs/run_layer_batched_tangent_sweep_direct_%j.out \
  --wrap="uv run python -m scripts.python.run_layer_batched_tangent_sweep_direct configs/tiny_llama.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --target-batch-gb 250 \
    --compute-node mauao \
    --storage-node mauao \
    --job-time-limit 12:00:00 \
    --knn-k 160 \
    --basis-method dual \
    --n 1000 \
    --radius 0.01 0.05 0.1 0.5 1.0"



TRAIN BATCH AE:

sbatch --job-name=layer_batched_driver --partition=hopper --nodelist=ruapehu --time=12:00:00 \
  --cpus-per-task=1 --mem=2G --output=logs/run_layer_batched_training_%j.out \
  --wrap="uv run python -m scripts.python.run_layer_batched_training configs/tiny_llama.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --target-batch-gb 250 \
    --compute-node ruapehu \
    --storage-node mauao \
    --train-time-limit 12:00:00 \
    --n-models-per-chunk 200 \
    --epochs 12 \
    --checkpoint-every 3 \
    --lr 3e-4"


AUTOENCODER SAMPLE:

srun --partition=ampere --nodelist=mauao --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  uv run python -m scripts.python.run_autoencoder_sampled_sweep configs/gpt2.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --out-dir onnx_models/gpt2/ae_sampled_sweep \
    --n 1000 \
    --alpha 0.3 1.5 3.0\
    --tau 1.0 \
    --noise 0.0 0.5 1.0

sbatch version:

sbatch scripts/batch_scripts/run_sampled_sweep_gpt2_mauao_sbatch.sh


PRODUCE STATS:

sbatch scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_mauao.sh configs/simple_stories.toml

analyze xi
uv run python scripts/analysis/print_xi_table.py onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4


check different fits:
srun --partition=ampere --nodelist=mauao --cpus-per-task=4 --mem=16G --time=00:30:00   uv run python -m scripts.analysis.compare_gpd_estimators     onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4     --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep     --sample-idx 5 --pos 7 --n-tail 3000



sbatch scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh \
    configs/tiny_llama.toml --target-batch-gb 250 --compute-node ruapehu --storage-node mauao



sbatch scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh \
    configs/simple_stories.toml --compute-node mauao --storage-node mauao


KDE fit over a sample's ratio histogram (plots fit + prints density at g=0):
uv run python -m scripts.analysis.plot_kde_fit_density_at_zero \
    onnx_models/simple_stories/lipschitz_ratio_bootstrap_eqd_sweep_gpu/embedding.add_4 \
    --sample-idx 10 --pos 0 --out kde_fit.png


PERTURBED JACOBIAN DET (scripts/python/compute_perturbed_jacobian_det.py):

For every perturbed-sweep row (sample_idx, pos, vocab_id), computes the
d_model x d_model diagonal Jacobian block d out[pos]/d in[pos] of each sublayer
at that perturbed sequence (ONNX pullback, built at the forward subgraph's fixed
batch, e.g. 32000 -- not the config's input_shape batch) and saves
row_idx, sample_idx, pos, vocab_id, sign, log_abs_det (det = sign * exp(log_abs_det))
per sublayer under --out-dir/{in}.{out}. Skips input_ids.embedding. Checks each
recomputed output against the sweep's stored vec (--verify-atol). Key flags:
--perturbed-sweep-dir (default onnx_models/simple_stories/perturbed_sweep),
--out-dir (project-relative, default onnx_models/simple_stories/perturbed_jacobian_det),
--sublayers, --samples (only rows with these sample_idx; row_idx stays global; use a
separate --out-dir), --points-per-call (default fixed_batch // d_model), --max-rows (smoke
tests only -- use a separate --out-dir), --shard-rows 500000, --flush-every 5, --no-push.
The batch script sets SAMPLES="0" -> --out-dir .../perturbed_jacobian_det_samples_0.
simple_stories' perturbed shards live on mauao's local scratch, so run on mauao.

sbatch scripts/batch_scripts/simple_stories/compute_perturbed_jacobian_det.sh

srun --partition=interactive --nodelist=mauao --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:30:00 \
  uv run python -m scripts.python.compute_perturbed_jacobian_det configs/simple_stories.toml \
    --sublayers embedding.add_4 --max-rows 500 --no-push \
    --out-dir onnx_models/simple_stories/perturbed_jacobian_det_smoke


DET SIGN INTERPOLATION (scripts/python/generate_det_sign_interpolation.py):

Reads compute_perturbed_jacobian_det.py's per-row (row_idx, sign, log_abs_det) for
one --sublayer, picks two perturbed points A (det > 0) and B (det < 0) with the same
(sample_idx, pos) at random (--seed), or explicitly via --row-a/--row-b, interpolates
the sublayer's full (seq, d_model) input x(t) = (1-t) X_A + t X_B for --n-points t in
[0, 1] (default 1000), runs the ONNX forward subgraph, and saves t, vec_in = x(t)[pos],
vec_out = f(x(t))[pos] (+ endpoint attrs) to
{--out-dir}/{sublayer}/row{A}_row{B}.h5 (default out-dir
onnx_models/simple_stories/det_sign_interpolation, written straight to NFS -- tiny).
Endpoint outputs are checked against the perturbed sweep's stored vecs (--verify-atol).
Works on partial det output (only completed shards are used). Run on mauao.

srun --partition=interactive --nodelist=mauao --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:30:00 \
  uv run python -m scripts.python.generate_det_sign_interpolation configs/simple_stories.toml \
    --sublayer add_4.add_6 --n-points 1000


DET SIGN INTERPOLATION LIPSCHITZ HISTOGRAM (scripts/analysis/plot_det_sign_interpolation_lipschitz.py):

Scatter of ||f(x_i)-f(x_j)|| vs ||x_i-x_j|| over all pairs (with min/max-slope lines),
histogram of g = their ratio, and local ratio vs t, for
generate_det_sign_interpolation.py output(s); PNG next to each h5
({stem}_lipschitz_hist.png). --min-dt (default 0.01) drops pairs closer than that in t
(float32 rounding noise) and sets the local-ratio stride; --bins 100. CPU-only, runs anywhere.

uv run python -m scripts.analysis.plot_det_sign_interpolation_lipschitz \
    onnx_models/simple_stories/det_sign_interpolation/add_4.add_6/row26685_row26140.h5


DET ZERO SPHERE (scripts/python/generate_det_zero_sphere.py + scripts/analysis/plot_det_zero_sphere_lipschitz.py):

Takes a generate_det_sign_interpolation.py h5, computes det J along the A -> B path
(ONNX pullback), bisects the --crossing'th sign change (default 0 = first; indices or 'all';
paths can cross many times) to t*, then samples --n-points (1000) uniformly on the sphere of
radius |x(t*) - x_A| (--radius-to a|b, or a fixed --radius R, which tags the file _r{R}; --ball for the
solid ball, uniform: Gaussian direction, radius R U^(1/d)) around x(t*)[pos],
other positions fixed at x(t*). Saves vec_in/vec_out/sign/log_abs_det for the sphere
points, the centre, det along the path, J's singular values at the centre, and --n-line
(401) points on the lines c + s d, s in [-r, r], along J's smallest singular direction
v_min and along the A -> B path direction, plus 2 x --n-line-log (200) points along v_min
with |s| log-spaced from --line-min-s (1e-5) to r (v_min: to --line-max-s if set, e.g. 10), to
{--out-dir}/{sublayer}/row{A}_row{B}_sphere_c{crossing:02d}.h5, one per crossing (default out-dir
onnx_models/simple_stories/det_zero_sphere). Run on mauao (GPU). The plot script
(CPU, anywhere) draws the all-pairs scatter/histogram, the centre-to-point stretch
by det sign, log|det| along the path, input vs output distance from the centre along
v_min / the path (log-log, with a k s^2 guide), and J's singular-value spectrum.
TF32 is OFF in compute_perturbed_jacobian_det._session (shared by the det /
interpolation / sphere scripts): ORT's CUDA EP defaults to TF32 on A40/H100, which adds
~1e-3 relative output noise and ~5e-3 error to Jacobian singular values.

srun --partition=interactive --nodelist=mauao --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:20:00 \
  uv run python -m scripts.python.generate_det_zero_sphere configs/simple_stories.toml \
    onnx_models/simple_stories/det_sign_interpolation/add_4.add_6/row26685_row26140.h5
uv run python -m scripts.analysis.plot_det_zero_sphere_lipschitz \
    onnx_models/simple_stories/det_zero_sphere/add_4.add_6/row26685_row26140_sphere_c00.h5


DET ZERO COMPOUND PLOT (scripts/analysis/plot_det_zero_sphere_compound.py):

v_min (critical direction) vs random points in the ball around c (uniform in the ball of radius r,
plus random-direction points at the v_min line's log-spaced radii; sampled by
generate_det_zero_sphere.py on the same uniform + log-spaced s grids) on shared axes: log-log
scatter of ||f(x)-f(c)|| vs ||x-c||, and linear histogram of the pairwise Lipschitz ratio
along each uniform line; plus v_min with noise (c + s v_min + eps |s| u, u random orthogonal
to v_min; generator --noise-levels, default 1e-3 1e-2 1e-1) against dotted linear-leak lines,
and a log10 ||f(x)-f(c)|| heatmap over the (v_min, v_2) plane of the two smallest singular
directions (generator --n-plane 101, --plane-extent default r) with the det J = 0 contour;
plus a log-log histogram of the centre-to-point ratio ||f(x)-f(c)||/||x-c|| over all points
(incl. log-spaced), split into above-floor (solid) / float32-floor (dashed) points. The float32
output floor is estimated from the rounding-dominated v_min points and the band below 2x it
is shaded grey on the log-log scatters. Writes {stem}_compound.png. CPU, anywhere; needs a sphere h5 made
after the control line was added (errors otherwise).

uv run python -m scripts.analysis.plot_det_zero_sphere_compound \
    onnx_models/simple_stories/det_zero_sphere/add_4.add_6/row26685_row26140_sphere_c00.h5


DET ZERO MANIFOLD (scripts/python/generate_det_zero_manifold.py):

Same input and crossing bisection as generate_det_zero_sphere.py (--crossing, default 0, or 'all'),
but the det = 0 point c is then projected onto the real activation manifold, and that point p
is the base for sampling. The manifold is A's (sample_idx, pos) `in`-layer cloud from the
perturbed sweep (load_sample_cloud; A and B are checked to be points of it). No autoencoder:
projection and checks use gmanifold's model-free validators (TangentCharts, KernelScore), ported
in. Charts: local PCA, m = --m (default TwoNN intrinsic dim), K = max(4m, 32); p = nearest chart
point over the --n-cand (16) nearest anchors, re-anchored until stable, in-chart step capped at
--chart-reach (2) spacings. Around p, radius --radius or --radius-units (1) x p's anchor spacing:
A = convex neighbour sampling (p + r unit(sum_j w_j (x_j - p)), w ~ Dirichlet over p's --n-nbrs (10)
nearest real points), B = ambient d_model-ball,
C = line along v_min of J(p); around the critical point c itself, same radius: cB = ambient
d_model-ball, cC = line along v_min of J(c) (the kernel direction); plus *_log sets of all five at
log-spaced radii from --line-min-s (1e-5). --kernel gauss (default) | laplace for the kernel score.
Per point: vec_in/vec_out/sign/log_abs_det, dist to p, chart residual, kernel score u; plus
calibration bands (held-out real, +0.5x/1x spacing noise; kernel = --kernel, default gauss, or the
paper's regularized Laplace e^{-sqrt(|x|^2/sigma^2 + --kernel-r)}), J's spectrum and v_min at p and c,
and ||U^T v_min||^2 (kernel-in-tangent overlap). Gotcha: local-PCA tangent directions
score like ambient ones on these clouds (flat local spectrum at ~4k points in 384-d), which is why
A samples into the hull of real neighbours instead; u >= ~0.93 (held-out 5th pct) counts as on the
cloud -- u > 1 just means between real points.
--log-radii LO HI draws the uniform sets' radii (|s| on the lines) log-uniformly in [LO, HI].
Never overwrites: refuses up front if an output exists unless --overwrite; use --tag (file suffix) or a
new --out-dir per run. Writes {--out-dir}/{sublayer}/row{A}_row{B}_manifold[_r{R}][_{tag}]_c{crossing:02d}.h5 (default out-dir
onnx_models/simple_stories/det_zero_manifold). Run on mauao (GPU, ~1 min per crossing).

srun --partition=interactive --nodelist=mauao --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:25:00 \
  uv run python -m scripts.python.generate_det_zero_manifold configs/simple_stories.toml \
    onnx_models/simple_stories/det_sign_interpolation/add_4.add_6/row26685_row26140.h5 --crossing all

Plots (scripts/analysis/plot_det_zero_manifold.py, CPU, anywhere): one {stem}.png per crossing
(7 panels; overlaid histograms each scaled to peak 1: kernel score, residual vs u, ||f(x)-f(centre)|| vs
||x-centre||, per-point K_L value ||f(x)-f(centre)||/||x-centre|| vs ||x-centre|| with the float32 floor,
stretch ratio,
log|det| vs distance from the centre, K_L histogram = pairwise ratio g over all pairs within each set
with each set's min marked, output changes < 2e-6 dropped as float32 noise) and {pair}_manifold_overview.png across crossings (det along the path,
|c-p|, u at c/p, sigma_min at c vs p, kernel-in-tangent overlap, distance to the first det sign flip),
and a separate {pair}_manifold_p_validation.png: u(p) of every crossing against the held-out real
points' u distribution (histogram + percentile of each p), u(c) shown for contrast.
Also {pair}_manifold_kernel_score.png: u of held-out real points vs critical points c vs nearest
manifold points p, as one strip plot (kernel named in the title).

uv run python -m scripts.analysis.plot_det_zero_manifold \
    onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c*.h5


DET ZERO MANIFOLD EVT (scripts/analysis/plot_det_zero_manifold_evt.py):

The paper EVT figure (plot_lipschitz_evt_paper.draw_evt_row: (a) pairwise distance scatter, (b) ratio
histogram, (c) lower tail + GPD fit with K_L-hat) for a generate_det_zero_manifold.py h5, one row per
sample set (--sets, default A B C cB cC: p on manifold / p ambient / p v_min line / c ambient / c kernel
line). All pairs within each uniform set; pairs with output change < --fp32-floor (2e-6) dropped as
float32 noise. Fit = the sweep's own bootstrap-EQD threshold + GPD at its defaults, on --n-pairs (200k)
random pairs. CPU (~20 s per file). Writes {stem}_evt.pdf / .png next to each input.

uv run python -m scripts.analysis.plot_det_zero_manifold_evt \
    onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c*.h5


DET ZERO MANIFOLD PAPER FIGURES (scripts/analysis/plot_det_zero_manifold_paper.py):

ICLR (5.5in, PAPER_RC) versions for one generate_det_zero_manifold.py h5: {--out}_main.pdf/png (panels
(a)-(g) of the per-crossing figure + legend slot, colours matching the other *_paper.py figures: kernel
v_min red, ambient grey, on-manifold blue; centre p solid/circles, c dashed/triangles) and {--out}_evt.pdf/png
(draw_evt_row per set). Every panel except (c), (d) and the EVT figure uses only samples with distance from
their centre in --dist-range (1e-6 1e1); (c), (d) span --wide-range (1e-8 1e2). Histograms: 1st-99th pooled
percentile x-range (--hist-q), peak-normalised log heights. Default --out figures/paper/det_zero_manifold_c{NN}.
Prints the caption numbers (|c - p|, sigma_min at c and p).

uv run python -m scripts.analysis.plot_det_zero_manifold_paper \
    onnx_models/simple_stories/det_zero_manifold_minS1e-12/add_4.add_6/row26685_row26140_manifold_r100_c05.h5


PRESENTATION VERSIONS OF EVERY PAPER FIGURE: set FIG_MODE=presentation in the env of any *_paper.py /
plot_det_zero_{kernel_score,output_distance,pairwise_distance,kl_value,kernel_distance,evt}.py run -- writes
only a 600 dpi PNG (tight bbox), legends 1.6x larger and at most 2 columns, to figures/presentation/ instead of
figures/paper/ (paper PDFs untouched). Helpers: legend_fs / legend_ncol / save_figure in
plot_kernel_tangent_overlap_paper.py. transformer_blocks falls back to alternating attn/FFN when a config has
no exported sub_blocks ONNX (GPT-2: layer 1 = add.add_2 / add_2.add_5, 6 = add_21.add_22 / add_22.add_25,
12 = add_45.add_46 / add_46.add_49).

FIG_MODE=presentation uv run python -m scripts.analysis.plot_det_zero_kl_value
FIG_MODE=presentation uv run python -m scripts.analysis.plot_det_zero_manifold_paper \
    onnx_models/gpt2/det_zero_manifold_paper/add_2.add_5/row182959_row176262_manifold_r1000_logr_c04.h5 \
    --dist-range 1e-5 1e2 --wide-range 1e-8 1e3 --out figures/paper/det_zero_manifold_gpt2_c04

DET ZERO STANDALONE PAPER FIGURES (scripts/analysis/plot_det_zero_{kernel_score,output_distance,pairwise_distance,
kl_value,kernel_distance,evt}.py): no flags -- each loops over a RUNS table at its top (simple_stories add_4.add_6 crossing 5,
GPT-2 add_2.add_5 crossing 4) and writes figures/paper/det_zero_<name>_c05 and _gpt2_c04 (.pdf/.png).
kl_value's RUNS row also sets the float32-floor window and the far-field line per model (GPT-2's outputs
are ~60x larger, so its floor is ~5e-5 vs ~1e-6). GPT-2 pipeline notes: its perturbed sweep lives only on
mauao scratch; the sweep was made with TF32 on, so pass --verify-atol 0.25 to compute_perturbed_jacobian_det
and generate_det_sign_interpolation (TF32-off recompute differs by ~3e-4 relative, ~0.05 absolute).


TOY SINGULAR IDENTITY (scripts/analysis/toy_singular_identity.py):

Known-answer toy for the ratio pipeline in high dimension: f: R^d -> R^d (dim 386) is the
identity except f(x)_0 = x_0^2, so det J = 2 x_0 and f is singular on the hyperplane x_0 = 0;
true K_L = min(1, inf |x_0 + y_0|). Three x_0 sampling regimes: one_side (one_side_range,
default [-1, -0.001]), near (both sides, |x_0| < near_width 0.1, K_L = 0), far (both sides,
|x_0| in far_range [0.5, 1.0], K_L = 0); coords 1..d-1 ~ U(-rest_scale, rest_scale). Each regime
runs the GPU pipeline's pairs -> bootstrap-EQD -> GPD/c_hat (same defaults as
lipschitz_ratio_bootstrap_eqd_sweep_gpu.py) and gets one figure row: x_0 domain vs image, ratio
histogram, lower tail vs true K_L / c_hat, ||x-y|| vs ||f(x)-f(y)|| scatter.
ALL params come from one positional YAML config (keys = the script's DEFAULTS dict; missing
keys keep their default, unknown keys are an error). Ready-made configs in
configs/toy_singular_identity/: isotropic.yaml (every key spelled out -- use as a template;
rest_scale 1), thin.yaml (rest_scale 0.01, regimes separate), near_very_thin.yaml,
far_fold.yaml, one_side_close.yaml, c_hat_zero.yaml (c_hat -> ~0: needs pairs mirrored
across x_0 = 0 AND rest_scale small enough that c_hat ~ 16 rest_scale / (2|x_0|) is tiny), quick.yaml (CPU smoke test). Each writes to its own
figures/toy_singular_identity/<name>.png (out key; the directory is created).
Gotcha: at rest_scale 1 the 385 identity dims dominate every pair distance and g sits just
under 1 in all regimes (the singularity is invisible to random pairs). ~5-15 s per config on
CPU, runs anywhere.

uv run python -m scripts.analysis.toy_singular_identity configs/toy_singular_identity/isotropic.yaml
uv run python -m scripts.analysis.toy_singular_identity configs/toy_singular_identity/thin.yaml


KERNEL vs TANGENT SPACE OVERLAP (scripts/python/compute_kernel_tangent_overlap.py + scripts/analysis/plot_kernel_tangent_overlap.py):

At every real perturbed-sweep point (sample, pos, vocab_id), per sublayer: the pos-diagonal
Jacobian block J = d out[pos]/d in[pos] (same ONNX pullback as compute_perturbed_jacobian_det,
TF32 off), its singular values and the --n-bottom (3) smallest + largest right (v) / left (u)
singular vectors (float64 eigh of J^T J / J J^T), and the local k-NN PCA (k = d_model + 1, same
neighbourhoods as generate_tangent_bases.py, recomputed so the eigenvalues are kept) of the `in`
and `out` clouds. Saves sigma, in_evals, out_evals, and v_coef2/u_coef2 = squared coefficients of
those singular vectors in the PCA basis, so ||P_{T_d} v||^2 = cumsum(...)[d-1] for any d.
--out-dir project-relative; --samples (default 0); --sublayers; --rows-per-block 4096;
--flush-every 2; --no-push; --verify-atol 1e-3 + --verify-rtol 1e-2 * max|stored row| (recomputed in[pos]/out[pos] vs the sweep's vecs;
the sweep was made with TF32 on, so the stored vecs carry relative noise growing with depth, ~4e-3 by add_30).
~590 MB/sublayer/sample. Needs an A40/H100 (the ORT CUDA build has no sm_75 kernels, so not
ngongotaha) and the perturbed sweep shards (mauao scratch).
Batch (all sublayers, sample 0, then both plots into figures/): 
sbatch scripts/batch_scripts/simple_stories/compute_kernel_tangent_overlap.sh
The plot script (CPU) picks d per point by --var-frac (0.95) of in_evals (--side in, kernel vs
T_pM) or out_evals (--side out, cokernel vs T_f(p)M) and writes {data_dir}/kernel_tangent_overlap_{side}_hist.png
(per-sublayer histograms: v_min, v_max, random-direction Beta(d/2,(D-d)/2) baseline) and
_profile.png (median/IQR of ||P_{T_d} v_min||^2 vs d against d/D), plus a summary table.

srun --partition=interactive --nodelist=mauao --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
  uv run python -m scripts.python.compute_kernel_tangent_overlap configs/simple_stories.toml \
    --samples 0 --out-dir onnx_models/simple_stories/kernel_tangent_overlap_samples_0
uv run python -m scripts.analysis.plot_kernel_tangent_overlap \
    onnx_models/simple_stories/kernel_tangent_overlap_samples_0 --side in

Paper (ICLR 5.5in) version of the histogram figure (scripts/analysis/plot_kernel_tangent_overlap_paper.py):
3 x 2 grid, rows = --layers (1-based, default 1 3 6), cols = Attention | FFN (classified from each
sublayer's ONNX subgraph: Softmax = attention; simple_stories alternates attn/FFN starting at
embedding.add_4). v_min red #f0614a solid, v_max green #00663a dashed (CVD-validated pair), random
grey; serif/STIX fonts, TrueType-embedded PDF + 300dpi PNG to figures/paper/kernel_tangent_overlap_{side}.
Needs the data shards locally, so run on mauao.

srun --partition=interactive --nodelist=mauao --cpus-per-task=4 --mem=16G --time=00:15:00 \
  uv run python -m scripts.analysis.plot_kernel_tangent_overlap_paper configs/simple_stories.toml \
    onnx_models/simple_stories/kernel_tangent_overlap_samples_0 --side in


EVT LOWER-LIPSCHITZ PAPER FIGURE (scripts/analysis/plot_lipschitz_evt_paper.py):

Full-page ICLR figure, rows = --layers (default 1 3 6) x {Attention, FFN} (same sublayers as the
kernel figure, via plot_kernel_tangent_overlap_paper.transformer_blocks), cols = (a) ||x-y|| vs
||f(x)-f(y)|| scatter with tail pairs (g < u) highlighted and the slope-K_L-hat line, (b) log-count
ratio histogram with u and K_L-hat, (c) log-density lower tail vs the fitted GPD density (finite
endpoint K_L-hat = u + sigma/xi). Fit (u, xi, sigma, c_hat) read from
lipschitz_ratio_bootstrap_eqd_sweep_gpu output; one (--sample-idx, --pos) for every row (default 47 2,
valid and mid-percentile in all six default sublayers -- errors if a row has no finite positive
bound). Recomputes ALL ~8M pairs of that position's 4019-point cloud from the perturbed sweep
(the fit used 200k, saved 10k), prints all-pairs vs fit tail fraction / min g, and rings any
pair with g < K_L-hat (sample 47 pos 2: 3 such pairs in add_4.add_6). Run on mauao.
--outlier-q (1e-5): display-only low-end cut for panels (b), (c) -- drops isolated stragglers
(incl. those 3 pairs) so they don't stretch the x-axis; (a) and the fit use every pair.
--max-points N: take all pairs over a random N-point subset of each position's cloud (GPT-2's ~50k-point
clouds would be ~1.3e9 pairs; use 4019, simple_stories' size). GPT-2 fits for layers 1/6/12 (samples 0-7):
sbatch scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_mauao.sh configs/gpt2.toml --n-samples 8 \
    --sublayers add.add_2 add_2.add_5 add_21.add_22 add_22.add_25 add_45.add_46 add_46.add_49
Gotcha: right after editing a script on the login node, run it on mauao with a fresh
PYTHONPYCACHEPREFIX -- NFS attribute caching let mauao reuse a stale .pyc once.

srun --partition=interactive --nodelist=mauao --cpus-per-task=8 --mem=32G --time=00:20:00 \
  uv run python -m scripts.analysis.plot_lipschitz_evt_paper configs/simple_stories.toml

EVT TOY-MODEL PAPER FIGURE (scripts/analysis/plot_lipschitz_evt_toy_paper.py):

Same three panels as plot_lipschitz_evt_paper.py (shared draw_evt_row/finish_evt_figure), one row per
dimension N (--dims, default 384 only -- the paper figure; --dims 2 16 64 384 for the full progression) for an analytical map from proofs/dim_analysis/evt_injectivity.py
(--f square|fold|fold_xy, --sampling cube|radius|volume; defaults square + cube). True K_L = 0 for every map,
marked in (b). Fit = the real sweep's pipeline at its defaults (--n-pairs 200k, bootstrap-EQD over n_tail
200..1000 step 200, --eqd-n-boot 200, GPD + K_L-hat = u + sigma/xi). Shows concentration of measure:
square/cube gives K_L-hat 0.003 at N=2, no finite bound at N=16, 0.910 at N=64, 0.991 at N=384.
CPU is fine (~1 min); writes figures/paper/lipschitz_evt_toy_{f}_{sampling}.{pdf,png}.

uv run python -m scripts.analysis.plot_lipschitz_evt_toy_paper
uv run python -m scripts.analysis.plot_lipschitz_evt_toy_paper --f fold_xy --sampling radius


KV-CACHE PERTURBED SWEEP (scripts/helper/generate_perturbed_sweep_kvcache.py) -- same output format as
generate_perturbed_sweep.py, but runs each (sample, pos) prefix once and the vocab candidates as batched
length-1 continuations. --boundary is attn_out | mlp_out | embed (--layer-idx ignored for embed). Needs
`block_prefix` in the config toml. GPU only; --out-dir is project-relative; pushes via the NFS relay.

sbatch --job-name=kv_sweep_gpt2 --partition=ampere --nodelist=mauao --time=04:00:00 \
  --output=logs/kv_sweep_gpt2_%j.out \
  --wrap="uv run python -m scripts.helper.generate_perturbed_sweep_kvcache configs/gpt2.toml \
    --layer-idx 1 --boundary attn_out --base-seqs-h5 data/gpt2_sequences \
    --out-dir onnx_models/gpt2/perturbed_sweep_kvcache --n-samples 64 --candidate-batch 2048"

KV-CACHE TESTS (tests/kv_tests, plan + status in tests/kv_tests/TESTS.md). GPU node required:

srun --partition=interactive --nodelist=mauao --gres=gpu:a40:1 --time=01:00:00 --cpus-per-task=8 --mem=64G \
  uv run pytest tests/kv_tests -q
