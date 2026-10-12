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


sbatch scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh \
    configs/tiny_llama.toml --target-batch-gb 250 --compute-node ruapehu --storage-node mauao



sbatch scripts/batch_scripts/run_lipschitz_ratio_bootstrap_eqd_sweep_gpu_sbatch.sh \
    configs/simple_stories.toml --compute-node mauao --storage-node mauao
