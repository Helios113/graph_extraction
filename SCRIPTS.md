

```
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



sbatch --job-name=tangent_bases_driver --partition=hopper --nodelist=ruapehu --time=12:00:00 \
  --cpus-per-task=1 --mem=2G --output=logs/run_layer_batched_tangent_bases_%j.out \
  --wrap="uv run python -m scripts.python.run_layer_batched_tangent_bases configs/gpt2.toml \
    --sample-idx $(seq 0 63 | tr '\n' ' ') \
    --target-batch-gb 250 \
    --compute-node ruapehu \
    --storage-node mauao \
    --job-time-limit 12:00:00 \
    --knn-k 160 \
    --basis-method dual"