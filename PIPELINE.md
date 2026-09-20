
## Scripts
Genereate base sequences V
    All samples

Generate base activation

Generate perturbed sweep V
    All samples

train batched manifold V


run sampled sweep V

generate tanged bases


generate tanged manifold distances



## Plan Code

### Must haves:



> Plot things that will go into paper

> Strategy depth first approach  
> Get everything sorted for simple stories then for gpt then for llama

1. Get all of data that we need

   - [x] Valid points

   - [ ] Autoencoder points
     - Run samples sweep on all three models, ideally all samples
        ```
        run_autoencoder_sampled_sweep.py now supports both requests: --alpha is nargs="+" (was a single float) — e.g. --alpha 0.3 1.5 sweeps both, each written to its own {src}.{dst}.alpha{a}.h5. --noise is a new nargs="*" flag — e.g. --noise 0.5 1.0 writes calibration variants ({src}.{dst}.noise{level}.h5): real fitted states displaced by Gaussian noise scaled by each anchor's own local ambient spacing, matching the "real + 0.5x/1x noise" bands from plot_autoencoder_sample_diagnostics.py. Omitting --noise writes nothing extra (unchanged default behavior).
        ```



   - [x] Autoencoder verification like what Jakub showed: WIP
      - sbatch scripts/batch_scripts/plot_autoencoder_sample_diagnostics_mauao.sh configs/s.toml <src>.<dst> <pos> [ae_checkpoints_dir]

   - [ ] Tangent points
     - Run samples sweep on all three models, ideally all samples

   - [ ] Ambient points
     - Run samples sweep on all three models, ideally all samples

2. Analysis for each of the above
    - [ ] Get KL values
    - [ ] Plot KL values
    - [ ] Find thresholds and plot


### Could haves:
    KV caching to get longer sequences




## Plan Paper