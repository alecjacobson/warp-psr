# GPU baseline experiment support

The [comparison](../../gpu-baselines.md) explains settings, scope and results.
Main runners are `tools.benchmark_gpu_baseline`, `tools.benchmark_cached_fft`,
`tools.compare_relaxation_diagonal` and `tools.plot_gpu_baselines`.

- `external-requirements.txt` records the PyTorch environment used for timing.
  Install PyTorch 2.5.1 from its CUDA 12.4 wheel index and torch_scatter from the
  corresponding PyG wheel index. No training dependencies/checkpoints are needed.
- `build_neural_galerkin.py` builds the pinned `pth` checkout in that environment,
  using the specified CUDA toolkit and a disposable build cache. The only source
  edit is the enum rename in `neural-galerkin-cuda12.patch`. The measured build
  used GCC 11.4, CUDA compiler 12.8, CUDA 12.4 PyTorch libraries, and SM 8.9.
- `coarse_graph.py` and `run_coarse_graph.py` reproduce the rejected coarse-graph
  experiment. The candidate reuses production kernels and keeps sweep order.
  Run from the repository with its Warp environment and explicit `--fixture`,
  `--depth` and `--output` arguments.
- `validate_cached_fft.py` repeats the full eagle field equality check. It expects
  the pinned SAP source and eagle fixture under `.build/gpu-baselines/` as in
  the experiment. Run it in the PyTorch environment.
- `warp-drill-profile.*` is an isolated **instrumented** profile, useful for call
  counts rather than uninstrumented timing. It supports follow-up issue #94.

Example setup in a separate virtual environment:

```sh
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install 'numpy==1.26.4' scipy==1.15.3 omegaconf ninja scikit-image
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python docs/results/gpu-baseline-scripts/build_neural_galerkin.py \
  --source /path/to/neural-galerkin --cuda-home /path/to/cuda-12.8 \
  --cache /path/to/disposable-build-cache --arch 8.9
```

The publication renders additionally use PyVista 0.46.4, VTK 9.5.2, Matplotlib
3.10.9, NumPy 1.26.4 and SciPy 1.15.3. Install rendering packages after timings,
so their setup does not contend with measured work. Inputs come from the existing
[dataset catalog](../../datasets.md); fixture hashes and point counts are in the
report. The timing scheduler reused `../depth9-audit-scripts/isolation.py`.
