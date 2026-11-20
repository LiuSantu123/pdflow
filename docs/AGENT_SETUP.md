# Agent setup guide

Goal: go from a machine with pre-installed design/prediction tools to a working
`pdflow` configuration without editing Python code. Everything is driven by
`backends.yaml`; `pdflow doctor` is the acceptance test.

## 0. Preconditions

- Linux or WSL, NVIDIA GPU with enough memory for the predictors you enable.
- `conda` or `mamba` available.
- US-align binary available.
- The design/prediction tools already installed in **separate** environments.
  `pdflow` does not install or modify them.

## 1. Install the orchestrator

```bash
# native conda profile
bash scripts/bootstrap.sh

# Blackwell acceleration profile
bash scripts/bootstrap.sh -- --profile kit
```

The script creates the `pdflow` conda env from `environment.yml`, installs the
package editable, and runs `pdflow setup`. Manual equivalent:

```bash
conda create -n pdflow python=3.12 -y
conda run -n pdflow pip install -e /path/to/pdflow
conda run -n pdflow pdflow setup --profile native
```

`pdflow setup` writes `backends.yaml` with detected `CONDA_BASE`, `MODEL_ROOT`,
`KIT_ROOT`, `AF3_ROOT` and `USALIGN`. Use `--dest` to write elsewhere,
`--no-input` for CI, and `--force` to replace an existing config.

## 2. Discover paths for each backend

Run these once and paste the results into `backends.yaml`.

```bash
conda env list
```

For each environment:

```bash
E=<env-name>
ls "$(conda info --base)/envs/$E/bin" | grep -Ei 'af3|run_alphafold|boltz|protenix|opendde|^mol$|rfd4|esmfold|tmol'
```

### AlphaFold 3

```bash
python - <<'PY'
import alphafold3, pathlib
print(pathlib.Path(alphafold3.__file__).parent)
PY
```

- `script`: the repository's `run_alphafold.py` (and `run_alphafold_5070ti.py`
  on RTX 50-series cards).
- `model_dir`: directory containing `af3.bin`.
- `db_dir`: leave empty when every chain gets a user-supplied MSA. Otherwise
  point it at the AF3 sequence databases.

### Boltz-2

```bash
ls ~/.boltz 2>/dev/null || echo "$BOLTZ_CACHE"
```

`cache` must contain `boltz2_conf.ckpt` and `boltz2_aff.ckpt`.

### Protenix / OpenDDE / Mol-1

These share the AF3-like JSON dialect.

```bash
# Protenix checkpoint root (contains checkpoint/protenix-v2.pt and common/)
echo "$PROTENIX_ROOT_DIR"
# OpenDDE runtime (contains checkpoint/opendde.pt and common/components.cif)
echo "$OPENDDE_ROOT_DIR"
# Mol-1 root (contains checkpoint/mol-1-base.pt and common/components.cif)
echo "$MOL_ROOT_DIR"
```

On RTX 50-series (sm_120), do **not** use the stock Protenix conda env: its
`torch 2.7.1+cu128` / `cuequivariance-cu12` stack produces NaN. Build the
Blackwell kit stack in a separate venv and set `LAYERNORM_TYPE=torch`:

```bash
uv venv --seed --python 3.11 ~/.venvs/ptxv2kit
~/.venvs/ptxv2kit/bin/pip install --no-deps -r <kit>/environment/requirements.lock
~/.venvs/ptxv2kit/bin/pip install --no-deps <kit>/stock/protenix-2.0.0-py3-none-any.whl
PATH=~/.venvs/ptxv2kit/bin:$PATH bash <kit>/run.sh install
# kalign 3.x for template realignment (any static build works)
cp /path/to/kalign ~/.venvs/ptxv2kit/bin/kalign
```

`<kit>` is the `protenix_v2` directory under the Blackwell acceleration kit.
Put `protenix-v2.pt` at `<PROTENIX_ROOT_DIR>/checkpoint/protenix-v2.pt`.
Then in `backends.yaml` point `protenix2.binary` at the venv's `protenix`,
set `clean_ld_library_path: true`, add
`extra_args: ["--need_atom_confidence", "True"]` (required for PAE),
`path_prepend: ~/.venvs/ptxv2kit/bin` (for `kalign`), and add
`LAYERNORM_TYPE: torch` plus `PROTENIX_ROOT_DIR` under `env:`.

Protenix's full-atom confidence output (`*_full_data_sample_*.json`) contains
`token_pair_pae`; without `--need_atom_confidence True` only summary pTM/iPTM/
pLDDT are written and `mean_pae`/`ipae`/`ipsae_ref` stay NaN.

For the kit's accelerated path, add a `launcher` instead of calling the stock
binary directly:

```yaml
launcher:
  - bash
  - <kit>/run.sh
  - pred
  - --config
  - sm120
  - --mode
  - fast
path_prepend:
  - ~/.venvs/ptxv2kit/bin
  - /path/to/nvcc/bin
env:
  PROTENIX_ROOT_DIR: /path/to/weights
  LAYERNORM_TYPE: torch
  MODEL_OPT_LEVERS_OFF: ln_core
  MODEL_OPT_JIT_ROOT: ~/.cache/ptxv2kit-jit
  CUDA_HOME: /path/to/cuda
  CUDACXX: /path/to/nvcc
  CPATH: <venv>/lib/python3.11/site-packages/nvidia/cu13/include:/path/to/cuda/include
```

On the reference machine, `exact` is 1.50x and `fast` is 1.55x faster than stock
on a 245-token example, with 0.03-0.04 A RMSD difference. Use `--mode exact` if
you want the closest-to-stock numerics.

### Boltz-2 acceleration

Build a dedicated pinned venv (the stock `boltz2` env uses torch 2.11 and the
kit's exact arm refuses it):

```bash
uv venv --seed --python /path/to/python3.12 ~/.venvs/boltz2kit
grep -v -E '^(#|boltz==)' <kit>/environment/requirements.lock > /tmp/boltz2stack.txt
~/.venvs/boltz2kit/bin/pip install --no-deps -r /tmp/boltz2stack.txt
~/.venvs/boltz2kit/bin/pip install --no-deps <kit>/stock/boltz-2.2.1-py3-none-any.whl
PATH=~/.venvs/boltz2kit/bin:$PATH bash <kit>/run.sh install
```

Wire it with `launcher: [bash, <kit>/run.sh, pred, --config, sm120, --mode, fast, --allow-partial]`,
`path_prepend: ~/.venvs/boltz2kit/bin`, `BOLTZ_CACHE`, and
`MODEL_OPT_JIT_ROOT`. The kit can exit 3 when the `exactln` lever cannot apply
even though the prediction is complete; `pdflow` accepts that with
`accept_returncodes: [0, 3]` and records `partial: true`.

### OpenDDE acceleration

The stock `opendde` env already matches the kit pin. Add the kit launcher
(`--config sm120 --mode fast`) and:

```yaml
env:
  TORCH_CUDA_ARCH_LIST: "12.0"   # host exports an arch list nvcc 13.1 rejects
  CUDA_HOME: /path/to/cuda
  CUDACXX: /path/to/nvcc
  OPENDDE_ROOT_DIR: /path/to/runtime
query_msa_fallback: true          # kit gate requires an explicit MSA
```

### ESMFold2 overlay

`esmfold2_overlay` runs ESMFold2 + ESMC-6B through the RTX 50 series launcher
in the [Blackwell acceleration kit](https://github.com/LiuSantu123/claude-blackwell-protein-accel-kit/tree/v0.2.0-sm120).
Point `repo` at the checkout that contains
`scripts/esmfold2_50.sh`, `python` at a compatible
[Biohub ESM environment](https://github.com/Biohub/esm), and both weight
variables at local checkpoint directories.

```yaml
predictors:
  esmfold2_overlay:
    enabled: true
    repo: /path/to/acceleration-kit
    python: /path/to/esmfold2-env/bin/python
    mode: blackwell                  # also accepts tiled or stock
    env:
      ESMFOLD2_WEIGHTS: /path/to/ESMFold2
      ESMFOLD2_ESMC: /path/to/ESMC-6B
```

For a campaign with one single-chain protein per FASTA file under
`sequences/`, include `esmfold2_overlay` in `score.predictors`.

```bash
pdflow doctor --predictor esmfold2_overlay
pdflow predict --campaign runs/monomer --predictors esmfold2_overlay
pdflow score --campaign runs/monomer
pdflow finalize --campaign runs/monomer
```

Each prediction call loads the model once. A supplied A3M must have the same
query sequence; a query-only A3M is handled as a single-sequence input. Each
batch writes its CIFs under `structures/`, raw PAE and pLDDT under
`confidences/`, and `summary.json`, `ranking.csv`, and `index.json` at
`predictions/_batches/esmfold2_overlay/batch-*/output/`. Finalization keeps this
batch output. Per-design failures appear in the prediction status and scores.

The legacy `esmfold2` configuration continues to run ESMFold v1.

### ESMFold

- `python`: the fair-esm environment.
- `script`: `run_esmfold.py` (the local wrapper that loads fair-esm weights).
- `pythonpath`: the OpenFold checkout that `run_esmfold.py` imports.
- `weights`: `esmfold_3B_v1.pt`.
- Keep `extra_args: ["--fp32"]` unless you have verified fp16 on your build.

### US-align

```bash
find / -maxdepth 5 -name USalign -type f 2>/dev/null | head
```

Test it:

```bash
env -u LD_LIBRARY_PATH /path/to/USalign 2>&1 | head -3
```

If it fails with `GLIBCXX_... not found`, keep
`usalign.clean_ld_library_path: true`.

### tmol

```bash
conda run -n tmol python -c "import tmol; print(tmol.__version__)"
```

`tmol.python` must point to the environment's interpreter, not the `tmol`
console script. pdflow 1.x uses only the top-level API
(`pose_stack_from_file`, `beta2016_score_function`, `fast_relax`).

## 3. Configure RFdiffusion4

```bash
which rfd4-generate
```

Set `design.rfd4.binary`. The command template is:

```text
{binary} inputs={input} num_replicates={num_designs} out_dir={out_dir} {extra}
```

- `design.input` is a condition-spec YAML/JSON or an annotated CIF, resolved
  relative to the campaign directory.
- For a first smoke test, use `extra: ["inference_experiment=quickstart"]`.
- For normal runs, use `extra: ["~metrics"]` so RFD4 does not refold.
- For a **native checkpoint**, set these in `default_extra`:

  ```yaml
  default_extra:
    - hf_ckpt_path=null
    - ckpt_path=/path/to/flow_checkpoints
    - ckpt_name=your-flow.ckpt
    - autoencoder_ckpt_path=/path/to/ae_checkpoints/your-ae.ckpt
  ```

  For a **local HF bundle directory**, use `hf_ckpt_path=/path/to/bundle`.
  The reference machine uses `/home/liusantu/hf-bundle` with the
  `rfd4_proteina_flow-ema.ckpt` / `rfd4_proteina_ae-ema.ckpt` pair.

## 4. Acceptance test

```bash
conda run -n pdflow pdflow doctor
```

Expected output ends with `all backends look usable.` Fix every `[FAIL]` line
before continuing.

Then run the no-GPU checks:

```bash
conda run -n pdflow pytest -q
```

## 5. First campaign

```bash
conda run -n pdflow pdflow new --campaign runs/smoke
# copy a final backbone directory
conda run -n pdflow pdflow ingest --campaign runs/smoke --designer rfd4 \
    --input /path/to/final/backbones
conda run -n pdflow pdflow predict --campaign runs/smoke --predictors boltz2
conda run -n pdflow pdflow score --campaign runs/smoke
conda run -n pdflow pdflow finalize --campaign runs/smoke --keep-intermediates
```

Inspect:

- `sequences/all.fasta` — every sequence with `backbone=` provenance.
- `predictions/<seq>/<predictor>/scores.json` — per-model metrics.
- `designs.csv` — flat table.
- `final/<seq>/best.cif` + `full_data.json` — archive.

When the smoke run passes, drop `--keep-intermediates` to keep only the archive.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `predictor exited successfully but produced no structures` | Read `predictions/<seq>/<predictor>/run.log`; the tool returned 0 after skipping. Usually a missing MSA or weight. |
| Boltz: `Missing MSA's in input` | pdflow writes a query-only a3m automatically; check the `msa:` path exists and use `--use_msa_server` only if you have network. |
| AF3: `Protein chain N is missing Templates` | pdflow writes `templates: []`; if this appears, the AF3 build expects a non-empty template list. |
| AF3 OOM | Lower `num_samples` or use `run_alphafold_5070ti.py` plus `--flash_attention_implementation=xla`. |
| Protenix NaN or `no kernel image` | Use the dedicated Blackwell venv (`torch 2.13/cu130`) and `LAYERNORM_TYPE=torch`; the stock `torch 2.7.1+cu128` conda env is not sm_120-safe. |
| Protenix PAE / ipSAE is NaN | Add `"--need_atom_confidence", "True"` to Protenix `extra_args`; the full_data JSON then contains `token_pair_pae`. |
| ESMFold `compute_tm` IndexError | Add `--fp32` to `extra_args`. |
| OpenDDE tries to download `components.cif` | Set `OPENDDE_ROOT_DIR` to a local runtime directory. |
| US-align `GLIBCXX` error | Keep `clean_ld_library_path: true`. |
| tmol import error | Ensure `tmol.python` is the tmol 1.x environment, not a 0.1.x install. |
