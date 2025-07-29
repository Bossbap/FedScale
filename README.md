# Bliss

> **Bliss** is a lightweight, deployment‑ready client selection strategy for **federated learning (FL)** that minimizes *wall‑clock time‑to‑accuracy*.
> **Sourced from and built on \[FedScale]**, keeping the same CLI and runtime so you can run jobs exactly as you would on FedScale.

---

## What is Bliss?

Bliss co‑designs **predictive sampling**, **adaptive local training**, and **server‑side coordination**:

* Before each round, the server gathers **real‑time system features** (CPU, network, battery) and **historical feedback** (past utility, past success) per client, feeds them to a regressor, and **predicts next‑round utility** to guide sampling.
* Selected clients run **adaptive local training** that trades communication for computation, removing the need for post‑hoc system penalties in the utility function.
* We integrate **Oort‑style robustness** (server pacer for round deadlines, utility clipping, outlier removal) so the components reinforce one another.

Bliss lives alongside FedScale so you can use all of FedScale’s datasets, models, and runtime while swapping in Bliss for client selection.

---

## Getting Started

### 1) Install (Conda)

> Bliss provides two environments: **CPU** and **GPU**. Pick one.

```bash
# From the repo root (this project alongside FedScale code)
conda env create -f environment_CPU.yml    # CPU-only setup
# or
conda env create -f environment_GPU.yml    # GPU-enabled setup

# Activate (the YAML sets the environment name, typically "bliss")
conda activate bliss

# Install FedScale (editable) from the checked-out source
pip install -e .
```

> If you plan to use GPUs, install a compatible **NVIDIA CUDA toolkit/driver** for your hardware.

### 2) Install dataset

Run:

```bash
bash benchmark/dataset/download.sh download <name_of_dataset>
```

Inspect `benchmark/dataset/download.sh` to configure the download and to see the available dataset names.


### 3) Run a job (same flow as FedScale)

From the **FedScale repo root**, submit a job with your config:

```bash
python docker/driver.py submit configs/<your_config>.yml
```

* Your `configs/*.yml` file holds runtime args and hyperparameters (e.g., `ps_ip`, `worker_ips`, GPU counts, `batch_size`, `sampling_strategy`, `gradient_policy`, and related hparams).
* To use Bliss sampling, set the sampling strategy to Bliss in your config (see examples in `configs/`).

---

## How it works (non‑adaptive mode)

When `adaptive_training: false` in your config, the execution is:

1. **`docker/driver.py`**

   * Loads config, extracts hardware args (`ps_ip`, `worker_ips`, GPU counts), builds `executor_configs`, launches **executors** on the distributed machines, then launches **`aggregator.py`**.

2. **`aggregator.py` (server)**

   * Creates queues to communicate with executors (a global round queue and per‑client callback queue via `event_monitor()`).
   * Runs the main loop tracking simulated wall‑clock time.
   * Uses **`ClientManager`** (from `client_manager.py`) to:

     * Check client availability per round,
     * Select participants,
     * Track per‑round performance and metadata via **`client_metadata.py`**.
   * Interacts with **`thirdparty/bliss/oort.py`** (Oort implementation) and **`thirdparty/bliss/bliss.py`** (Bliss selection logic).

3. **`executor.py` (clients)**

   * Initializes frameworks (PyTorch/TensorFlow wrappers), partitions data, connects to the server over gRPC.
   * Enters an event loop: sleeps, `client_ping()` when idle, and executes server‑sent events to **update/train/test/shutdown**.
   * For PyTorch runs (`conf.engine == commons.PYTORCH`), creates a **`TorchClient`** from `torch_client.py` which performs the actual local train/test.

> **Adaptive mode:** when `adaptive_training: true`, some responsibilities shift to `adaptive_torch_client.py`. (Out of scope here; see code comments for details.)

---

## Repo Structure (Bliss additions)

```
Repo Root
|---- configs/              # All configuration files used for experiments
|---- notebooks/            # Notebooks for analysis and figure generation
|---- images/               # Figures used in the report/README
|---- run_global_loggings/  # Aggregated logs from evaluation runs

# FedScale runtime (selected)
|---- docker/
|---- fedscale/
|     |---- cloud/          # Aggregator, Executors, Client manager, etc.
|     |---- edge/           
|     |---- utils/
|     |---- dataloaders/
|     |---- utils
|---- thirdparty/bliss/
      |---- bliss.py        # Bliss sampling
      |---- oort.py         # Oort strategy (for robustness mechanisms)
```

---

## Configuration & Examples

* Start from the provided files in `configs/` and customize:

  * **Cluster:** `ps_ip`, `worker_ips`, `num_gpus` per host.
  * **Training:** `dataset`, `model`, `batch_size`, `lr`, `rounds`, `deadline`.
  * **Selection:** `sampling_strategy: bliss` (or oort, random, etc.), related hparams.
  * **Execution:** `engine: pytorch`, `adaptive_training: false` (for the flow above).

Run with:

```bash
python docker/driver.py submit configs/<experiment>.yml
```

Logs and metrics will appear in your run‑specific directories, and global summaries in `run_global_loggings/`.

---

## Results & Artifacts

* **Figures:** see `images/` (generated from the notebooks).
* **Notebooks:** see `notebooks/` for end‑to‑end analysis and plot generation.
* **Logs:** see `run_global_loggings/` for consolidated evaluation logs.

---

## Acknowledgments & Citation

This project is **sourced from FedScale** and integrates with its runtime and datasets. If you use Bliss, please also cite:

* **FedScale** (ICML’22) and **Oort** (OSDI’21) as appropriate for runtime/selection baselines.

---

## Contributing & Questions

Issues and PRs are welcome.
For FedScale runtime questions, the FedScale Slack is active (badge above).
For Bliss‑specific questions, open a GitHub issue in this repo.

---

**Happy federating!** 🎉