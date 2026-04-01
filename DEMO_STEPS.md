## ALOHA Sim baseline demo

This is the first end-to-end validation path from `PLAN.md`.

### 1. Start the baseline policy server

```bash
uv run scripts/serve_policy.py --env ALOHA_SIM
```

### 2. In a second terminal, run ALOHA Sim

```bash
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

### 3. Inspect saved rollout videos

By default the simulator writes rollout videos under:

```bash
data/aloha_sim/videos
```

### 4. Optional: use Docker instead

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## CoreWeave notes

- Submit the Slurm jobs from the SUNK login node, not from a compute node.
- Use `jobs/smoke_openpi.sbatch` before the first real training run.
- Use `jobs/train_aloha_sim_jax.sbatch` for the first JAX training pass.
