# OTR Agent Optimization Guide

You are operating inside an automated Loop Engineering optimization harness. Your mission is to iteratively maximize performance metrics for specific architectural features without breaking system logic.

---

## 📂 Repository State Architecture

* **`g_best.json`**: Read this first to find the active milestone generation (`GEN_0`, `GEN_1`, etc.) and the high-water mark latency/VRAM metrics you need to beat.
* **`intra_gen_workspace.json`**: Volatile local scratchpad showing recent trials within the current generation. Review these to avoid repeating past optimization failures.
* **`history_journal.json`**: Log of promoted generation history. Do not modify.

---

## 🤖 Headless & Agent Safety Features

When running in an automated container, CI pipeline, or terminal where `sys.stdin.isatty()` is `False` (headless execution):
1. **Automated Promotion**: OTR will automatically confirm generation promotions when a new high-water mark is set.
2. **Git Staging Guard**: Staging (`git add`) is restricted strictly to OTR configuration files and test directories (`.otr/*` and `tests/*`) to prevent automated agents from staging unrelated codebase changes or half-written files.
3. **Commit Control**: OTR only automatically stages files in Git. It never executes `git commit`. The actual commit step must be run manually or by your surrounding agent driver script.

---

## 🏃 Execution Commands

### 1. Test-Only Mode (Recommended for Tweaks)
To run a trial and profile it without updating databases or staging changes in git:
```bash
uv run python otr_run.py --name="my_trial" --notes="Applying flash attention" --feature="1" --test-only
```

### 2. Multi-Feature Subset Running
To run a specific subset of features (e.g. features 1 and 3):
```bash
uv run python otr_run.py --name="flash_routing" --notes="Fused gating kernels" --feature="1,3"
```

### 3. Prompting Promotions
When you are ready to advance the generation context, run without `--test-only`. Headless runner will auto-confirm and stage the successful optimization files in Git.
```bash
uv run python otr_run.py --name="optimized_routing" --notes="Reduced redundant kernel allocations" --feature="1"
```
