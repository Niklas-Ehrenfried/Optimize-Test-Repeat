# OTR (Optimize-Test-Repeat) Optimization Framework

An agentic, multi-metric performance profiling and continuous optimization framework for deep learning architectures and high-throughput software algorithms. 

OTR automates performance monitoring across iterations by storing metrics, tracking improvement milestones (generations), and preventing regression using dynamic local state ledgers and git branch integrations.

---

## 🚀 Key Features

* **Multi-Metric Telemetry Profiling**: Captures media/tensor throughput latency, transient device memory (VRAM) usage, and low-level hardware operation bounds (Compute vs. Memory bound ratios).
* **Dual Execution Mode**: Supports PyTorch deep learning models (with CUDA acceleration, peak memory tracking, and forward/backward path profiling) as well as standard Python library algorithms with zero overhead.
* **Intelligent Local Ledgers**:
  * `g_best.json`: The immutable "high-water mark" tracking the best performance across generations.
  * `intra_gen_workspace.json`: A volatile local scratchpad playground for logging local, iterative development runs.
  * `history_journal.json`: The immutable evolutionary log storing only successfully promoted generation milestones.
* **Component-Wide Purging**: When any profile of a component advances to a new generation, all other experimental runs for that component are automatically pruned from the volatile workspace to prevent table bloat.
* **Headless Security Guard**: Prevents CI/Agent executions from staging unrelated repository changes by scoping `git add` exclusively to OTR metrics and test directories (`.otr/*` and `tests/*`).
* **Stage-Only Workflow**: To avoid cluttering your commit history with automated spams, OTR only automatically stages files (`git add`) upon milestone promotion, allowing you to review changes and run `git commit` manually when ready.

---

## 🛠️ Setup

Ensure you have `uv` installed. Set up the virtual environment and install dependencies:

```bash
# Install dependencies
uv sync
```

---

## 📋 Command-Line Arguments

The runner tool (`otr_run.py`) provides the following options:

| Flag | Long Flag | Description |
|---|---|---|
| `-n` | `--name` | Unique tag name identifying the code optimization experiment run (e.g. `fused_kernel`). |
| `-m` | `--notes` | A 1-2 sentence description detailing the code changes or hypothesis. |
| `-f` | `--feature` | Target benchmark index (e.g. `1`), list of indexes (e.g. `1,3`), or `all`/`a` to run everything. |
| `-l` | `--list-milestones` | Displays all historical milestone entries in the terminal table. |
| `--no-stage` | `--no-stage` | Advances generation milestones locally but skips Git staging. |
| `-t` | `--test-only` | Runs profiling and outputs results without modifying any JSON files or staging changes in Git. |

---

## 🏃 Running Benchmarks

### 1. Run All Benchmarks
To run the entire test suite and profile all registered components:
```bash
uv run python otr_run.py --name="my_run" --notes="My optimization changes" --feature="all"
```

### 2. Run a Single Selected Benchmark
Run only the benchmark associated with feature index `1`:
```bash
uv run python otr_run.py --name="single_test" --notes="Testing single feature" --feature="1"
```

### 3. Run a Subset of Benchmarks (e.g., 1 and 3)
Run multiple specific features by passing a comma-separated list of keys:
```bash
uv run python otr_run.py --name="subset_run" --notes="Testing features 1 and 3" --feature="1,3"
```

### 4. Run in Test-Only Mode (No State Mutation)
Validate your code changes without modifying databases or staging changes in git:
```bash
uv run python otr_run.py --name="dry_run" --notes="Verifying speedup" --feature="1" --test-only
```

---

## 📊 Evolutionary Space Table Layout

When the suite finishes, a high-fidelity console dashboard will render highlighting:
* 🌟 **Best Row**: Marked with a gold star representing the current milestone baseline.
* 📈 **Delta Indicators**: Real-time performance improvements or regressions shown with colored indicators (e.g., `▲▲ (-12.5%)` or `▼ (+1.5%)`).
* ⚙️ **Hardware Bound Profile**: Breakdowns showing the percentage of execution spent on **Compute** operations versus **Memory** transfers.
