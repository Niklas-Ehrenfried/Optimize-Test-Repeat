"""
OTR State Manager Module.

Manages persistent database states for OTR benchmarks, including:
- g_best.json (current high-water mark bests per component and profile).
- intra_gen_workspace.json (volatile scratchpad area for running experiments).
- history_journal.json (immutable historical ledger of successful promotions).

Coordinates promotion prompts, workspace cleanup, and Git commit integrations.
"""
import os
import json
import subprocess
import sys
from typing import Dict, List, Set, Any, Optional, Union

class OTRStateManager:
    """
    State manager for tracking and updating OTR state files. Handles evaluating
    telemetry records, archiving promotion candidates, workspace pruning, and 
    automated or interactive Git commits.
    """
    
    # Session tracking caches
    pending_promotions: List[Dict[str, Any]] = []
    session_profiled_components: Dict[str, Set[str]] = {}  # component_id -> set(profile_sizes)

    def __init__(self, base_dir: str = ".otr") -> None:
        """
        Initializes the state manager paths.

        Args:
            base_dir (str): Base directory storing OTR state JSON files.
        """
        self.best_path: str = os.path.join(base_dir, "g_best.json")
        self.workspace_path: str = os.path.join(base_dir, "intra_gen_workspace.json")
        self.journal_path: str = os.path.join(base_dir, "history_journal.json")

    def _load(self, path: str, default_type: type = dict) -> Any:
        """
        Loads a JSON file safely, returning a default initialized structure on missing files or error.

        Args:
            path (str): File path to load.
            default_type (type): Type to instantiate on failure (dict or list).

        Returns:
            any: Loaded JSON object or default_type instance.
        """
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return default_type()
        return default_type()

    def _save(self, path: str, data: Any) -> None:
        """
        Writes data structures to a JSON file with pretty formatting.

        Args:
            path (str): File path to write.
            data (any): JSON-serializable structure to save.
        """
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def evaluate_and_log(
        self,
        component: str,
        profile_size: str,
        run_name: str,
        notes: str,
        metrics: Dict[str, Any],
        test_only: bool = False
    ) -> str:
        """
        Evaluates a new benchmark execution.

        Processes:
        1. Rounding telemetry numbers according to OTR config precision.
        2. Checking for stale generation entries of this component and purging them from the workspace.
        3. Determining whether the execution is an improvement over the current baseline in g_best.json.
        4. Staging the run inside the local workspace (intra_gen_workspace.json).
        5. Registering promotion candidates for final session approval when applicable.

        Args:
            component (str): The component name/ID.
            profile_size (str): Target profile size (e.g. "Small").
            run_name (str): Identifier tag for this experiment execution.
            notes (str): Detailed commentary notes for the run.
            metrics (dict): Unrounded telemetry metrics.
            test_only (bool): If True, skips staging the run for generation advancement.

        Returns:
            str: Trial outcome status ("FAILED", "IMPROVED_INTRA", or "REJECTED").
        """
        if component not in OTRStateManager.session_profiled_components:
            OTRStateManager.session_profiled_components[component] = set()
        OTRStateManager.session_profiled_components[component].add(profile_size)

        # Round metrics according to DECIMAL_PLACES if defined in otr_config
        try:
            from otr_config import DECIMAL_PLACES
        except ImportError:
            DECIMAL_PLACES = {"latency": 2, "vram": 2, "percentage": 1}

        lat_dec: int = DECIMAL_PLACES.get("latency", 2)
        vram_dec: int = DECIMAL_PLACES.get("vram", 2)
        pct_dec: int = DECIMAL_PLACES.get("percentage", 1)

        key_mapping: Dict[str, int] = {
            "latency_ms": lat_dec,
            "forward_latency_ms": lat_dec,
            "backward_latency_ms": lat_dec,
            "vram_base_mb": vram_dec,
            "vram_transient_forward_mb": vram_dec,
            "vram_transient_backward_mb": vram_dec,
            "vram_transient_mb": vram_dec,
            "compute_ratio_pct": pct_dec,
            "memory_transfer_pct": pct_dec,
            "gpu_idle_overhead_pct": pct_dec
        }

        for k, dec in key_mapping.items():
            if k in metrics and isinstance(metrics[k], (int, float)):
                metrics[k] = round(float(metrics[k]), dec)

        g_best: Dict[str, Any] = self._load(self.best_path, dict)
        workspace: List[Dict[str, Any]] = self._load(self.workspace_path, list)

        # Purge all workspace entries for any component if any of its profiles have reached a new generation context
        components_to_purge: Set[str] = set()
        for t in workspace:
            t_comp: str = t.get("component", "")
            t_prof: str = t.get("profile_size", "")
            t_key: str = f"{t_comp}::{t_prof}"
            t_best: Optional[Dict[str, Any]] = g_best.get(t_key)
            t_best_gen: str = t_best.get("generation", "GEN_0") if t_best else "GEN_0"
            
            try:
                t_gen_num: int = int(t.get("generation_context", "GEN_0").split("_")[1])
                best_gen_num: int = int(t_best_gen.split("_")[1])
            except (IndexError, ValueError):
                t_gen_num = 0
                best_gen_num = 0
                
            if t_gen_num < best_gen_num:
                components_to_purge.add(t_comp)

        if components_to_purge:
            workspace = [t for t in workspace if t.get("component") not in components_to_purge]

        feature_key: str = f"{component}::{profile_size}"
        best_record: Optional[Dict[str, Any]] = g_best.get(feature_key, None)
        
        current_latency: float = metrics.get("latency_ms", -1.0)
        is_candidate: bool = False
        proposed_gen: str = "GEN_0"

        status: str
        if metrics.get("status") == "FAILED" or current_latency < 0:
            status = "FAILED"
        elif best_record is None:
            is_candidate = not test_only
            proposed_gen = "GEN_0"
            status = "IMPROVED_INTRA"
        else:
            target_best_latency: float = best_record.get("latency_ms", float("inf"))
            if current_latency < target_best_latency:
                prev_gen_str: str = best_record.get("generation", "GEN_0")
                try:
                    prev_gen_num: int = int(prev_gen_str.split("_")[1])
                except (IndexError, ValueError):
                    prev_gen_num = 0
                proposed_gen = f"GEN_{prev_gen_num + 1}"
                is_candidate = not test_only
                status = "IMPROVED_INTRA"
            else:
                status = "REJECTED"

        # Create clean copy of metrics for saving in JSON (remove status: SUCCESS)
        clean_metrics: Dict[str, Any] = dict(metrics)
        if clean_metrics.get("status") == "SUCCESS":
            clean_metrics.pop("status", None)

        # Construct workspace log segment entry
        run_entry: Dict[str, Any] = {
            "component": component,
            "profile_size": profile_size,
            "run_name": run_name,
            "notes": notes,
            "generation_context": best_record.get("generation", "GEN_0") if best_record else "GEN_0",
            "metrics": clean_metrics
        }

        # Track candidate status in memory for session finish processing
        if is_candidate:
            OTRStateManager.pending_promotions.append({
                "component": component,
                "profile_size": profile_size,
                "run_name": run_name,
                "notes": notes,
                "metrics": clean_metrics,
                "best_record": best_record,
                "proposed_gen_str": proposed_gen
            })

        # CRITICAL: Clean Deduplication. Overwrite matching run names in place.
        replaced: bool = False
        for i, entry in enumerate(workspace):
            if (entry.get("component") == component and 
                entry.get("profile_size") == profile_size and 
                entry.get("run_name") == run_name):
                workspace[i] = run_entry
                replaced = True
                break
        
        if not replaced:
            workspace.append(run_entry)
            
        self._save(self.workspace_path, workspace)
        return status

    def process_pending_promotions(self, no_stage: bool = False, test_only: bool = False) -> None:
        """
        Interates through all pending promotions collected during the session.

        - If running in an interactive terminal, prompts the user for confirmation.
        - If headless (CI / Agentic execution), automatically confirms the promotion.
        - Promotes confirmed candidates to g_best.json.
        - Appends records to history_journal.json (historical milestones).
        - Purges the workspace of all entries belonging to the promoted component.
        - Triggers Git staging for all promoted metrics and source code when enabled.

        Args:
            no_stage (bool): If True, updates local JSON files but skips Git staging.
            test_only (bool): If True, bypasses all updates/staging entirely.
        """
        if test_only or not OTRStateManager.pending_promotions:
            return

        g_best: Dict[str, Any] = self._load(self.best_path, dict)
        journal: List[Dict[str, Any]] = self._load(self.journal_path, list)
        confirmed_promos: List[Dict[str, Any]] = []

        for promo in OTRStateManager.pending_promotions:
            component: str = promo["component"]
            profile_size: str = promo["profile_size"]
            run_name: str = promo["run_name"]
            notes: str = promo["notes"]
            metrics: Dict[str, Any] = promo["metrics"]
            best_record: Optional[Dict[str, Any]] = promo["best_record"]
            proposed_gen_str: str = promo["proposed_gen_str"]
            feature_key: str = f"{component}::{profile_size}"

            promotion_confirmed: bool
            if sys.stdin.isatty():
                if best_record is None:
                    print(f"\n[OTR] Milestone reached: {proposed_gen_str} baseline initialized for {component} ({profile_size})!")
                    prompt = "Confirm baseline promotion? (y/n) [y]: "
                else:
                    print(f"\n[OTR] Milestone reached: Improvement detected for {component} ({profile_size})!")
                    print(f"      Latency: {best_record['latency_ms']:.2f}ms -> {metrics['latency_ms']:.2f}ms")
                    prompt = f"Advance workspace to milestone generation {proposed_gen_str}? (y/n) [y]: "
                
                sys.stdout.write(prompt)
                sys.stdout.flush()
                ans: str = sys.stdin.readline().strip().lower()
                promotion_confirmed = (ans != "n")
            else:
                promotion_confirmed = True

            if promotion_confirmed:
                # Create a clean flat record copy for g_best.json
                best_record_to_save: Dict[str, Any] = dict(metrics)
                for k in ["status", "outcome", "run_name", "notes", "generation"]:
                    best_record_to_save.pop(k, None)
                best_record_to_save["generation"] = proposed_gen_str
                best_record_to_save["run_name"] = run_name
                best_record_to_save["notes"] = notes

                # Promote Best Record anchors
                g_best[feature_key] = best_record_to_save
                self._save(self.best_path, g_best)

                # Migration: Write ONLY this single successful run into the historical journal
                # Make sure the journal's metrics doesn't contain duplicate fields
                journal_metrics: Dict[str, Any] = dict(metrics)
                for k in ["status", "outcome", "run_name", "notes", "generation"]:
                    journal_metrics.pop(k, None)
                journal_metrics["generation"] = proposed_gen_str

                journal.append({
                    "component_key": feature_key,
                    "run_name": run_name,
                    "notes": notes,
                    "metrics": journal_metrics
                })
                self._save(self.journal_path, journal)

                # Clean up the scratchpad workspace for this entire feature sequence
                workspace_data: List[Dict[str, Any]] = self._load(self.workspace_path, list)
                workspace_data = [t for t in workspace_data if t.get("component") != component]
                self._save(self.workspace_path, workspace_data)

                confirmed_promos.append({
                    "component": component,
                    "profile_size": profile_size,
                    "run_name": run_name,
                    "target_gen": proposed_gen_str,
                    "latency": metrics["latency_ms"],
                    "vram": metrics.get("vram_transient_mb", 0.0)
                })

        if confirmed_promos:
            if no_stage:
                print(f"\n[OTR] {len(confirmed_promos)} milestones advanced in configurations. (Git staging skipped via flag)")
            else:
                self._git_stage_all_promotions(confirmed_promos)

        OTRStateManager.pending_promotions = []

    def _git_stage_all_promotions(self, confirmed_promos: List[Dict[str, Any]]) -> None:
        """
        Stages newly promoted milestone files in Git.

        - Scans current `git status --porcelain` to identify changed files.
        - If interactive: prompts the user to select specific files (indices, all, none) to stage.
        - If headless (agent/CI): stages only OTR configurations and test files (.otr/* and tests/*)
          as a safety boundary, preventing accidental staging of unrelated files.

        Args:
            confirmed_promos (list): A list of dictionaries representing confirmed promotions.
        """
        if not confirmed_promos:
            return

        msg: str
        if len(confirmed_promos) == 1:
            p: Dict[str, Any] = confirmed_promos[0]
            msg = f"OTR Milestone: {p['target_gen']} [{p['component']} | {p['profile_size']}] -> '{p['run_name']}' | Latency: {p['latency']:.2f}ms | Transient VRAM: {p['vram']:.2f}MB"
        else:
            msg = f"OTR Milestone: Multiple Promotions ({len(confirmed_promos)} features updated)\n\n" + "\n".join(
                f"- {p['target_gen']} [{p['component']} | {p['profile_size']}] -> '{p['run_name']}' | Latency: {p['latency']:.2f}ms | Transient VRAM: {p['vram']:.2f}MB"
                for p in confirmed_promos
            )

        changed_files: List[str] = []
        try:
            res: subprocess.CompletedProcess[str] = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            )
            for line in res.stdout.splitlines():
                if len(line) >= 4:
                    f_path: str = line[3:].strip()
                    if "->" in f_path:
                        f_path = f_path.split("->")[1].strip()
                    if f_path.startswith('"') and f_path.endswith('"'):
                        f_path = f_path[1:-1]
                    changed_files.append(f_path)
        except Exception:
            pass

        selected_files: List[str] = []
        if sys.stdin.isatty():
            if changed_files:
                print("\nChanged files detected in git status:")
                for idx, f in enumerate(changed_files):
                    print(f"  [{idx}] {f}")
                print("  [a] All files")
                print("  [n] None (metrics only)")
                
                sys.stdout.write("Select files to stage (comma-separated indices, 'a', or 'n') [a]: ")
                sys.stdout.flush()
                selection: str = sys.stdin.readline().strip().lower()
                if not selection:
                    selection = "a"
                
                if selection == "a":
                    selected_files = changed_files
                elif selection == "n":
                    selected_files = []
                else:
                    try:
                        indices: List[int] = [int(i.strip()) for i in selection.split(",")]
                        selected_files = [changed_files[i] for i in indices if 0 <= i < len(changed_files)]
                    except ValueError:
                        selected_files = changed_files
            else:
                print("\nNo changed files detected in git status.")
        else:
            # Headless Agent Security Mode: Only stage metrics and tests to prevent codebase pollution
            selected_files = [f for f in changed_files if f.startswith(".otr/") or f.startswith("tests/")]

        try:
            subprocess.run(["git", "add", self.best_path, self.journal_path, self.workspace_path], check=True)
            for f in selected_files:
                if os.path.exists(f):
                    try:
                        subprocess.run(["git", "add", f], check=True)
                    except subprocess.CalledProcessError:
                        try:
                            # Try force adding if the file is tracked-but-ignored
                            subprocess.run(["git", "add", "-f", f], check=True)
                        except subprocess.CalledProcessError:
                            pass
            print(f"\n[OTR] Staged milestone changes in Git. Run 'git commit' to commit them.")
            print(f"Suggested commit message:\n{msg}")
        except subprocess.CalledProcessError as e:
            print(f"[OTR] Warning: Git staging failed: {e}")
