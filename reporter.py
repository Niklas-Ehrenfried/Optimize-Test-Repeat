"""
OTR Terminal Reporting & Dashboards.

Leverages the `rich` library to render a high-fidelity console dashboard summarizing 
historical milestones and active local scratchpad workspace trials with delta performance
indicators (latency/VRAM changes) and hardware bound limits.
"""
from typing import Optional, List, Dict, Set, Any
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from storage import OTRStateManager

def get_delta_indicator(current_val: Optional[float], best_val: Optional[float]) -> str:
    """
    Computes percentage change between current trial value and high-water mark best value,
    and returns a Rich-formatted color delta indicator string.

    - Positive changes indicate regression (red, downward triangles).
    - Negative changes indicate performance improvements (green, upward triangles).

    Args:
        current_val (float): The current trial metric value.
        best_val (float): The best milestone baseline metric value.

    Returns:
        str: Format styled string containing delta indicator arrows and percentage change.
    """
    if best_val is None or best_val <= 0.0:
        return ""
    if current_val is None or current_val < 0.0:
        return ""
    
    delta_pct: float = ((current_val - best_val) / best_val) * 100.0
    abs_pct: float = abs(delta_pct)
    
    if abs_pct < 0.05:
        return " [dim]• (0.0%)[/dim]"
        
    if delta_pct < 0.0:  # Performance improvement
        if abs_pct >= 15.0:
            return f" [bold green]▲▲▲ (-{abs_pct:.1f}%)[/bold green]"
        elif abs_pct >= 10.0:
            return f" [bold green]▲▲ (-{abs_pct:.1f}%)[/bold green]"
        elif abs_pct >= 5.0:
            return f" [bold green]▲ (-{abs_pct:.1f}%)[/bold green]"
        else:
            return f" [green](-{abs_pct:.1f}%)[/green]"
    else:  # Regression / drop in speed
        if abs_pct >= 15.0:
            return f" [bold red]▼▼▼ (+{abs_pct:.1f}%)[/bold red]"
        elif abs_pct >= 10.0:
            return f" [bold red]▼▼ (+{abs_pct:.1f}%)[/bold red]"
        elif abs_pct >= 5.0:
            return f" [bold red]▼ (+{abs_pct:.1f}%)[/bold red]"
        else:
            return f" [red](+{abs_pct:.1f}%)[/red]"


class OTRReporter:
    """
    Consolidates data from g_best.json, intra_gen_workspace.json, and history_journal.json
    to render formatted Rich dashboard tables in the terminal.
    """

    @staticmethod
    def render_dashboard(
        active_component: str,
        active_profiles: Optional[List[str]] = None,
        list_milestones: bool = False,
        base_dir: str = ".otr"
    ) -> None:
        """
        Renders the Consolidation Dashboard Table for a specific feature component.

        Displays:
        - The current active best run for each profile size (highlighted with a star).
        - Previous generation milestones (if list_milestones is True).
        - Active scratchpad workspace trials with color-coded latency/VRAM improvements/regressions.

        Args:
            active_component (str): The name/ID of the component to render.
            active_profiles (list, optional): Filter to only show these profile sizes.
            list_milestones (bool): Show full historical generation milestone rows.
            base_dir (str): Base directory storing OTR state JSON files.
        """
        console = Console()
        state_mgr = OTRStateManager(base_dir=base_dir)
        
        g_best: Dict[str, Any] = state_mgr._load(state_mgr.best_path, dict)
        workspace: List[Dict[str, Any]] = state_mgr._load(state_mgr.workspace_path, list)
        journal: List[Dict[str, Any]] = state_mgr._load(state_mgr.journal_path, list)

        try:
            from otr_config import DECIMAL_PLACES
        except ImportError:
            DECIMAL_PLACES = {"latency": 2, "vram": 2, "percentage": 1}
        
        latency_dec: int = DECIMAL_PLACES.get("latency", 2)
        vram_dec: int = DECIMAL_PLACES.get("vram", 2)
        pct_dec: int = DECIMAL_PLACES.get("percentage", 1)

        # Initialize Rich layout table
        table = Table(title=f"OTR Evolutionary Space: {active_component}", header_style="bold magenta", expand=True)
        table.add_column("Run Experiment Info / Notes", style="cyan", ratio=2.5)
        table.add_column("Generation", style="bold white", ratio=0.8)
        table.add_column("Latency (ms)", style="white", ratio=1.0)
        table.add_column("Transient VRAM", style="white", ratio=1.2)
        table.add_column("Hardware Bound Profile", style="white", ratio=2.0)

        # Gather profile sizes that are relevant to this component
        profiles_in_best: Set[str] = {key.split("::")[1] for key in g_best.keys() if key.startswith(active_component + "::")}
        profiles_in_workspace: Set[str] = {t["profile_size"] for t in workspace if t.get("component") == active_component and t.get("profile_size")}
        
        all_profiles: Set[str]
        if active_profiles:
            all_profiles = set(active_profiles)
        else:
            all_profiles = {"Small", "Medium", "Large", "Ultra"}.union(profiles_in_best).union(profiles_in_workspace)
        
        profile_order: Dict[str, int] = {"Small": 0, "Medium": 1, "Large": 2, "Ultra": 3, "Custom": 4}
        sorted_profiles: List[str] = sorted(list(all_profiles), key=lambda p: profile_order.get(p, 99))

        # Render Milestone Ledger Section
        for p_size in sorted_profiles:
            feature_key: str = f"{active_component}::{p_size}"
            best_record: Optional[Dict[str, Any]] = g_best.get(feature_key, None)
            best_gen: str = best_record.get("generation", "") if best_record else ""

            # Extract milestones directly out of clean journal matrix records
            records: List[Dict[str, Any]] = [rj for rj in journal if rj.get("component_key") == feature_key]

            for rj in records:
                m: Dict[str, Any] = rj.get("metrics", {})
                gen_str: str = m.get("generation", "GEN_0")
                is_best: bool = bool(best_record and gen_str == best_gen and rj.get("run_name") == best_record.get("run_name"))

                if not list_milestones and not is_best:
                    continue

                ratio_str: str = f"{m.get('compute_ratio_pct', 0.0):.{pct_dec}f}% Comp / {m.get('memory_transfer_pct', 0.0):.{pct_dec}f}% Mem"
                latency_str: str = f"{m.get('latency_ms', 0.0):.{latency_dec}f}ms"
                if m.get("requires_backward") and "forward_latency_ms" in m:
                    latency_str += f"\n[dim](F: {m['forward_latency_ms']:.{latency_dec}f} | B: {m['backward_latency_ms']:.{latency_dec}f})[/dim]"
                
                vram_str: str = f"{m.get('vram_transient_mb', 0.0):.{vram_dec}f} MB"
                if m.get("requires_backward") and "vram_transient_forward_mb" in m:
                    vram_str += f"\n[dim](F: {m['vram_transient_forward_mb']:.{vram_dec}f} | B: {m['vram_transient_backward_mb']:.{vram_dec}f})[/dim]"

                title: str
                if is_best:
                    title = f"[bold gold1]★ Best ({p_size}): {rj.get('run_name')}[/bold gold1]\n[dim]{rj.get('notes')}[/dim]"
                else:
                    title = f"Milestone ({p_size}): {rj.get('run_name')}\n[dim]{rj.get('notes')}[/dim]"

                table.add_row(title, gen_str, latency_str, vram_str, f"[bold green]{ratio_str}[/bold green]" if is_best else f"[green]{ratio_str}[/green]")

            # Fallback if best record is initialized but history file hasn't flushed yet
            if len(records) == 0 and best_record:
                ratio_str = f"{best_record.get('compute_ratio_pct', 0.0):.{pct_dec}f}% Comp / {best_record.get('memory_transfer_pct', 0.0):.{pct_dec}f}% Mem"
                latency_str = f"{best_record.get('latency_ms', 0.0):.{latency_dec}f}ms"
                vram_str = f"{best_record.get('vram_transient_mb', 0.0):.{vram_dec}f} MB"
                table.add_row(
                    f"[bold gold1]★ Best ({p_size}): {best_record.get('run_name')}[/bold gold1]\n[dim]{best_record.get('notes')}[/dim]",
                    best_gen, latency_str, vram_str, f"[bold green]{ratio_str}[/bold green]"
                )
            elif len(records) == 0 and not best_record:
                table.add_row(f"★ Best ({p_size}): No baseline run yet", "N/A", "N/A", "N/A", "[dim]N/A[/dim]")

        table.add_section()

        # Render Active Local Scratchpad Workspace Rows
        active_workspace_trials: List[Dict[str, Any]] = [t for t in workspace if t.get("component") == active_component]
        for trial in active_workspace_trials:
            m = trial.get("metrics") or {}
            p_size = trial.get("profile_size", "Unknown")
            if active_profiles and p_size not in active_profiles:
                continue

            best_rec: Optional[Dict[str, Any]] = g_best.get(f"{active_component}::{p_size}")
            is_failed: bool = bool(m.get("status") == "FAILED" or "error" in m or m.get("latency_ms", -1.0) < 0.0)

            if is_failed:
                table.add_row(f"{trial.get('run_name')} ({p_size})\n[dim]{trial.get('notes')}[/dim]", "[bold red]FAILED[/bold red]", "[bold red]FAILED[/bold red]", "0.00 MB", f"[bold red]Error: {m.get('error')}[/bold red]")
            else:
                best_lat: float = best_rec.get("latency_ms", float("inf")) if best_rec else float("inf")
                curr_lat: float = m.get("latency_ms", -1.0)
                outcome_str: str
                if best_rec is None or curr_lat < best_lat:
                    outcome_str = "IMPROVED"
                else:
                    outcome_str = "REJECTED"

                lat_ind: str = get_delta_indicator(m.get("latency_ms"), best_rec.get("latency_ms") if best_rec else None)
                vram_ind: str = get_delta_indicator(m.get("vram_transient_mb"), best_rec.get("vram_transient_mb") if best_rec else None)
                
                latency_str = f"{m['latency_ms']:.{latency_dec}f}ms{lat_ind}"
                if m.get("requires_backward") and "forward_latency_ms" in m:
                    latency_str += f"\n[dim](F: {m['forward_latency_ms']:.{latency_dec}f} | B: {m['backward_latency_ms']:.{latency_dec}f})[/dim]"
                
                vram_str = f"{m['vram_transient_mb']:.{vram_dec}f} MB{vram_ind}"
                if m.get("requires_backward") and "vram_transient_forward_mb" in m:
                    vram_str += f"\n[dim](F: {m['vram_transient_forward_mb']:.{vram_dec}f} | B: {m['vram_transient_backward_mb']:.{vram_dec}f})[/dim]"

                bound_str: str = f"{m['compute_ratio_pct']:.{pct_dec}f}% Comp / {m['memory_transfer_pct']:.{pct_dec}f}% Mem"
                table.add_row(f"{trial.get('run_name')} ({p_size})\n[dim]{trial.get('notes')}[/dim]", outcome_str, latency_str, vram_str, bound_str)

        console.print("\n")
        console.print(Panel(table, border_style="magenta", title="[bold white]Execution Logs[/bold white]"))
