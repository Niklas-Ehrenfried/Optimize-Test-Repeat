"""
OTR (Optimize-Test-Repeat) Benchmark Runner Utility & Pytest Harness.

This file serves as both:
1. The command-line entrypoint (otr_run.py) for discovering and running OTR benchmarks.
2. The implementation of pytest hooks and fixtures used by pytest sessions.
"""
import os
import sys
import argparse
import subprocess
import re
from typing import Dict, Tuple, Any, Callable, Optional, List, Union

def discover_features(test_dir: str = "tests") -> Dict[str, Tuple[str, str]]:
    """
    Scans the tests/ directory to dynamically discover OTR benchmark components.
    
    Walks the files in tests/, matching `component_id` arguments to index them.
    This dynamically populates the CLI runner selection menu.

    Args:
        test_dir (str): Directory containing benchmark test files.

    Returns:
        dict: A dictionary mapping index strings (e.g. "1", "2") to a tuple
              containing the component name and its test file path.
    """
    features: Dict[str, Tuple[str, str]] = {}
    idx: int = 1
    if not os.path.isdir(test_dir):
        return features
        
    for fname in sorted(os.listdir(test_dir)):
        if fname.startswith("test_") and fname.endswith(".py"):
            path = os.path.join(test_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Match component_id="..." or component_id='...'
                matches = re.findall(r'component_id\s*=\s*["\']([^"\']+)["\']', content)
                for comp_id in matches:
                    if comp_id not in [v[0] for v in features.values()]:
                        features[str(idx)] = (comp_id, path)
                        idx += 1
            except Exception:
                pass
    return features


def prompt_user(text: str, default: str = "") -> str:
    """
    Helper function to prompt user for input via stdin, supporting default values
    and handling KeyboardInterrupt cleanly.

    Args:
        text (str): The prompt message to show.
        default (str): The default value to return if input is empty.

    Returns:
        str: The user input or default value.
    """
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
        val = sys.stdin.readline().strip()
        if not val:
            return default
        return val
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


# =====================================================================
# PYTEST HOOKS AND FIXTURES (Unified from conftest.py)
# =====================================================================

def pytest_addoption(parser: Any) -> None:
    """
    Registers custom command-line options for the OTR pytest plugin.

    Args:
        parser (pytest.Parser): The pytest command-line parser object.
    """
    parser.addoption(
        "--run-name",
        action="store",
        required=True,
        help="Unique identifying tag string for this specific code optimization experiment.",
    )
    parser.addoption(
        "--notes",
        action="store",
        required=True,
        help="A 1-2 sentence summary statement detailing what code changes were made.",
    )
    parser.addoption(
        "--list-milestones",
        action="store_true",
        default=False,
        help="List all previous generation milestones in the execution logs table.",
    )
    parser.addoption(
        "--no-stage",
        action="store_true",
        default=False,
        help="Promote successful runs to the next generation in OTR files but skip Git staging.",
    )
    parser.addoption(
        "--test-only",
        action="store_true",
        default=False,
        help="Run benchmarks without promoting generations or committing changes.",
    )
    parser.addoption(
        "--test-dir",
        action="store",
        default="tests",
        help="Directory containing the target OTR test files, config, and tracking data.",
    )


def pytest_configure(config: Any) -> None:
    """
    Initial configuration hook for pytest. Creates the OTR metadata
    directory (.otr/) if it does not already exist.

    Args:
        config (pytest.Config): The pytest config object.
    """
    test_dir = config.getoption("--test-dir")
    os.makedirs(os.path.join(test_dir, ".otr"), exist_ok=True)
    # Inject the host repository's test directory path to resolve custom configs natively
    sys.path.insert(0, os.path.abspath(test_dir))


def otr_bench(request: Any) -> Callable[..., Dict[str, Any]]:
    """
    Pytest fixture injected into benchmark tests to profile and log execution metrics.

    Resolves selected configuration profile parameters, runs the function telemetry 
    via OTRProfilerEngine, and logs the metrics to storage.

    Args:
        request (pytest.FixtureRequest): The pytest fixture request context.

    Returns:
        callable: The benchmark execution wrapper function.
    """
    import pytest
    from engine import OTRProfilerEngine
    from storage import OTRStateManager

    run_name: str = request.config.getoption("--run-name")
    notes: str = request.config.getoption("--notes")
    test_only: bool = request.config.getoption("--test-only")
    test_dir: str = request.config.getoption("--test-dir")
    base_dir: str = os.path.join(test_dir, ".otr")
    state_mgr = OTRStateManager(base_dir=base_dir)

    def _execute(
        component_id: str,
        profile_input: Union[str, Dict[str, Any]],
        func: Callable[..., Any],
        inputs_factory: Callable[[Dict[str, Any]], Tuple[Any, ...]],
        require_backward: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Executes and profiles the target benchmark function.

        Args:
            component_id (str): The unique name of the feature component.
            profile_input (str or dict): The name of the config profile size (e.g. "Small")
                                        or a dict of custom dimensions.
            func (callable): The target function to benchmark.
            inputs_factory (callable): A factory returning arguments for func given shape dims.
            require_backward (bool, optional): Force backward pass execution.

        Returns:
            dict: The compiled and rounded performance telemetry metrics.
        """
        # Resolve profile input to shape dictionary
        if isinstance(profile_input, str):
            profile_name: str = profile_input
            try:
                from otr_config import PROFILES

                # Support multi-domain flat structure lookups smoothly
                shapes: Dict[str, Any] = {}
                if profile_name in PROFILES:
                    shapes = PROFILES[profile_name]
                else:
                    for domain, sub_profiles in PROFILES.items():
                        if (
                            isinstance(sub_profiles, dict)
                            and profile_name in sub_profiles
                        ):
                            shapes = sub_profiles[profile_name]
                            break
                    if not shapes:
                        shapes = PROFILES.get("LLM_Architecture", {}).get(
                            profile_name,
                            PROFILES.get("LLM_Architecture", {}).get("Small", {}),
                        )
            except ImportError:
                shapes = {}
        elif isinstance(profile_input, dict):
            shapes = profile_input
            profile_name = "Custom"
        else:
            shapes = {}
            profile_name = str(profile_input)

        # Generate inputs using inputs_factory
        inputs: Tuple[Any, ...] = inputs_factory(shapes)

        # Capture multi-metric profile (with optional backward pass config)
        metrics: Dict[str, Any] = OTRProfilerEngine.run_telemetry(
            func, inputs, require_backward=require_backward
        )

        # Update state records and handle Git milestones
        status: str = state_mgr.evaluate_and_log(
            component_id, profile_name, run_name, notes, metrics, test_only=test_only
        )
        metrics["status"] = status

        return metrics

    # Decorate internal _execute as a pytest fixture helper
    import pytest
    return _execute


# Register otr_bench as a pytest fixture explicitly
import pytest
otr_bench = pytest.fixture(scope="function")(otr_bench)


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    """
    Hook executed at the completion of the pytest suite execution.
    Renders the consolidation dashboard tables for all components profiled in the session,
    and processes pending promotion candidates (updating milestones, committing to Git).

    Args:
        session (pytest.Session): The pytest session object.
        exitstatus (int): Pytest exit status.
    """
    # Prevent parallel xdist worker threads from corrupting metrics databases
    if hasattr(session.config, "workerinput"):
        return

    from storage import OTRStateManager
    from reporter import OTRReporter

    # Print one unified console dashboard for each component profiled during this session
    list_milestones: bool = session.config.getoption("--list-milestones")
    no_stage: bool = session.config.getoption("--no-stage")
    test_dir: str = session.config.getoption("--test-dir")
    base_dir: str = os.path.join(test_dir, ".otr")

    for component, profiles_run in sorted(
        OTRStateManager.session_profiled_components.items()
    ):
        OTRReporter.render_dashboard(
            component, list(profiles_run), list_milestones=list_milestones, base_dir=base_dir
        )

    # Process promotions and ask for confirmations
    test_only: bool = session.config.getoption("--test-only")
    state_mgr = OTRStateManager(base_dir=base_dir)
    state_mgr.process_pending_promotions(no_stage=no_stage, test_only=test_only)


def pytest_pyfunc_call(pyfuncitem: Any) -> Optional[bool]:
    """
    Intercepts the execution of benchmark test functions.
    Wraps the entire test run in a try-except block so that compilation errors,
    OOMs, and other runtime failures are caught, logged gracefully as FAILED,
    and do not crash the runner session.
    """
    if "otr_bench" not in pyfuncitem.fixturenames:
        return None  # Let pytest run it normally

    import gc
    from storage import OTRStateManager

    # Resolve component_id and profile_name
    module = pyfuncitem.module
    component_id = getattr(module, "component_id", pyfuncitem.name)

    profile_name = "Unknown"
    if hasattr(pyfuncitem, "callspec") and pyfuncitem.callspec:
        profile_name = pyfuncitem.callspec.params.get("profile_name", "Unknown")

    try:
        # Run the test function via standard pytest invocation
        pyfuncitem.obj(**{name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames})
        return True
    except Exception as e:
        # Perform memory cleanup immediately
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Log the failure metrics to the state manager
        test_dir = pyfuncitem.config.getoption("--test-dir")
        base_dir = os.path.join(test_dir, ".otr")
        state_mgr = OTRStateManager(base_dir=base_dir)

        run_name = pyfuncitem.config.getoption("--run-name")
        notes = pyfuncitem.config.getoption("--notes")
        test_only = pyfuncitem.config.getoption("--test-only")

        metrics = {
            "latency_ms": -1.0,
            "forward_latency_ms": -1.0,
            "backward_latency_ms": -1.0,
            "vram_base_mb": 0.0,
            "vram_transient_mb": 0.0,
            "vram_transient_forward_mb": 0.0,
            "vram_transient_backward_mb": 0.0,
            "compute_ratio_pct": 0.0,
            "memory_transfer_pct": 0.0,
            "gpu_idle_overhead_pct": 0.0,
            "requires_backward": False,
            "status": "FAILED",
            "error": f"{type(e).__name__}: {str(e)}",
        }
        state_mgr.evaluate_and_log(
            component_id, profile_name, run_name, notes, metrics, test_only=test_only
        )

        import traceback
        traceback.print_exc()
        print(f"\n[OTR] Profile '{profile_name}' failed: {type(e).__name__}: {str(e)}")
        return True


# =====================================================================
# CLI MAIN RUNNER ENTRYPOINT
# =====================================================================

def main() -> None:
    """
    Main CLI executable routine. Coordinates target component discovery,
    prompts the user for interactive inputs when missing, constructs
    the forwarded pytest command, and invokes the pytest subprocess.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-d", "--test-dir", default="tests")
    pre_args, _ = pre_parser.parse_known_args()

    FEATURES: Dict[str, Tuple[str, str]] = discover_features(pre_args.test_dir)
    
    parser = argparse.ArgumentParser(description="OTR Benchmark Runner Utility")
    parser.add_argument("-n", "--name", help="Experiment run name")
    parser.add_argument("-m", "--notes", help="Description notes detailing changes")
    parser.add_argument("-f", "--feature", help="Target feature key(s) (comma-separated, e.g. '1,3') or 'all'")
    parser.add_argument("-l", "--list-milestones", action="store_true", help="Display previous milestones in the table")
    parser.add_argument("--no-stage", action="store_true", help="Skip Git staging during promotions")
    parser.add_argument("-t", "--test-only", action="store_true", help="Run benchmarks without promoting generations or committing changes")
    parser.add_argument("-d", "--test-dir", default="tests", help="Directory containing OTR test files and config")
    
    args, unknown = parser.parse_known_args()

    # Resolve run name
    run_name: str = args.name or ""
    if not run_name:
        run_name = prompt_user("Enter experiment run name: ").strip()
        while not run_name:
            print("Run name cannot be empty.")
            run_name = prompt_user("Enter experiment run name: ").strip()

    # Resolve notes
    notes: str = args.notes or ""
    if not notes:
        notes = prompt_user("Enter experiment notes: ").strip()
        while not notes:
            print("Notes cannot be empty.")
            notes = prompt_user("Enter experiment notes: ").strip()

    # Resolve feature selection
    feature: str = args.feature or ""
    
    def is_valid_selection(val: str) -> bool:
        if val in ["a", "all"]:
            return True
        parts = [p.strip() for p in val.split(",") if p.strip()]
        if not parts:
            return False
        return all(p in FEATURES for p in parts)

    if not feature:
        print("\nSelect feature to benchmark:")
        for k, (name, path) in FEATURES.items():
            print(f"  [{k}] {name} ({path})")
        print("  [a] All Features (runs all test modules)")
        
        feature = prompt_user("Choose feature [a]: ", "a").strip().lower()
        while not is_valid_selection(feature):
            print("Invalid selection.")
            feature = prompt_user(f"Choose feature [a] (options: {', '.join(FEATURES.keys())}, a): ", "a").strip().lower()

    # Determine target test paths
    test_targets: List[str] = []
    if feature in ["a", "all"]:
        test_targets = [args.test_dir]
        print(f"\nRunning benchmarks for all features in {args.test_dir}...")
    else:
        selected_keys: List[str] = [k.strip() for k in feature.split(",") if k.strip()]
        selected_details: List[str] = []
        for k in selected_keys:
            comp_id, path = FEATURES[k]
            if path not in test_targets:
                test_targets.append(path)
            selected_details.append(f"{comp_id} ({path})")
        print(f"\nRunning benchmark for: {', '.join(selected_details)}...")

    # Build pytest command
    cmd: List[str] = ["uv", "run", "pytest"] + test_targets + ["-s"]
    cmd.extend(["--run-name", run_name])
    cmd.extend(["--notes", notes])
    cmd.extend(["--test-dir", args.test_dir])
    if args.list_milestones:
        cmd.append("--list-milestones")
    if args.no_stage:
        cmd.append("--no-stage")
    if args.test_only:
        cmd.append("--test-only")
    
    # Forward the otr_run plugin explicitly since conftest.py is deleted
    cmd.extend(["-p", "otr_run"])
    
    # Forward any other unknown arguments to pytest
    cmd.extend(unknown)

    # Set PYTHONPATH to include current dir so pytest can load the 'otr_run' module as a plugin
    env = os.environ.copy()
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
