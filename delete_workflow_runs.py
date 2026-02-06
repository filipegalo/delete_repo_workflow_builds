#!/usr/bin/env python3
"""
gh-cleanup — Delete GitHub Actions workflow runs in bulk.

Supports interactive and fully non-interactive (scriptable) modes.
Uses the GitHub CLI (gh) for authentication and API access.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

VERSION = "2.0.0"
PAGE_SIZE = 100
DEFAULT_WORKERS = 5
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
CANCEL_TIMEOUT = 60
CANCEL_POLL_INTERVAL = 2


# ── Terminal Styling ──────────────────────────────────────────────────────


class Style:
    """ANSI styling with automatic detection. Respects NO_COLOR convention."""

    _enabled: bool = (
        hasattr(sys.stderr, "isatty")
        and sys.stderr.isatty()
        and os.environ.get("NO_COLOR") is None
    )

    BOLD = "\033[1m" if _enabled else ""
    DIM = "\033[2m" if _enabled else ""
    RED = "\033[31m" if _enabled else ""
    GREEN = "\033[32m" if _enabled else ""
    YELLOW = "\033[33m" if _enabled else ""
    BLUE = "\033[34m" if _enabled else ""
    CYAN = "\033[36m" if _enabled else ""
    RESET = "\033[0m" if _enabled else ""

    @classmethod
    def success(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.RESET}"

    @classmethod
    def error(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.RESET}"

    @classmethod
    def warn(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.RESET}"

    @classmethod
    def info(cls, text: str) -> str:
        return f"{cls.CYAN}{text}{cls.RESET}"

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.RESET}"

    @classmethod
    def dim(cls, text: str) -> str:
        return f"{cls.DIM}{text}{cls.RESET}"


# ── Exceptions ────────────────────────────────────────────────────────────


class GitHubCLIError(Exception):
    """Raised when a gh CLI command fails."""

    def __init__(self, command: str, stderr: str, returncode: int):
        self.command = command
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f"gh failed (exit {returncode}): {stderr}")


# ── GitHub API Layer ──────────────────────────────────────────────────────


def gh_api(
    endpoint: str,
    method: str = "GET",
    retries: int = MAX_RETRIES,
) -> str:
    """Execute a gh api call with retry and rate-limit backoff."""
    cmd = ["gh", "api"]
    if method != "GET":
        cmd.extend(["-X", method])
    cmd.append(endpoint)

    last_error: Optional[GitHubCLIError] = None

    for attempt in range(retries):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            last_error = GitHubCLIError(" ".join(cmd), stderr, exc.returncode)

            is_rate_limit = "rate limit" in stderr.lower()
            is_server_error = any(
                code in stderr for code in ("500", "502", "503")
            )

            if is_rate_limit or is_server_error:
                delay = RETRY_BASE_DELAY * (2**attempt)
                if is_rate_limit:
                    print(
                        Style.warn(f"  Rate limited. Retrying in {delay:.0f}s..."),
                        file=sys.stderr,
                    )
                time.sleep(delay)
                continue

            raise last_error

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unexpected: no attempts were made")


def gh_api_paginated(endpoint: str, key: str) -> List[Dict]:
    """Fetch all pages from a paginated GitHub API endpoint.

    Uses manual pagination to avoid the gh --paginate + --jq
    concatenation bug that produces invalid JSON on multi-page results.
    """
    items: List[Dict] = []
    page = 1

    while True:
        separator = "&" if "?" in endpoint else "?"
        url = f"{endpoint}{separator}per_page={PAGE_SIZE}&page={page}"
        raw = gh_api(url)

        if not raw:
            break

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GitHubCLIError(
                f"gh api {endpoint}", f"Invalid JSON response: {exc}", 0
            ) from exc

        page_items = data.get(key, [])

        if not page_items:
            break

        items.extend(page_items)

        if len(page_items) < PAGE_SIZE:
            break

        page += 1

    return items


# ── Repository Parsing ────────────────────────────────────────────────────


def parse_repo(repo_input: str) -> Tuple[str, str]:
    """Parse a GitHub repository identifier into (owner, repo).

    Accepts:
      - owner/repo
      - https://github.com/owner/repo
      - git@github.com:owner/repo.git
    """
    text = repo_input.strip()
    owner, repo = "", ""

    # SSH format
    if text.startswith("git@"):
        path = text.split(":", 1)[-1]
        path = path.removesuffix(".git")
        parts = path.split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]

    # HTTPS URL
    elif text.startswith(("http://", "https://")):
        parsed = urlparse(text)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            owner, repo = path_parts[0], path_parts[1].removesuffix(".git")

    # owner/repo
    else:
        parts = text.split("/")
        if len(parts) == 2 and all(parts):
            owner, repo = parts[0], parts[1]

    if not owner or not repo:
        raise ValueError(
            f"Cannot parse repository: '{repo_input}'. "
            f"Expected 'owner/repo' or a GitHub URL."
        )

    valid_pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
    if not valid_pattern.match(owner) or not valid_pattern.match(repo):
        raise ValueError(
            f"Invalid characters in repository: '{owner}/{repo}'. "
            f"Only alphanumerics, dots, hyphens, and underscores are allowed."
        )

    return owner, repo


# ── Workflow Operations ───────────────────────────────────────────────────


def fetch_workflows(owner: str, repo: str) -> List[Dict]:
    """Fetch all workflows for a repository."""
    return gh_api_paginated(
        f"repos/{owner}/{repo}/actions/workflows", "workflows"
    )


def fetch_workflow_runs(
    owner: str,
    repo: str,
    workflow_id: int,
    branch: Optional[str] = None,
    actor: Optional[str] = None,
) -> List[Dict]:
    """Fetch all runs for a specific workflow with optional server-side filters."""
    endpoint = f"repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
    params = []
    if branch:
        params.append(f"branch={quote(branch, safe='')}")
    if actor:
        params.append(f"actor={quote(actor, safe='')}")

    if params:
        endpoint += "?" + "&".join(params)

    return gh_api_paginated(endpoint, "workflow_runs")


def cancel_run(owner: str, repo: str, run_id: int) -> bool:
    """Cancel a workflow run. Returns True on success."""
    try:
        gh_api(
            f"repos/{owner}/{repo}/actions/runs/{run_id}/cancel",
            method="POST",
        )
        return True
    except GitHubCLIError:
        return False


def delete_run(owner: str, repo: str, run_id: int) -> bool:
    """Delete a workflow run. Returns True on success."""
    try:
        gh_api(
            f"repos/{owner}/{repo}/actions/runs/{run_id}",
            method="DELETE",
        )
        return True
    except GitHubCLIError:
        return False


def get_run_status(owner: str, repo: str, run_id: int) -> Optional[str]:
    """Get the current status of a workflow run."""
    try:
        raw = gh_api(f"repos/{owner}/{repo}/actions/runs/{run_id}")
        data = json.loads(raw)
        return data.get("status")
    except (GitHubCLIError, json.JSONDecodeError):
        return None


def wait_for_cancellation(
    owner: str,
    repo: str,
    run_id: int,
    timeout: int = CANCEL_TIMEOUT,
    interval: int = CANCEL_POLL_INTERVAL,
) -> bool:
    """Wait for a run to reach a terminal state after cancellation."""
    elapsed = 0
    consecutive_failures = 0

    while elapsed < timeout:
        status = get_run_status(owner, repo, run_id)

        if status in ("completed", "cancelled"):
            return True

        if status is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print(
                    Style.warn(f"  Could not verify status for run {run_id}"),
                    file=sys.stderr,
                )
                return False
        else:
            consecutive_failures = 0

        time.sleep(interval)
        elapsed += interval

    return False


# ── Filtering ─────────────────────────────────────────────────────────────


def parse_run_time(run: Dict) -> Optional[datetime]:
    """Parse the created_at timestamp from a workflow run."""
    created_at = run.get("created_at", "")
    if not created_at:
        return None
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def filter_runs(
    runs: List[Dict],
    cutoff: Optional[datetime] = None,
    conclusions: Optional[List[str]] = None,
    keep_last: int = 0,
) -> List[Dict]:
    """Filter workflow runs by time and conclusion.

    Returns a new list — never mutates the input.
    Runs are sorted newest-first before applying keep_last.
    """
    sorted_runs = sorted(
        runs,
        key=lambda r: r.get("created_at", ""),
        reverse=True,
    )

    if keep_last > 0:
        sorted_runs = sorted_runs[keep_last:]

    filtered = []
    for run in sorted_runs:
        if cutoff is not None:
            run_time = parse_run_time(run)
            if run_time is not None and run_time < cutoff:
                continue

        if conclusions:
            run_conclusion = run.get("conclusion") or ""
            if run_conclusion not in conclusions:
                continue

        filtered.append(run)

    return filtered


# ── Interactive Prompts ───────────────────────────────────────────────────


def prompt_repo() -> str:
    """Interactively ask for repository."""
    repo = input(
        f"\n{Style.bold('Repository')} (owner/repo or URL): "
    ).strip()
    if not repo:
        print(Style.error("Repository is required."))
        sys.exit(1)
    return repo


def prompt_select_workflows(workflows: List[Dict]) -> List[Dict]:
    """Interactively select which workflows to target."""
    print(f"\n{Style.bold('Available workflows:')}")
    print(Style.dim("-" * 50))

    for i, wf in enumerate(workflows, 1):
        name = wf.get("name", f"ID: {wf['id']}")
        state = wf.get("state", "unknown")
        badge = Style.success(state) if state == "active" else Style.dim(state)
        print(f"  {Style.bold(str(i))}. {name}  {badge}")

    print(f"  {Style.bold(str(len(workflows) + 1))}. {Style.info('All workflows')}")

    raw = input(
        f"\n{Style.bold('Select')} (comma-separated, e.g. 1,3): "
    ).strip()

    if raw == str(len(workflows) + 1) or raw.lower() == "all":
        return list(workflows)

    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        selected = [workflows[i] for i in indices if 0 <= i < len(workflows)]
        if selected:
            return selected
    except (ValueError, IndexError):
        pass

    print(Style.warn("Invalid selection — using all workflows."))
    return list(workflows)


def prompt_timeframe() -> Optional[datetime]:
    """Interactively select a time-frame cutoff."""
    presets = {
        "1": ("All time", None),
        "2": ("Last 24 hours", 1),
        "3": ("Last 7 days", 7),
        "4": ("Last 30 days", 30),
        "5": ("Last 90 days", 90),
        "6": ("Custom", -1),
    }

    print(f"\n{Style.bold('Time frame:')}")
    print(Style.dim("-" * 50))
    for key, (label, _) in presets.items():
        print(f"  {Style.bold(key)}. {label}")

    choice = input(f"\n{Style.bold('Select')} (1-6): ").strip()

    if choice not in presets:
        print(Style.warn("Invalid choice — using all time."))
        return None

    _, days = presets[choice]
    if days is None:
        return None

    if days == -1:
        try:
            days = int(input("  Enter number of days: ").strip())
        except ValueError:
            print(Style.warn("Invalid input — using all time."))
            return None

    return datetime.now(timezone.utc) - timedelta(days=days)


CONCLUSION_CHOICES = [
    "success",
    "failure",
    "cancelled",
    "timed_out",
    "action_required",
    "skipped",
]


def prompt_conclusion_filter() -> Optional[List[str]]:
    """Interactively select conclusion filter."""
    print(f"\n{Style.bold('Filter by conclusion:')}")
    print(Style.dim("-" * 50))
    print(f"  {Style.bold('0')}. {Style.info('All conclusions (no filter)')}")
    for i, c in enumerate(CONCLUSION_CHOICES, 1):
        print(f"  {Style.bold(str(i))}. {c}")

    raw = input(
        f"\n{Style.bold('Select')} (comma-separated, or 0 for all): "
    ).strip()

    if raw == "0" or not raw:
        return None

    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        selected = [
            CONCLUSION_CHOICES[i]
            for i in indices
            if 0 <= i < len(CONCLUSION_CHOICES)
        ]
        if selected:
            return selected
    except (ValueError, IndexError):
        pass

    return None


def confirm_action(message: str) -> bool:
    """Ask for yes/no confirmation."""
    response = input(f"\n{Style.warn(message)} (yes/no): ").strip().lower()
    return response == "yes"


# ── Output Helpers ────────────────────────────────────────────────────────


def print_banner(owner: str, repo: str, dry_run: bool = False) -> None:
    """Print the tool header."""
    line = Style.dim("=" * 58)
    print(f"\n{line}")
    print(
        f"  {Style.bold(Style.info('gh-cleanup'))} {Style.dim(f'v{VERSION}')}"
    )
    print(f"  {owner}/{repo}")
    if dry_run:
        print(f"  {Style.warn('DRY RUN — no changes will be made')}")
    print(line)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Style.bold(title)}")
    print(Style.dim("-" * 58))


def format_conclusion(conclusion: str) -> str:
    """Color-code a run conclusion."""
    if conclusion == "success":
        return Style.success(conclusion)
    if conclusion in ("failure", "timed_out"):
        return Style.error(conclusion)
    if conclusion == "cancelled":
        return Style.warn(conclusion)
    return Style.dim(conclusion or "-")


def print_run_line(
    run: Dict, index: int = 0, total: int = 0, suffix: str = ""
) -> None:
    """Print a formatted workflow run line."""
    name = (run.get("display_title") or run.get("name") or "Unknown")[:42]
    conclusion = run.get("conclusion") or run.get("status") or "?"
    branch = (run.get("head_branch") or "?")[:16]
    created = (run.get("created_at") or "?")[:10]

    counter = f"[{index}/{total}]" if total > 0 else ""
    conclusion_str = format_conclusion(conclusion)

    print(
        f"  {Style.dim(counter):>10} {name:<42} "
        f"{conclusion_str:<20} {Style.dim(branch):<18} "
        f"{Style.dim(created)} {suffix}"
    )


def print_summary(
    succeeded: int, failed: int, elapsed: float, workers: int
) -> None:
    """Print the final summary."""
    line = Style.dim("=" * 58)
    print(f"\n{line}")
    print(f"  {Style.bold('Summary')}")
    print(f"  {Style.success(f'Deleted:  {succeeded}')}")
    if failed > 0:
        print(f"  {Style.error(f'Failed:   {failed}')}")
    print(f"  {Style.dim(f'Duration: {elapsed:.1f}s ({workers} workers)')}")
    print(line)


# ── CLI Argument Parser ───────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all supported flags."""
    parser = argparse.ArgumentParser(
        prog="gh-cleanup",
        description="Delete GitHub Actions workflow runs — interactively or via flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s owner/repo                           Interactive mode
  %(prog)s owner/repo --days 30 -y              Delete runs from last 30 days
  %(prog)s owner/repo -w "CI" --dry-run         Preview deletions for CI workflow
  %(prog)s owner/repo --conclusion failure -y   Delete only failed runs
  %(prog)s owner/repo --keep-last 5 -y          Keep 5 most recent per workflow
  %(prog)s owner/repo -b main --days 7 -y       Delete runs on main from last week
""",
    )

    parser.add_argument(
        "repo",
        nargs="?",
        help="Repository (owner/repo or GitHub URL)",
    )
    parser.add_argument(
        "-w",
        "--workflow",
        action="append",
        dest="workflows",
        help="Workflow name or ID (repeatable). Omit for all.",
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        help="Only target runs created within the last N days",
    )
    parser.add_argument(
        "--conclusion",
        action="append",
        dest="conclusions",
        choices=[
            "success",
            "failure",
            "cancelled",
            "timed_out",
            "action_required",
            "skipped",
            "stale",
            "startup_failure",
        ],
        help="Filter by conclusion (repeatable). Omit for all.",
    )
    parser.add_argument(
        "-b",
        "--branch",
        help="Filter runs by branch name",
    )
    parser.add_argument(
        "--actor",
        help="Filter runs by the user who triggered them",
    )
    parser.add_argument(
        "-k",
        "--keep-last",
        type=int,
        default=0,
        help="Keep the N most recent runs per workflow (default: 0)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel deletion workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without making changes",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--no-cancel",
        action="store_true",
        help="Skip cancelling active runs before deletion",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    return parser


# ── Main ──────────────────────────────────────────────────────────────────


def resolve_workflows(
    all_workflows: List[Dict],
    targets: Optional[List[str]],
    interactive: bool,
) -> List[Dict]:
    """Resolve the target workflows from CLI flags or interactive prompt."""
    if targets:
        selected = []

        for target in targets:
            matches = [
                wf
                for wf in all_workflows
                if str(wf["id"]) == target
                or wf.get("name", "").lower() == target.lower()
            ]
            if matches:
                selected.extend(matches)
            else:
                print(Style.warn(f"  Workflow not found: '{target}' — skipping"))

        if not selected:
            print(Style.error("No matching workflows found."))
            sys.exit(1)

        return selected

    if interactive:
        return prompt_select_workflows(all_workflows)

    return list(all_workflows)


def resolve_cutoff(
    days: Optional[int], interactive: bool
) -> Optional[datetime]:
    """Resolve the time cutoff from CLI flag or interactive prompt."""
    if days is not None:
        return datetime.now(timezone.utc) - timedelta(days=days)
    if interactive:
        return prompt_timeframe()
    return None


def resolve_conclusions(
    conclusions: Optional[List[str]], interactive: bool
) -> Optional[List[str]]:
    """Resolve conclusion filter from CLI flag or interactive prompt."""
    if conclusions is not None:
        return conclusions
    if interactive:
        return prompt_conclusion_filter()
    return None


def cancel_active_runs(
    owner: str, repo: str, runs: List[Dict]
) -> None:
    """Cancel any in-progress or queued runs, then wait for completion."""
    active_statuses = ("in_progress", "queued", "waiting", "pending")
    active_runs = [r for r in runs if r.get("status") in active_statuses]

    if not active_runs:
        return

    print_section(f"Cancelling {len(active_runs)} active run(s)")

    for run in active_runs:
        name = (run.get("display_title") or run.get("name") or "?")[:42]
        print(f"  {name}...", end=" ", flush=True)
        if cancel_run(owner, repo, run["id"]):
            print(Style.success("cancelled"))
        else:
            print(Style.dim("skipped"))

    print(Style.dim("  Waiting for cancellations to settle..."))
    for run in active_runs:
        wait_for_cancellation(owner, repo, run["id"])


def delete_runs_parallel(
    owner: str,
    repo: str,
    runs: List[Dict],
    workers: int,
) -> Tuple[int, int]:
    """Delete runs concurrently. Returns (succeeded, failed) counts."""
    total = len(runs)
    succeeded = 0
    failed = 0

    def _delete(run: Dict) -> Tuple[Dict, bool]:
        return run, delete_run(owner, repo, run["id"])

    effective_workers = min(workers, total)

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(_delete, run): run for run in runs}

        for i, future in enumerate(as_completed(futures), 1):
            run = futures[future]
            try:
                _, ok = future.result()
            except Exception:
                ok = False

            if ok:
                succeeded += 1
                suffix = Style.success("deleted")
            else:
                failed += 1
                suffix = Style.error("failed")

            print_run_line(run, index=i, total=total, suffix=suffix)

    return succeeded, failed


def main() -> int:
    """Entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve repository ────────────────────────────────────────────
    if args.parallel < 1:
        print(Style.error("--parallel must be at least 1"))
        return 1
    if args.keep_last < 0:
        print(Style.error("--keep-last must be 0 or greater"))
        return 1

    repo_input = args.repo or prompt_repo()

    try:
        owner, repo = parse_repo(repo_input)
    except ValueError as exc:
        print(Style.error(str(exc)))
        return 1

    interactive = not (
        args.yes
        or args.workflows
        or args.days is not None
        or args.conclusions
    )

    print_banner(owner, repo, dry_run=args.dry_run)

    # ── Fetch workflows ───────────────────────────────────────────────
    print(Style.info("Fetching workflows..."))

    try:
        all_workflows = fetch_workflows(owner, repo)
    except GitHubCLIError as exc:
        print(Style.error(f"Failed to fetch workflows: {exc.stderr}"))
        return 1

    if not all_workflows:
        print(Style.warn("No workflows found in this repository."))
        return 0

    print(f"Found {Style.bold(str(len(all_workflows)))} workflow(s)")

    # ── Resolve filters ───────────────────────────────────────────────
    selected = resolve_workflows(all_workflows, args.workflows, interactive)
    cutoff = resolve_cutoff(args.days, interactive)
    conclusions = resolve_conclusions(args.conclusions, interactive)

    # ── Print active filters ──────────────────────────────────────────
    print_section("Filters")
    wf_names = ", ".join(
        w.get("name", str(w["id"])) for w in selected
    )
    cutoff_label = (
        cutoff.strftime("%Y-%m-%d %H:%M UTC") if cutoff else "all time"
    )
    conclusion_label = ", ".join(conclusions) if conclusions else "all"

    print(f"  Workflows:   {wf_names}")
    print(f"  Time range:  {cutoff_label}")
    print(f"  Conclusions: {conclusion_label}")
    if args.branch:
        print(f"  Branch:      {args.branch}")
    if args.actor:
        print(f"  Actor:       {args.actor}")
    if args.keep_last > 0:
        print(f"  Keep last:   {args.keep_last} per workflow")

    # ── Collect and filter runs ───────────────────────────────────────
    print_section("Collecting runs")
    all_runs: List[Dict] = []

    for wf in selected:
        wf_name = wf.get("name", str(wf["id"]))
        print(f"  {Style.info(wf_name)}...", end=" ", flush=True)

        try:
            runs = fetch_workflow_runs(
                owner,
                repo,
                wf["id"],
                branch=args.branch,
                actor=args.actor,
            )
        except GitHubCLIError as exc:
            print(Style.error(f"error: {exc.stderr}"))
            continue

        filtered = filter_runs(
            runs,
            cutoff=cutoff,
            conclusions=conclusions,
            keep_last=args.keep_last,
        )

        print(
            f"{Style.bold(str(len(filtered)))} of {len(runs)} runs matched"
        )
        all_runs.extend(filtered)

    if not all_runs:
        print(Style.warn("\nNo runs match the specified criteria."))
        return 0

    total = len(all_runs)
    print(f"\n  {Style.bold(str(total))} total run(s) to process")

    # ── Dry run ───────────────────────────────────────────────────────
    if args.dry_run:
        print_section("Dry Run Preview")
        for i, run in enumerate(all_runs, 1):
            print_run_line(run, index=i, total=total)
        print(f"\n{Style.info('Dry run complete.')} No runs were deleted.")
        return 0

    # ── Confirm ───────────────────────────────────────────────────────
    if not args.yes:
        if not confirm_action(f"Permanently delete {total} workflow run(s)?"):
            print("Cancelled.")
            return 0

    # ── Cancel active runs ────────────────────────────────────────────
    if not args.no_cancel:
        cancel_active_runs(owner, repo, all_runs)

    # ── Delete ────────────────────────────────────────────────────────
    print_section(f"Deleting {total} run(s)")

    start_time = time.monotonic()
    succeeded, failed = delete_runs_parallel(
        owner, repo, all_runs, args.parallel
    )
    elapsed = time.monotonic() - start_time

    print_summary(succeeded, failed, elapsed, min(args.parallel, total))

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(Style.warn("\nInterrupted."))
        sys.exit(130)
    except GitHubCLIError as exc:
        print(Style.error(f"\nGitHub API error: {exc}"))
        sys.exit(1)
