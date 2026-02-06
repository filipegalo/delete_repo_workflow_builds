# gh-cleanup

Delete GitHub Actions workflow runs in bulk — interactively or from a single command.

## Features

- **Interactive & scriptable** — full CLI with flags for CI/CD pipelines, or guided interactive mode
- **Dry run** — preview exactly what would be deleted before committing
- **Parallel deletion** — concurrent workers for fast bulk operations
- **Smart filtering** — by workflow, time range, conclusion, branch, actor
- **Keep last N** — protect recent runs while cleaning old ones
- **Auto-cancel** — cancels active runs before deletion so nothing is skipped
- **Retry with backoff** — handles GitHub rate limits and transient errors automatically
- **Color output** — ANSI colors with `NO_COLOR` support for piping

## Prerequisites

- **Python 3.10+**
- **GitHub CLI** (`gh`) installed and authenticated (`gh auth login`)
- Permission to delete workflow runs in the target repository

## Installation

```bash
git clone <this-repo>
cd delete_repo_workflow_builds
chmod +x delete_workflow_runs.py
```

## Quick Start

```bash
# Interactive mode — walks you through every option
./delete_workflow_runs.py owner/repo

# One-liner: delete all failed runs from the last 30 days
./delete_workflow_runs.py owner/repo --conclusion failure --days 30 -y

# Preview what would be deleted (no changes made)
./delete_workflow_runs.py owner/repo --dry-run

# Keep the 10 most recent runs per workflow, delete everything else
./delete_workflow_runs.py owner/repo --keep-last 10 -y
```

## Usage

```
gh-cleanup [-h] [-w WORKFLOW] [-d DAYS] [--conclusion CONCLUSION]
           [-b BRANCH] [--actor ACTOR] [-k KEEP_LAST] [-p PARALLEL]
           [--dry-run] [-y] [--no-cancel] [--version]
           [repo]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `repo` | Repository in `owner/repo`, HTTPS URL, or SSH URL format. Prompted if omitted. |

### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--workflow` | `-w` | Target a specific workflow by name or ID. Repeatable. |
| `--days` | `-d` | Only target runs created within the last N days. |
| `--conclusion` | | Filter by conclusion: `success`, `failure`, `cancelled`, `timed_out`, `action_required`, `skipped`, `stale`, `startup_failure`. Repeatable. |
| `--branch` | `-b` | Filter runs by branch name (server-side). |
| `--actor` | | Filter runs by the user who triggered them (server-side). |
| `--keep-last` | `-k` | Keep the N most recent runs per workflow. Default: 0 (keep none). |
| `--parallel` | `-p` | Number of parallel deletion workers. Default: 5. |
| `--dry-run` | | Preview what would be deleted without making changes. |
| `--yes` | `-y` | Skip the confirmation prompt. |
| `--no-cancel` | | Skip cancelling active runs before deletion. |
| `--version` | | Show version and exit. |

### Interactive vs Non-Interactive Mode

The tool enters **interactive mode** when no filtering flags are provided — it walks you through workflow selection, time frame, and conclusion filters with a guided menu.

When any of `--yes`, `--workflow`, `--days`, or `--conclusion` are provided, the tool runs in **non-interactive mode**, which is ideal for scripts and CI.

## Examples

```bash
# Delete only cancelled runs
./delete_workflow_runs.py owner/repo --conclusion cancelled -y

# Delete failed + timed-out runs from the CI workflow on main branch
./delete_workflow_runs.py owner/repo -w "CI" --conclusion failure --conclusion timed_out -b main -y

# Dry run: see what the "Deploy" workflow cleanup would look like
./delete_workflow_runs.py owner/repo -w "Deploy" --days 90 --dry-run

# Clean up everything older than 7 days, keep 3 most recent per workflow
./delete_workflow_runs.py owner/repo --days 7 --keep-last 3 -y

# Use with a GitHub URL
./delete_workflow_runs.py https://github.com/owner/repo --days 30 -y

# Use with an SSH URL
./delete_workflow_runs.py git@github.com:owner/repo.git --days 30 -y

# Speed up deletion with more workers
./delete_workflow_runs.py owner/repo -y -p 10

# Pipe-friendly (no colors)
NO_COLOR=1 ./delete_workflow_runs.py owner/repo --days 30 -y
```

## How It Works

1. **Fetches** all workflows from the repository via the GitHub API
2. **Resolves** which workflows to target (from flags or interactive selection)
3. **Collects** all runs, applying server-side filters (branch, actor) and client-side filters (time range, conclusion, keep-last)
4. **Previews** the runs (in dry-run mode) or asks for confirmation
5. **Cancels** any active runs (in-progress/queued) and waits for them to settle
6. **Deletes** all matching runs in parallel with progress output
7. **Summarizes** results: deleted count, failures, and elapsed time

## Important Notes

- Deleted workflow runs **cannot be recovered**
- Uses your existing `gh` authentication — no tokens to configure
- Handles GitHub API rate limits with automatic exponential backoff
- `--keep-last` sorts runs newest-first per workflow before skipping
- Exits with code 1 if any deletions fail, 0 otherwise

## Troubleshooting

### "gh failed" errors

- Verify authentication: `gh auth status`
- Check permissions: you need write access to Actions in the repository
- Update GitHub CLI: `gh version` (needs 2.0+)

### Rate limiting

The tool automatically retries with exponential backoff when rate-limited. For very large repositories, consider using `--days` or `--workflow` to reduce the number of API calls per run.

### "No workflows found"

- Verify the repository has GitHub Actions workflows (`.github/workflows/` directory)
- Check that you can access the repository: `gh repo view owner/repo`

## License

MIT
