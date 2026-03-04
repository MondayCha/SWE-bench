#!/usr/bin/env python3
"""
build_all_images.py
===================
Bulk-build Docker images for an SWE-bench dataset, optionally push them to a
private registry, and remove local copies to reclaim disk space.

Instances are sorted first by *repo* (to maximise Docker layer reuse within
the same repository), then by *instance_id*.  After every instance in a given
repo has been processed the corresponding instance-level and environment-level
images are pushed (when ``--registry`` is given) and deleted from the local
Docker daemon.

Usage examples
--------------
# 1. Dry-run on the full Verified set — no Docker calls, just log what would happen
python build_all_images.py --dry-run

# 2. Build everything and keep images locally (no push)
python build_all_images.py

# 3. Build, push to a private registry with a date tag, then clean up locals
python build_all_images.py \\
    --registry registry.example.ai/agentbox \\
    --tag 20260304

# 4. Build only two specific instances
python build_all_images.py \\
    --instance-ids astropy__astropy-12907 astropy__astropy-13033

# 5. Retry the failures from a previous run
python build_all_images.py \\
    --registry registry.example.ai/agentbox \\
    --tag 20260304 \\
    --retry-from failed_builds.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import docker
import docker.errors

from swebench.harness.constants import (
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_env_images,
    build_instance_image,
)
from swebench.harness.docker_utils import remove_image
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import load_swebench_dataset


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_main_logger(log_dir: Path) -> logging.Logger:
    """
    Create the main application logger.

    * Console  — INFO level, human-readable output on stdout.
    * Log file — DEBUG level, timestamped file inside *log_dir* that captures
      every detail for post-mortem analysis.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_all_images")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -- stdout handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # -- file handler (DEBUG captures everything)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"build_{ts}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Print log file path so users know where to look
    # (use print here because the logger itself is not fully set up yet)
    print(f"[build_all_images] Log file: {log_file}", flush=True)

    return logger


# ---------------------------------------------------------------------------
# Checkpoint / state file helpers
# ---------------------------------------------------------------------------


def load_state(state_file: Path, logger: logging.Logger) -> set[str]:
    """
    Load the set of already-succeeded instance IDs from *state_file*.

    Returns an empty set if the file does not exist or cannot be parsed.
    """
    if not state_file.exists():
        logger.info("State file '%s' not found — starting fresh.", state_file)
        return set()
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        succeeded: list[str] = data.get("succeeded", [])
        logger.info(
            "Loaded state from '%s': %d already-succeeded instance(s) will be skipped.",
            state_file,
            len(succeeded),
        )
        return set(succeeded)
    except Exception as exc:
        logger.warning(
            "Could not parse state file '%s' (%s) — ignoring and starting fresh.",
            state_file,
            exc,
        )
        return set()


def save_state(
    state_file: Path,
    succeeded: set[str],
    logger: logging.Logger,
) -> None:
    """
    Atomically write the accumulated succeeded-instance set to *state_file*.

    Uses a sibling ``.tmp`` file + os.replace so a crash mid-write cannot
    corrupt the previously saved state.
    """
    data = {
        "succeeded": sorted(succeeded),
        "count": len(succeeded),
        "updated_at": datetime.now().isoformat(),
    }
    tmp = state_file.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(state_file)
        logger.debug(
            "State saved: %d succeeded instance(s) → '%s'", len(succeeded), state_file
        )
    except Exception as exc:
        logger.warning("Failed to save state to '%s': %s", state_file, exc)


# ---------------------------------------------------------------------------
# Real-time Docker build log streaming
# ---------------------------------------------------------------------------


class BuildLogTailer:
    """
    Tail the Docker build log written by ``swebench.harness.docker_build.build_image``
    and forward each new line to a logger in real time.

    The tailer runs in a background daemon thread.  Call ``start()`` before
    invoking the build function, and ``stop()`` once it returns.

    The log file is created *inside* ``build_image()`` (``open(path, 'w')``),
    so the tailer will wait up to *file_wait_s* seconds for it to appear.
    If the image already exists, ``build_image`` is never called and the file
    will not be created; the tailer exits gracefully after the timeout.
    """

    def __init__(
        self,
        log_file: Path,
        logger: logging.Logger,
        instance_id: str,
        file_wait_s: float = 30.0,
    ) -> None:
        self._log_file = log_file
        self._logger = logger
        self._instance_id = instance_id
        self._file_wait_s = file_wait_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"logtail-{instance_id}",
        )

    def start(self) -> "BuildLogTailer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        # Wait for the log file to be created by build_image()
        deadline = time.monotonic() + self._file_wait_s
        while not self._log_file.exists():
            if self._stop.is_set() or time.monotonic() > deadline:
                self._logger.debug(
                    "[logtail/%s] Log file did not appear within %.0fs — "
                    "image may already exist or build was skipped.",
                    self._instance_id,
                    self._file_wait_s,
                )
                return
            time.sleep(0.15)

        self._logger.debug(
            "[logtail/%s] Tailing: %s", self._instance_id, self._log_file
        )
        try:
            with open(self._log_file, "r", encoding="utf-8", errors="replace") as fh:
                while not self._stop.is_set():
                    line = fh.readline()
                    if line:
                        stripped = line.rstrip("\n")
                        if stripped:
                            self._logger.info("  │docker│ %s", stripped)
                    else:
                        time.sleep(0.2)
                # Drain any lines written after stop() was called
                for line in fh:
                    stripped = line.rstrip("\n")
                    if stripped:
                        self._logger.info("  │docker│ %s", stripped)
        except Exception as exc:
            self._logger.debug(
                "[logtail/%s] Error reading log: %s", self._instance_id, exc
            )


# ---------------------------------------------------------------------------
# Registry / tagging helpers
# ---------------------------------------------------------------------------


def get_registry_image_name(local_image: str, registry: str, tag: str) -> str:
    """
    Map a local image name to its registry-tagged counterpart.

    The existing tag (if any) is stripped and replaced with *tag*.  The
    registry prefix is prepended before the image name.

    Example
    -------
    >>> get_registry_image_name(
    ...     "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
    ...     "registry.example.ai/agentbox",
    ...     "20260304",
    ... )
    "registry.example.ai/agentbox/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:20260304"
    """
    name_part = local_image.split(":")[0] if ":" in local_image else local_image
    return f"{registry.rstrip('/')}/{name_part}:{tag}"


def tag_and_push_image(
    client: docker.DockerClient,
    local_image_name: str,
    registry_image_name: str,
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    """
    Tag *local_image_name* as *registry_image_name* and push it to the registry.

    Returns ``True`` on success, ``False`` on any failure.
    The function never raises — errors are logged and ``False`` is returned.
    """
    if dry_run:
        logger.info("[DRY RUN] tag  %s  →  %s", local_image_name, registry_image_name)
        logger.info("[DRY RUN] push %s", registry_image_name)
        return True

    repo_part, tag_part = registry_image_name.rsplit(":", 1)

    # -- Tag
    try:
        logger.info(
            "Tagging  %s  →  %s ...", local_image_name, registry_image_name
        )
        image = client.images.get(local_image_name)
        image.tag(repo_part, tag=tag_part)
        logger.info("Tag applied successfully.")
    except Exception as exc:
        logger.error("Failed to tag '%s': %s", local_image_name, exc)
        logger.debug(traceback.format_exc())
        return False

    # -- Push
    try:
        logger.info("Pushing %s …", registry_image_name)
        t_push = time.monotonic()
        layers_done = 0
        for line in client.api.push(
            repo_part, tag=tag_part, stream=True, decode=True
        ):
            if "error" in line:
                logger.error("  [push] Error: %s", line["error"])
                return False
            status = line.get("status", "")
            layer_id = line.get("id", "")
            detail = line.get("progressDetail", {})
            if status in ("Layer already exists", "Pushed", "Mounted from"):
                layers_done += 1
                logger.info("  [push] %-20s  %s", status, layer_id)
            elif status == "Pushing" and detail:
                pct = ""
                current = detail.get("current", 0)
                total = detail.get("total", 0)
                if total:
                    pct = f"  {100*current//total}%"
                logger.debug("  [push] %-20s  %s%s", status, layer_id, pct)
            elif status:
                logger.debug("  [push] %s %s", status, layer_id)
        elapsed_push = time.monotonic() - t_push
        logger.info(
            "Push succeeded: %s  (%.1fs, %d layers processed)",
            registry_image_name, elapsed_push, layers_done,
        )
        return True
    except Exception as exc:
        logger.error("Failed to push '%s': %s", registry_image_name, exc)
        logger.debug(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def cleanup_repo_images(
    client: docker.DockerClient,
    specs: list[TestSpec],
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    """
    Remove instance images and their shared environment images for *specs*.

    Base images (``sweb.base.*``) are intentionally left untouched because they
    are shared across many repositories and cheap to reuse.
    """
    # Collect unique env-image keys for this batch
    env_keys = {s.env_image_key for s in specs}

    for spec in specs:
        if dry_run:
            logger.info(
                "[DRY RUN] Would remove instance image: %s", spec.instance_image_key
            )
        else:
            logger.info("Removing instance image: %s", spec.instance_image_key)
            remove_image(client, spec.instance_image_key, logger)

    for key in sorted(env_keys):
        if dry_run:
            logger.info("[DRY RUN] Would remove env image: %s", key)
        else:
            logger.info("Removing env image: %s", key)
            remove_image(client, key, logger)


# ---------------------------------------------------------------------------
# Core per-repo build / push logic
# ---------------------------------------------------------------------------


def process_repo(
    *,
    repo: str,
    instances: list,
    client: Optional[docker.DockerClient],
    logger: logging.Logger,
    namespace: Optional[str],
    registry: Optional[str],
    tag: str,
    dry_run: bool,
    max_workers: int,
    already_succeeded: set[str],
    state_file: Optional[Path],
    state_succeeded: set[str],
    verbose_build: bool,
) -> tuple[list[str], list[str]]:
    """
    Orchestrate the full lifecycle for all instances belonging to *repo*:

    1. Skip instances already recorded as succeeded in *already_succeeded*.
    2. Build base + environment images (shared within the repo).
    3. For each instance image (sequential):
       a. Build it.
       b. Immediately tag and push (when ``--registry`` is set).
       c. Write to the state file so progress survives a crash/restart.
    4. After *all* builds in the repo are done, remove local instance and
       environment images to free disk space.

    Parameters
    ----------
    repo : str
        Repository name, e.g. ``"astropy/astropy"``.
    instances : list
        Raw SWE-bench instance dicts (pre-sorted by instance_id).
    client : docker.DockerClient | None
        Live Docker client, or ``None`` in dry-run mode.
    logger : logging.Logger
        Main application logger.
    namespace : str | None
        Image namespace prefix (e.g. ``"swebench"``).
    registry : str | None
        Optional registry prefix for pushing.
    tag : str
        Tag applied to pushed registry images.
    dry_run : bool
        When ``True``, log every action without touching Docker.
    max_workers : int
        Parallelism used when building environment images.
    already_succeeded : set[str]
        Instance IDs to skip — loaded from the checkpoint state file.
    state_file : Path | None
        Where to persist succeeded IDs after each build+push.
    state_succeeded : set[str]
        Mutable set shared with ``main()``; each newly-succeeded ID is added
        here so the caller can persist final state even on interruption.
    verbose_build : bool
        Whether to stream Docker build log lines to the console in real time.

    Returns
    -------
    succeeded_ids : list[str]
        IDs that were built (and pushed) successfully in this call.
    failed_ids : list[str]
        IDs that failed at any stage in this call.
    """
    sep = "─" * 72
    logger.info(sep)
    logger.info("REPO  %-45s  (%d instances)", repo, len(instances))
    logger.info(sep)

    # ── 0. Create TestSpec objects ──────────────────────────────────────────
    all_specs: list[TestSpec] = [
        make_test_spec(inst, namespace=namespace) for inst in instances
    ]

    # ── 1. Skip already-succeeded instances (checkpoint) ───────────────────
    skipped_specs = [s for s in all_specs if s.instance_id in already_succeeded]
    specs = [s for s in all_specs if s.instance_id not in already_succeeded]

    if skipped_specs:
        logger.info(
            "[%s] Skipping %d already-succeeded instance(s) (checkpoint): %s",
            repo,
            len(skipped_specs),
            [s.instance_id for s in skipped_specs],
        )

    if not specs:
        logger.info("[%s] All instances already succeeded — nothing to do.", repo)
        return [], []

    # ── 2. Build base + environment images ─────────────────────────────────
    logger.info(
        "[%s] Step 1/3 — building base and env images for %d instance(s) …",
        repo,
        len(specs),
    )
    env_failed_keys: set[str] = set()

    if dry_run:
        for s in specs:
            logger.info("[DRY RUN] Would build base image : %s", s.base_image_key)
            logger.info("[DRY RUN] Would build env  image : %s", s.env_image_key)
    else:
        try:
            _, env_failed_payloads = build_env_images(
                client=client,
                dataset=specs,
                force_rebuild=False,
                max_workers=max_workers,
            )
            # run_threadpool returns the original arg-tuples for failed jobs;
            # the first element of each tuple is the image name.
            env_failed_keys = {p[0] for p in env_failed_payloads}
            if env_failed_keys:
                logger.warning(
                    "[%s] %d env image(s) failed to build: %s",
                    repo,
                    len(env_failed_keys),
                    sorted(env_failed_keys),
                )
            else:
                logger.info("[%s] All environment images are ready.", repo)
        except Exception as exc:
            logger.error(
                "[%s] build_env_images raised an unexpected exception: %s", repo, exc
            )
            logger.debug(traceback.format_exc())
            # Treat every env image as failed so instance builds are skipped safely
            env_failed_keys = {s.env_image_key for s in specs}

    blocked_specs = [s for s in specs if s.env_image_key in env_failed_keys]
    buildable_specs = [s for s in specs if s.env_image_key not in env_failed_keys]

    failed_ids: list[str] = [s.instance_id for s in blocked_specs]
    if blocked_specs:
        logger.warning(
            "[%s] %d instance(s) blocked by failed env-image build: %s",
            repo,
            len(blocked_specs),
            [s.instance_id for s in blocked_specs],
        )

    if not buildable_specs:
        logger.error(
            "[%s] No buildable instances remain for this repo — skipping.", repo
        )
        return [], failed_ids

    # ── 3. Build each instance image, push immediately after each success ───
    logger.info(
        "[%s] Step 2/3 — building %d instance image(s) sequentially …",
        repo,
        len(buildable_specs),
    )

    # Specs eligible for local cleanup at the end of the repo
    cleanable_specs: list[TestSpec] = []
    # Specs whose push failed — kept locally for manual inspection
    push_failed_specs: list[TestSpec] = []
    succeeded_ids: list[str] = []

    for idx, spec in enumerate(buildable_specs, 1):
        iid = spec.instance_id
        pct = 100.0 * idx / len(buildable_specs)
        logger.info(
            "[%s] [%d/%d | %.0f%%] Building: %s",
            repo, idx, len(buildable_specs), pct, iid,
        )
        logger.debug("  Image key : %s", spec.instance_image_key)
        logger.debug("  Env key   : %s", spec.env_image_key)
        t0 = time.monotonic()
        build_ok = False

        if dry_run:
            logger.info(
                "[DRY RUN] Would build instance image: %s", spec.instance_image_key
            )
            build_ok = True
        else:
            # Locate the build log file for optional real-time tailing.
            # build_image() writes to <INSTANCE_IMAGE_BUILD_DIR>/<key>/build_image.log
            build_dir = INSTANCE_IMAGE_BUILD_DIR / spec.instance_image_key.replace(
                ":", "__"
            )
            build_log_file = build_dir / "build_image.log"

            tailer: Optional[BuildLogTailer] = None
            if verbose_build:
                tailer = BuildLogTailer(
                    log_file=build_log_file,
                    logger=logger,
                    instance_id=iid,
                ).start()

            try:
                # Pass logger=None so build_instance_image creates its own
                # file logger (INSTANCE_IMAGE_BUILD_DIR/.../prepare_image.log)
                build_instance_image(
                    test_spec=spec,
                    client=client,
                    logger=None,
                    nocache=False,
                )
                build_ok = True
            except BuildImageError as exc:
                logger.error(
                    "[%s] ✗ BuildImageError for '%s' after %.1fs:\n  %s",
                    repo, iid, time.monotonic() - t0, exc,
                )
                logger.debug(traceback.format_exc())
                failed_ids.append(iid)
            except Exception as exc:
                logger.error(
                    "[%s] ✗ Unexpected error building '%s' after %.1fs:\n  %s",
                    repo, iid, time.monotonic() - t0, exc,
                )
                logger.debug(traceback.format_exc())
                failed_ids.append(iid)
            finally:
                if tailer is not None:
                    tailer.stop()

            # When --verbose-build is off, dump the build log to the file
            # logger at DEBUG so it is always preserved in the log file.
            if not verbose_build and build_log_file.exists():
                try:
                    for line in build_log_file.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines():
                        if line.strip():
                            logger.debug("  [build-log] %s", line)
                except Exception:
                    pass

        if not build_ok:
            continue

        build_elapsed = time.monotonic() - t0
        logger.info(
            "[%s] ✓ Build complete: '%s'  (%.1fs)", repo, iid, build_elapsed
        )

        # ── 3b. Push immediately after each successful build ────────────────
        push_ok = True
        if registry:
            reg_image = get_registry_image_name(
                spec.instance_image_key, registry, tag
            )
            logger.info("[%s] Pushing '%s' …", repo, iid)
            logger.info("      %s", spec.instance_image_key)
            logger.info("   →  %s", reg_image)
            t_push = time.monotonic()
            push_ok = tag_and_push_image(
                client=client,
                local_image_name=spec.instance_image_key,
                registry_image_name=reg_image,
                logger=logger,
                dry_run=dry_run,
            )
            push_elapsed = time.monotonic() - t_push
            if push_ok:
                logger.info(
                    "[%s] ✓ Push complete: '%s'  (%.1fs)", repo, iid, push_elapsed
                )
            else:
                logger.error(
                    "[%s] ✗ Push FAILED for '%s' after %.1fs — "
                    "local image retained for manual inspection.",
                    repo, iid, push_elapsed,
                )
                push_failed_specs.append(spec)
                failed_ids.append(iid)
        else:
            logger.debug(
                "[%s] No --registry set; skipping push for '%s'.", repo, iid
            )

        if push_ok:
            cleanable_specs.append(spec)
            succeeded_ids.append(iid)
            # Persist checkpoint immediately so a crash/interrupt loses at most
            # one in-flight instance rather than the entire session.
            state_succeeded.add(iid)
            if state_file is not None:
                save_state(state_file, state_succeeded, logger)

    # ── 4. Cleanup: remove local images now that whole repo batch is done ───
    logger.info(
        "[%s] Step 3/3 — cleaning up %d local image(s) "
        "(deferred until full repo batch is complete) …",
        repo,
        len(cleanable_specs),
    )
    if cleanable_specs:
        cleanup_repo_images(
            client=client,
            specs=cleanable_specs,
            logger=logger,
            dry_run=dry_run,
        )
    else:
        logger.info("[%s] Nothing to clean up for this repo.", repo)

    if push_failed_specs:
        logger.warning(
            "[%s] %d image(s) left on disk due to push failures: %s",
            repo,
            len(push_failed_specs),
            [s.instance_image_key for s in push_failed_specs],
        )

    logger.info(
        "[%s] Repo done — succeeded: %d  |  failed: %d  |  skipped (checkpoint): %d",
        repo,
        len(succeeded_ids),
        len(failed_ids),
        len(skipped_specs),
    )
    return succeeded_ids, failed_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bulk-build Docker images for an SWE-bench dataset, optionally push "
            "them to a registry, and clean up local copies to save disk space.\n\n"
            "Instances are processed repo-by-repo (sorted by repo name, then by "
            "instance_id) to maximise Docker layer reuse.  Each instance is pushed "
            "immediately after it is built; local image cleanup happens once the "
            "entire repo batch is done."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_grp = p.add_argument_group("Dataset")
    dataset_grp.add_argument(
        "--dataset-name",
        default="princeton-nlp/SWE-bench_Verified",
        metavar="NAME",
        help=(
            "HuggingFace dataset name or path to a local .json/.jsonl file. "
            "(default: princeton-nlp/SWE-bench_Verified)"
        ),
    )
    dataset_grp.add_argument(
        "--split",
        default="test",
        metavar="SPLIT",
        help="Dataset split to load.  (default: test)",
    )
    dataset_grp.add_argument(
        "--instance-ids",
        nargs="+",
        metavar="ID",
        help="Only process these specific instance IDs (space-separated).",
    )
    dataset_grp.add_argument(
        "--repos",
        nargs="+",
        metavar="REPO",
        help=(
            "Only process repositories whose name contains any of these strings "
            "(space-separated, case-insensitive substring match).  "
            "Examples:  --repos astropy/astropy   or   --repos astropy sympy"
        ),
    )

    # ── Checkpoint / retry ───────────────────────────────────────────────────
    ckpt_grp = p.add_argument_group("Checkpoint / Retry")
    ckpt_grp.add_argument(
        "--state-file",
        default="build_state.json",
        metavar="FILE",
        help=(
            "JSON checkpoint file.  Already-succeeded instance IDs stored here "
            "are skipped automatically on the next run, so you can safely restart "
            "a hung or interrupted job without re-building from scratch.  "
            "Updated atomically after every successful build+push.  "
            "(default: build_state.json)"
        ),
    )
    ckpt_grp.add_argument(
        "--no-state",
        action="store_true",
        help="Disable the checkpoint state file (ignore existing state, write nothing).",
    )
    ckpt_grp.add_argument(
        "--retry-from",
        metavar="FILE",
        help=(
            "Path to a failed_builds.json written by a previous run.  "
            "Only the instance IDs listed in that file will be processed "
            "(stacks with --repos / --instance-ids filters)."
        ),
    )
    ckpt_grp.add_argument(
        "--failed-output",
        default="failed_builds.json",
        metavar="FILE",
        help=(
            "Write the JSON list of failed instance IDs here at end of run.  "
            "Pass this file to --retry-from for a targeted retry.  "
            "(default: failed_builds.json)"
        ),
    )

    # ── Build ────────────────────────────────────────────────────────────────
    build_grp = p.add_argument_group("Build")
    build_grp.add_argument(
        "--namespace",
        default="swebench",
        metavar="NS",
        help=(
            "Namespace prefix prepended to instance image names, e.g. 'swebench' "
            "produces 'swebench/sweb.eval.x86_64.<id>:…'.  "
            "Pass 'none' to omit.  (default: swebench)"
        ),
    )
    build_grp.add_argument(
        "--max-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel workers for building environment images within a "
            "single repo.  Instance images are always built sequentially.  "
            "(default: 1)"
        ),
    )
    build_grp.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the entire workflow — no Docker API calls are made.",
    )
    build_grp.add_argument(
        "--verbose-build",
        action="store_true",
        help=(
            "Stream Docker build log lines to the console in real time via a "
            "background thread.  Without this flag, build output is still fully "
            "captured to the log file (at DEBUG level)."
        ),
    )

    # ── Registry ─────────────────────────────────────────────────────────────
    reg_grp = p.add_argument_group("Registry")
    reg_grp.add_argument(
        "--registry",
        metavar="HOST/PROJECT",
        help=(
            "Registry and project prefix, e.g. 'registry.example.ai/agentbox'.  "
            "When set, each successfully built image is tagged as "
            "<registry>/<local_name>:<tag> and pushed immediately.  "
            "Omit to build locally only."
        ),
    )
    reg_grp.add_argument(
        "--tag",
        default="latest",
        metavar="TAG",
        help=(
            "Tag applied to pushed registry images, e.g. '20260304'.  "
            "Tip: avoid 'latest' in Kubernetes environments — use a date or "
            "commit-hash tag so imagePullPolicy works correctly.  (default: latest)"
        ),
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    misc_grp = p.add_argument_group("Misc")
    misc_grp.add_argument(
        "--log-dir",
        default="logs/build_all_images",
        metavar="DIR",
        help="Directory for per-run log files.  (default: logs/build_all_images)",
    )

    return p.parse_args()


def main() -> int:  # noqa: C901
    args = parse_args()
    log_dir = Path(args.log_dir)
    logger = setup_main_logger(log_dir)

    # Normalise namespace
    namespace: Optional[str] = args.namespace
    if namespace in ("none", "null", "", "None"):
        namespace = None

    # Resolve state file path
    state_file: Optional[Path] = None if args.no_state else Path(args.state_file)

    # ── Banner ───────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("SWE-bench Image Bulk Builder")
    logger.info("=" * 72)
    logger.info("  Dataset       : %s  (split: %s)", args.dataset_name, args.split)
    logger.info("  Namespace     : %s", namespace or "(none)")
    logger.info("  Registry      : %s", args.registry or "(none — local build only)")
    logger.info("  Tag           : %s", args.tag)
    logger.info("  Max workers   : %d", args.max_workers)
    logger.info("  Dry run       : %s", args.dry_run)
    logger.info("  Verbose build : %s", args.verbose_build)
    logger.info("  State file    : %s", state_file or "(disabled via --no-state)")
    logger.info("  Retry from    : %s", args.retry_from or "(none)")
    logger.info("  Failed output : %s", args.failed_output)
    if args.repos:
        logger.info("  Repo filter   : %s", args.repos)
    if args.instance_ids:
        logger.info("  Instance IDs  : %s", args.instance_ids)
    logger.info("=" * 72)

    # ── Load checkpoint state ────────────────────────────────────────────────
    # state_succeeded is a mutable set shared with process_repo so each
    # successful build+push is persisted immediately (crash-safe).
    already_succeeded: set[str] = set()
    state_succeeded: set[str] = set()
    if state_file is not None:
        already_succeeded = load_state(state_file, logger)
        state_succeeded = set(already_succeeded)

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info("Loading dataset '%s' …", args.dataset_name)
    try:
        dataset = load_swebench_dataset(args.dataset_name, args.split)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        logger.debug(traceback.format_exc())
        return 1
    logger.info("Loaded %d instance(s) from the dataset.", len(dataset))

    # ── Apply filters ────────────────────────────────────────────────────────
    if args.instance_ids:
        wanted = set(args.instance_ids)
        before = len(dataset)
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in wanted]
        logger.info(
            "--instance-ids filter: %d → %d instance(s).", before, len(dataset)
        )

    if args.retry_from:
        retry_path = Path(args.retry_from)
        if not retry_path.exists():
            logger.error("--retry-from file not found: '%s'", retry_path)
            return 1
        try:
            retry_ids: set[str] = set(json.loads(retry_path.read_text()))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Cannot parse --retry-from file '%s': %s", retry_path, exc)
            return 1
        before = len(dataset)
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in retry_ids]
        logger.info(
            "--retry-from '%s': %d ID(s) in file, %d → %d instance(s) matched.",
            retry_path, len(retry_ids), before, len(dataset),
        )

    if not dataset:
        logger.warning("No instances to process after filtering — exiting.")
        return 0

    # ── Group by repo, sort by repo then instance_id ─────────────────────────
    repo_groups: dict[str, list] = defaultdict(list)
    for inst in dataset:
        repo_groups[inst["repo"]].append(inst)

    sorted_repos = sorted(repo_groups.keys())
    for repo in sorted_repos:
        repo_groups[repo].sort(key=lambda x: x[KEY_INSTANCE_ID])

    # ── Filter repos ─────────────────────────────────────────────────────────
    if args.repos:
        filters = [r.lower() for r in args.repos]
        before_repos = len(sorted_repos)
        sorted_repos = [
            r for r in sorted_repos if any(f in r.lower() for f in filters)
        ]
        excluded = before_repos - len(sorted_repos)
        if not sorted_repos:
            logger.warning(
                "--repos filter %s matched 0 repos.  Available repos: %s",
                args.repos,
                sorted(repo_groups.keys()),
            )
            return 0
        logger.info(
            "--repos filter: %d total → %d selected (%d excluded).  Selected: %s",
            before_repos, len(sorted_repos), excluded, sorted_repos,
        )

    total_instances = sum(len(repo_groups[r]) for r in sorted_repos)
    total_repos = len(sorted_repos)
    logger.info(
        "Processing plan: %d instance(s) across %d repo(s).",
        total_instances, total_repos,
    )

    # Per-repo breakdown including checkpoint info
    for repo in sorted_repos:
        insts = repo_groups[repo]
        done = sum(1 for i in insts if i[KEY_INSTANCE_ID] in already_succeeded)
        logger.info(
            "  %-45s  total=%-3d  checkpoint_done=%-3d  to_build=%d",
            repo, len(insts), done, len(insts) - done,
        )

    # ── Connect to Docker ────────────────────────────────────────────────────
    client: Optional[docker.DockerClient] = None
    if not args.dry_run:
        try:
            client = docker.from_env()
            ver = client.version()
            logger.info(
                "Connected to Docker Engine %s (API %s).",
                ver.get("Version", "?"),
                ver.get("ApiVersion", "?"),
            )
        except Exception as exc:
            logger.error("Cannot connect to Docker daemon: %s", exc)
            logger.debug(traceback.format_exc())
            return 1
    else:
        logger.info("[DRY RUN] Skipping Docker connection.")

    # ── Process each repo ────────────────────────────────────────────────────
    all_succeeded: list[str] = []
    all_failed: list[str] = []
    t_overall = time.monotonic()

    for repo_idx, repo in enumerate(sorted_repos, 1):
        instances = repo_groups[repo]
        to_build_count = sum(
            1 for i in instances if i[KEY_INSTANCE_ID] not in already_succeeded
        )
        logger.info(
            "\n[Repo %d/%d] %s  (total=%d  to_build=%d)",
            repo_idx, total_repos, repo, len(instances), to_build_count,
        )
        t_repo = time.monotonic()

        succeeded, failed = process_repo(
            repo=repo,
            instances=instances,
            client=client,
            logger=logger,
            namespace=namespace,
            registry=args.registry,
            tag=args.tag,
            dry_run=args.dry_run,
            max_workers=args.max_workers,
            already_succeeded=already_succeeded,
            state_file=state_file,
            state_succeeded=state_succeeded,
            verbose_build=args.verbose_build,
        )

        all_succeeded.extend(succeeded)
        all_failed.extend(failed)

        # Per-repo timing + ETA
        elapsed_repo = time.monotonic() - t_repo
        elapsed_total = time.monotonic() - t_overall
        newly_processed = len(all_succeeded) + len(all_failed)
        # ETA is based on newly-built instances only (already_succeeded were instant)
        eta_s = (elapsed_total / max(newly_processed, 1)) * max(
            total_instances - newly_processed - len(already_succeeded), 0
        )
        logger.info(
            "[Repo %d/%d] %s  finished in %.1fs  |  "
            "succeeded=%d  failed=%d  |  "
            "session total: %d built  ETA ≈%.0fs",
            repo_idx, total_repos, repo,
            elapsed_repo, len(succeeded), len(failed),
            newly_processed, eta_s,
        )

    # ── Final summary ────────────────────────────────────────────────────────
    t_total = time.monotonic() - t_overall
    logger.info("")
    logger.info("=" * 72)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 72)
    logger.info("  Total wall time     : %.1fs (%.2fh)", t_total, t_total / 3600)
    logger.info("  Skipped (checkpoint): %d", len(already_succeeded))
    logger.info("  Succeeded this run  : %d", len(all_succeeded))
    logger.info("  Failed    this run  : %d", len(all_failed))
    logger.info(
        "  Grand total done    : %d / %d",
        len(all_succeeded) + len(already_succeeded),
        total_instances + len(already_succeeded),
    )

    if all_failed:
        logger.warning("  Failed IDs : %s", all_failed)
        out_path = Path(args.failed_output)
        try:
            out_path.write_text(json.dumps(sorted(all_failed), indent=2))
            logger.info("  Failed IDs written to '%s'.", out_path)
            logger.info(
                "  To retry:  python build_all_images.py --retry-from %s", out_path
            )
        except OSError as exc:
            logger.error(
                "  Could not write failed-output file '%s': %s", out_path, exc
            )
    else:
        logger.info("  All processed instances built successfully! 🎉")

    # Final state save (belt-and-suspenders in case a per-instance save was missed)
    if state_file is not None and state_succeeded:
        save_state(state_file, state_succeeded, logger)
        logger.info(
            "  State file '%s' updated (%d total succeeded).",
            state_file, len(state_succeeded),
        )

    logger.info("=" * 72)
    return 0 if not all_failed else 1


if __name__ == "__main__":
    sys.exit(main())
