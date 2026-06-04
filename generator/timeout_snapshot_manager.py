# timeout_snapshot_manager.py

import json
import shutil
import time
from collections import deque
from pathlib import Path

from rdkit import Chem


class TimeoutSnapshotManager:
    """
    Stores timeout branches and later restores full DFS state.

    Semantics:
    - timeout branch is NOT completed;
    - normally finished branches are stored in completed_branch_keys;
    - after normal DFS reaches the end, the first timeout snapshot is restored;
    - restored run continues normal DFS from timeout_iter / timeout_mol_idx;
    - if restored DFS reaches already completed branches, they are skipped.
    """

    def __init__(
        self,
        *,
        path_to_working_dir,
        max_execution_time,
        start_time,
        max_branch_seconds,
        log=None,
        verbose=True,
        snapshots_dir_name="timeout_branch_snapshots",
    ):
        self.path_to_working_dir = Path(path_to_working_dir)
        self.start_time = start_time
        self.global_deadline = start_time + max_execution_time
        self.max_branch_seconds = max_branch_seconds
        self.log = log
        self.verbose = verbose

        self.snapshots_dir = self.path_to_working_dir / snapshots_dir_name
        self.snapshots_dir.mkdir(exist_ok=True)

        self.index_path = self.snapshots_dir / "timeout_snapshots_index.jsonl"

        self.snapshot_counter = 0
        self.snapshot_queue = deque()

        # Active restored snapshot means:
        # "we are continuing DFS from a previously timed-out point".
        self.active_snapshot = None

        # Branches fully closed by normal DFS or restored DFS.
        self.completed_branch_keys = set()

        # Branches that timed out and therefore must NOT be treated as completed.
        self.timeout_branch_keys = set()

        # Snapshot ids already fully restored and closed.
        self.finished_snapshot_ids = set()

        # Prevent duplicate snapshots for the exact same timeout branch while queued.
        self.queued_branch_keys = set()

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    def _log(self, text):
        if self.verbose:
            print(text)
        if self.log is not None:
            self.log.write(text + "\n")

    # ------------------------------------------------------------
    # Branch keys
    # ------------------------------------------------------------
    @staticmethod
    def branch_key_from_a(a_state, iter_level):
        """
        Full branch key including current mol.
        Example for iter_level=5:
            (a[1], a[2], a[3], a[4], a[5])
        """
        return tuple(int(a_state[i]) for i in range(1, iter_level + 1))

    @staticmethod
    def parent_key_from_a(a_state, iter_level):
        """
        Parent branch key.
        Example for iter_level=5:
            (a[1], a[2], a[3], a[4])
        """
        return tuple(int(a_state[i]) for i in range(1, iter_level))

    def selected_count(self, iter_level):
        selected_path = self.path_to_working_dir / f"iter{iter_level - 1}_selected.sdf"

        if not selected_path.exists():
            return 0

        return len(Chem.SDMolSupplier(str(selected_path)))

    @staticmethod
    def safe_close_writer(writer):
        try:
            writer.close()
        except Exception:
            pass

    # ------------------------------------------------------------
    # Completion / skip logic
    # ------------------------------------------------------------
    def mark_parent_completed(self, *, a, iter_level):
        """
        Called when all mols on current iter_level are studied.

        If all mols from iter{iter_level - 1}_selected.sdf are done,
        then the parent branch at iter_level - 1 is fully completed.
        """
        if iter_level <= 1:
            return

        completed_key = self.parent_key_from_a(a, iter_level)

        if not completed_key:
            return

        self.completed_branch_keys.add(completed_key)
        self.timeout_branch_keys.discard(completed_key)

        self._log(f"Branch completed: {completed_key}")

        # If this completed branch belongs to active restored snapshot,
        # and we closed the original timeout branch itself, mark snapshot done.
        if self.active_snapshot is not None:
            timeout_key = tuple(self.active_snapshot["branch_key"])

            if completed_key == timeout_key:
                snapshot_id = self.active_snapshot["snapshot_id"]
                self.finished_snapshot_ids.add(snapshot_id)

                self._log(
                    f"Restored timeout branch completed: "
                    f"{snapshot_id}, branch {timeout_key}"
                )

                self.active_snapshot = None

    def should_skip_completed_branch(self, *, a, iter_level):
        """
        Called before generation of current branch.

        Skip only branches that are completed and not currently marked as timeout.
        """
        current_key = self.branch_key_from_a(a, iter_level)

        if current_key in self.completed_branch_keys and current_key not in self.timeout_branch_keys:
            self._log(f"Skipping already completed branch: {current_key}")
            return True

        return False

    # ------------------------------------------------------------
    # Snapshot saving
    # ------------------------------------------------------------
    def _write_index_record(self, snapshot_meta):
        with open(self.index_path, "a") as f:
            f.write(json.dumps(snapshot_meta, sort_keys=True) + "\n")

    def save_snapshot(self, *, a, iter_level, mol_idx, reason):
        self.snapshot_counter += 1

        snapshot_id = f"timeout_{self.snapshot_counter:06d}"
        snapshot_dir = self.snapshots_dir / snapshot_id
        snapshot_dir.mkdir(exist_ok=False)

        a_state = {str(k): int(v) for k, v in a.items()}

        copied_selected_files = []

        # To resume DFS from iter_level, we need selected files up to
        # iter{iter_level - 1}_selected.sdf.
        for k in range(0, iter_level):
            src = self.path_to_working_dir / f"iter{k}_selected.sdf"

            if src.exists():
                dst = snapshot_dir / f"iter{k}_selected.sdf"
                shutil.copyfile(src, dst)
                copied_selected_files.append(f"iter{k}_selected.sdf")

        branch_key = self.branch_key_from_a(a, iter_level)
        parent_key = self.parent_key_from_a(a, iter_level)

        snapshot_meta = {
            "snapshot_id": snapshot_id,
            "snapshot_dir": str(snapshot_dir),
            "reason": reason,
            "created_at_seconds_from_start": float(time.time() - self.start_time),

            "timeout_iter": int(iter_level),
            "timeout_mol_idx": int(mol_idx),

            "branch_key": list(branch_key),
            "parent_key": list(parent_key),

            "a_state": a_state,

            # Not used for skipping directly; kept for debug.
            "finished_before_timeout": list(range(int(mol_idx))),
            "timeout_mol_is_finished": False,
            "parent_selected_count_at_timeout": int(self.selected_count(iter_level)),

            "copied_selected_files": copied_selected_files,
            "retry_count": 0,
        }

        with open(snapshot_dir / "snapshot_meta.json", "w") as f:
            json.dump(snapshot_meta, f, indent=2, sort_keys=True)

        self._write_index_record(snapshot_meta)

        return snapshot_meta

    def save_timeout_if_new(self, *, a, iter_level, mol_idx, reason):
        """
        Save timeout branch unless exact branch is already queued.

        If timeout happens during restored DFS, save the current timeout point
        as a new snapshot too. Do not mark it completed.
        """
        branch_key = self.branch_key_from_a(a, iter_level)

        self.timeout_branch_keys.add(branch_key)
        self.completed_branch_keys.discard(branch_key)

        if branch_key in self.queued_branch_keys:
            self._log(
                f"Timeout branch already queued, not duplicated: "
                f"iter {iter_level}, mol {mol_idx}, branch {branch_key}"
            )
            return None

        snapshot_meta = self.save_snapshot(
            a=a,
            iter_level=iter_level,
            mol_idx=mol_idx,
            reason=reason,
        )

        self.snapshot_queue.append(snapshot_meta)
        self.queued_branch_keys.add(branch_key)

        self._log(
            f"Timeout branch snapshot saved: "
            f"{snapshot_meta['snapshot_id']}, iter {iter_level}, "
            f"mol {mol_idx}, branch {branch_key}, reason={reason}"
        )

        return snapshot_meta

    # ------------------------------------------------------------
    # Snapshot restoring
    # ------------------------------------------------------------
    def restore_snapshot(self, *, snapshot_meta, a, deadline_by_iter):
        """
        Restore full DFS state from timeout point.

        This resumes DFS, not just one mol.
        """
        snapshot_dir = Path(snapshot_meta["snapshot_dir"])
        timeout_iter = int(snapshot_meta["timeout_iter"])
        timeout_mol_idx = int(snapshot_meta["timeout_mol_idx"])

        for filename in snapshot_meta["copied_selected_files"]:
            src = snapshot_dir / filename
            dst = self.path_to_working_dir / filename
            shutil.copyfile(src, dst)

        for k in a:
            a[k] = 0

        for k_str, v in snapshot_meta["a_state"].items():
            k = int(k_str)
            if k in a:
                a[k] = int(v)

        a[timeout_iter] = timeout_mol_idx

        for k in a:
            if k > timeout_iter:
                a[k] = 0

        deadline_by_iter.clear()
        deadline_by_iter[0] = self.global_deadline

        snapshot_meta["retry_count"] = int(snapshot_meta.get("retry_count", 0)) + 1

        with open(snapshot_dir / "snapshot_meta.json", "w") as f:
            json.dump(snapshot_meta, f, indent=2, sort_keys=True)

        self.active_snapshot = snapshot_meta

        self._log(
            f"Restored timeout DFS snapshot {snapshot_meta['snapshot_id']}: "
            f"resume from iter {timeout_iter}, mol {timeout_mol_idx}, "
            f"retry #{snapshot_meta['retry_count']}"
        )

        return timeout_iter

    def restore_next_if_available(self, *, a, deadline_by_iter):
        """
        Restore next queued timeout snapshot after normal DFS is exhausted.
        """
        while self.snapshot_queue and time.time() < self.global_deadline:
            snapshot_meta = self.snapshot_queue.popleft()

            branch_key = tuple(snapshot_meta["branch_key"])
            self.queued_branch_keys.discard(branch_key)

            if snapshot_meta["snapshot_id"] in self.finished_snapshot_ids:
                continue

            # If branch was completed later, no need to restore it.
            if branch_key in self.completed_branch_keys and branch_key not in self.timeout_branch_keys:
                continue

            restored_iter = self.restore_snapshot(
                snapshot_meta=snapshot_meta,
                a=a,
                deadline_by_iter=deadline_by_iter,
            )

            return restored_iter

        return None

    # ------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------
    def active_snapshot_budget_override(self, *, iter_n, time_left):
        """
        For the exact restored timeout level, give one clean retry budget.
        Deeper levels then use normal dynamic splitting.
        """
        if self.active_snapshot is None:
            return None

        if iter_n != int(self.active_snapshot["timeout_iter"]):
            return None

        return min(self.max_branch_seconds, time_left)