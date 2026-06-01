"""
deploy_sync.py
--------------
Sync the backend (03_web_app) INTO the HF Space clone.
Then, inside the clone, review with `git status`, commit, and push to HF Space.

Safe by design:
  - Only READS from SRC (03_web_app) and WRITES to DST (the clone).
  - Never runs git. Never touches the AgoraPottery working tree.
  - Refuses to run unless SRC looks like the backend and DST looks like the clone.
  - Prints a report of NEW / CHANGED / DELETED files and asks for confirmation
    BEFORE changing anything.
"""

import filecmp
import shutil
import sys
from pathlib import Path

# ── CONFIG ─────────────────────────
# Repo local paths
SRC = Path(r"C:\Users\dimit\PycharmProjects\AgoraPottery\03_web_app")
DST = Path(r"C:\Users\dimit\PycharmProjects\PotteryChronologyPredictorAPI-HFSpace")

# Never copied, never deleted (mirrors .dockerignore and the clone's .gitignore).
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".idea", "tmp", ".cache",
    "scripts", "seeders", "uploaders",
}

# Files left untouched in DST (the clone's own config — must NOT be deleted!).
EXCLUDE_FILES = {".env", ".gitignore", ".gitattributes"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def is_excluded(file_rel_path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in file_rel_path.parts):
        return True
    if file_rel_path.name in EXCLUDE_FILES or file_rel_path.name.startswith(".env"):
        return True
    if file_rel_path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def deployable_files(root: Path) -> set[Path]:
    """Relative paths of files under root, minus the excluded ones."""
    return {
        file.relative_to(root)
        for file in root.rglob("*")  # recursively find all files under the root directory tree
        if file.is_file() and not is_excluded(file.relative_to(root))
    }


def main():
    # ── Safety guards (prevent the 'wrong folder' catastrophe) ──
    if not (SRC / "main.py").is_file():
        sys.exit(f"❌ SRC has no main.py - is this really the backend?\n   {SRC}")
    if not (DST / ".git").is_dir():
        sys.exit(f"❌ DST has no .git - is this really the HF Space clone?\n   {DST}")
    if SRC.resolve() == DST.resolve():
        sys.exit("❌ SRC and DST are the same folder!")

    print(f"🔄 Sync   {SRC}\n   →   {DST}")

    src_files = deployable_files(SRC)
    dst_files = deployable_files(DST)

    to_add = sorted(src_files - dst_files)
    to_delete = sorted(dst_files - src_files)
    to_overwrite = sorted(
        file_rel_path for file_rel_path in (src_files & dst_files)
        if not filecmp.cmp(SRC / file_rel_path, DST / file_rel_path, shallow=False)
        # compare contents of files with same relative paths
    )

    if not (to_add or to_overwrite or to_delete):
        print("✅ Nothing to do - the clone already matches the original.")
        return

    # ── Report ──
    def show(title, items):
        if not bool(items): return

        print(f"\n{title} ({len(items)} files):")
        for file_rel_path in items:
            print(f"        {file_rel_path.as_posix()}")

    show("🟢 NEW (will be created)", to_add)
    show("🔵 CHANGED (will be overwritten)", to_overwrite)
    show("🔴 STALE (will be DELETED from the clone)", to_delete)

    # ── Confirm ──
    if input("\nApply these changes? [y/N] ").strip().lower() != "y":
        print("Aborted. Nothing changed.")
        return

    # ── Apply ──
    for file_rel_path in to_add + to_overwrite:
        file_dst_path = DST / file_rel_path
        file_dst_path.parent.mkdir(parents=True, exist_ok=True) # create file's parent directories if they don't exist
        shutil.copy2(SRC / file_rel_path, file_dst_path) # copy file from original to clone (replace or create new)
    for file_rel_path in to_delete:
        (DST / file_rel_path).unlink()

    print(
        f"\n✅ Done: {len(to_add)} new, {len(to_overwrite)} changed, "
        f"{len(to_delete)} deleted."
    )
    print(f"   To deploy, inside `{DST}`:\n`git add -A`, review `git status`, commit, push.")


if __name__ == "__main__":
    main()
