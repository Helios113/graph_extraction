"""Locate / print / install the agent skill (SKILL.md) that ships with construct-onnx.

    python -m construct.skill                 # print the skill
    python -m construct.skill --path          # print the file path
    python -m construct.skill --install DIR   # copy to DIR/construct-onnx/SKILL.md (e.g. .claude/skills)
"""
import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path


def path() -> Path:
    """Filesystem path of the packaged SKILL.md."""
    return Path(str(resources.files("construct") / "SKILL.md"))


def text() -> str:
    return path().read_text()


def install(skills_dir: str | Path) -> Path:
    """Copy the skill to <skills_dir>/construct-onnx/SKILL.md and return the new path."""
    dest = Path(skills_dir) / "construct-onnx" / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path(), dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--path", action="store_true", help="print the SKILL.md path instead of its contents")
    ap.add_argument("--install", metavar="SKILLS_DIR", help="copy the skill into SKILLS_DIR/construct-onnx/")
    a = ap.parse_args(argv)
    if a.install:
        print(install(a.install))
    elif a.path:
        print(path())
    else:
        sys.stdout.write(text())


if __name__ == "__main__":
    main()
