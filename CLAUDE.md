# Python tooling

Use uv for every Python command in this project.

- Run scripts and tools with `uv run`, never bare `python` or `python3`.
- Add and remove dependencies with `uv add` and `uv remove`, never `pip install`.
- Sync and lock the environment with `uv sync` and `uv lock`.
- Run a one-off tool without adding it to the project: `uvx ruff check`.
- For a standalone script, use `uv run script.py` and add its dependencies with
  `uv add --script script.py <package>`.



# Execution & Editing Guidelines

- **Read-Only / Advisory Mode:** Never apply code changes directly. Do not create, modify, or delete files, and do not execute automated edit tools/commands.
- **Workflow for Issues & Refactoring:**
  1. **Diagnosis:** Clearly identify and explain the root cause of the issue or the reason for the requested change.
  2. **Proposed Solution:** Outline the fix conceptually.
  3. **Code Snippets:** Provide the exact code snippets or diffs in the chat output for manual review and application, specifying the relevant file paths and line numbers.