"""
Remote file browser (Textual TUI) for upload.py ls.

Returns (chosen_to_pull, chosen_to_delete) — both lists of ptr dicts.
Confirmation prompts and SFTP deletion happen in cmd_ls (upload.py).

Keys:
  Enter          — navigate into directory
  Space          — toggle pull (green ✓); clears delete if set
  d              — toggle delete (red ✗); clears pull if set
  p              — confirm pull and exit
  x              — confirm delete and exit
  q              — quit without action

Glyph column (one character, same for files and dirs):
  (blank)  — unselected
  ✓        — selected for pull  (green)
  ✗        — selected for delete (red)

For directories the glyph reflects the aggregate state of all files under them.
"""

import hashlib
import os
from datetime import datetime, timezone

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Label, ListItem, ListView

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_GLYPH_NONE   = "☐"
_GLYPH_PULL   = "☑"
_GLYPH_DELETE = "☒"
_CACHED_YES   = "●"
_CACHED_NO    = "○"


def _stable_id(prefix: str, raw: str) -> str:
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


class Entry(ListItem):
    def __init__(self, label: str, kind: str, payload, **kwargs):
        super().__init__(Label(label), **kwargs)
        self.kind = kind
        self.payload = payload


class BrowserApp(App):
    CSS = """
    ListView { height: 1fr; border: solid $primary; }
    ListItem { padding: 0 1; }
    ListItem:hover { background: $boost; }
    ListItem.pull            Label { color: $success; }
    ListItem.delete          Label { color: $error; }
    ListItem.delete-partial  Label { color: orange; }
    ListItem.dir             Label { color: $warning; }
    #status { height: 1; background: $surface; padding: 0 1; }
    """
    BINDINGS = [
        Binding("space", "toggle_pull",   "Toggle pull"),
        Binding("d",     "toggle_delete", "Toggle delete"),
        Binding("p",     "confirm_pull",  "Pull selected"),
        Binding("x",     "confirm_delete","Delete selected"),
        Binding("q",     "quit_app",      "Quit"),
    ]

    def __init__(self, tree: dict):
        super().__init__()
        self._tree = tree
        self._cwd = "."
        self._pull_sel: set[str] = set()   # original_path keys
        self._del_sel:  set[str] = set()   # original_path keys
        self.chosen_pull:   list = []
        self.chosen_delete: list = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield ListView()
        yield Label("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Remote file browser"
        self._refresh_list()

    # ── tree helpers ─────────────────────────────────────────────────────────

    def _files_under(self, dir_path: str) -> list:
        node = self._tree.get(dir_path, {"subdirs": set(), "files": []})
        results = list(node["files"])
        for sub in node["subdirs"]:
            results.extend(self._files_under(sub))
        return results

    def _all_files(self) -> list:
        return [p for node in self._tree.values() for p in node["files"]]

    # ── label builders ───────────────────────────────────────────────────────

    def _file_glyph(self, key: str) -> str:
        if key in self._pull_sel:
            return _GLYPH_PULL
        if key in self._del_sel:
            return _GLYPH_DELETE
        return _GLYPH_NONE

    def _dir_state(self, dir_path: str) -> str:
        """Return 'none', 'pull', 'pull-partial', 'delete', 'delete-partial'."""
        files = self._files_under(dir_path)
        if not files:
            return "none"
        keys = {p["original_path"] for p in files}
        if keys.issubset(self._del_sel):
            return "delete"
        if keys.issubset(self._pull_sel):
            return "pull"
        if keys & self._del_sel:
            return "delete-partial"
        if keys & self._pull_sel:
            return "pull-partial"
        return "none"

    def _dir_glyph(self, dir_path: str) -> str:
        state = self._dir_state(dir_path)
        if "delete" in state:
            return _GLYPH_DELETE
        if "pull" in state:
            return _GLYPH_PULL
        return _GLYPH_NONE

    def _make_file_label(self, p: dict) -> str:
        local       = os.path.join(PROJECT_ROOT, p["original_path"])
        cached      = _CACHED_YES if os.path.exists(local) else _CACHED_NO
        glyph       = self._file_glyph(p["original_path"])
        size_mb     = p["size"] / 1_048_576
        uploaded_at = p.get("uploaded_at")
        date_str    = (
            datetime.fromtimestamp(uploaded_at, timezone.utc).strftime("%Y-%m-%d %H:%M")
            if uploaded_at else "unknown"
        )
        return f"{glyph} {cached}  {p['filename']:<40} {size_mb:7.1f} MB  {date_str}"

    def _make_dir_label(self, dir_path: str) -> str:
        name  = dir_path.split("/")[-1]
        glyph = self._dir_glyph(dir_path)
        files = self._files_under(dir_path)
        n_cached = sum(1 for p in files if os.path.exists(os.path.join(PROJECT_ROOT, p["original_path"])))
        cached = _CACHED_YES if files and n_cached == len(files) else _CACHED_NO
        return f"{glyph} {cached}  {name}/"

    def _dir_css_class(self, dir_path: str) -> str:
        state = self._dir_state(dir_path)
        return {
            "delete":         "delete",
            "delete-partial": "delete-partial",
            "pull":           "pull",
            "pull-partial":   "pull",
            "none":           "dir",
        }[state]

    # ── list rendering ───────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        node    = self._tree.get(self._cwd, {"subdirs": set(), "files": []})
        subdirs = sorted(node["subdirs"])
        files   = sorted(node["files"], key=lambda p: p["filename"])
        self.sub_title = f"{self._cwd}/"

        lv = self.query_one(ListView)
        lv.clear()

        if self._cwd != ".":
            parent = "/".join(self._cwd.split("/")[:-1]) or "."
            self._safe_append(lv, Entry("     ▲  ../", "parent", parent,
                                        id=_stable_id("parent", self._cwd)))

        for d in subdirs:
            item = Entry(self._make_dir_label(d), "dir", d, id=_stable_id("d", d))
            item.add_class(self._dir_css_class(d))
            self._safe_append(lv, item)

        for p in files:
            key  = p["original_path"]
            item = Entry(self._make_file_label(p), "file", p,
                         id=_stable_id("f", key))
            if key in self._pull_sel:
                item.add_class("pull")
            elif key in self._del_sel:
                item.add_class("delete")
            self._safe_append(lv, item)

        self._update_status()

    def _safe_append(self, lv: ListView, item: ListItem) -> None:
        try:
            lv.append(item)
        except Exception as exc:
            self.query_one("#status", Label).update(f"  [error] {exc}")

    def _update_status(self) -> None:
        parts = []
        if self._pull_sel:
            parts.append(f"{len(self._pull_sel)} pull")
        if self._del_sel:
            parts.append(f"{len(self._del_sel)} delete")
        sel_str = ", ".join(parts) if parts else "none"
        self.query_one("#status", Label).update(
            f"  [{sel_str}]  Space: pull · D: delete · P: confirm pull · X: confirm delete · Q: quit"
        )

    # ── item update (without full list rebuild) ──────────────────────────────

    def _repaint_item(self, item: Entry) -> None:
        if item.kind == "file":
            p   = item.payload
            key = p["original_path"]
            item.query_one(Label).update(self._make_file_label(p))
            item.remove_class("pull", "delete")
            if key in self._pull_sel:
                item.add_class("pull")
            elif key in self._del_sel:
                item.add_class("delete")
        elif item.kind == "dir":
            d = item.payload
            item.query_one(Label).update(self._make_dir_label(d))
            item.remove_class("pull", "delete", "delete-partial", "dir")
            item.add_class(self._dir_css_class(d))

    def _repaint_all_dirs(self) -> None:
        for item in self.query_one(ListView).children:
            if isinstance(item, Entry) and item.kind == "dir":
                self._repaint_item(item)

    # ── event handlers ───────────────────────────────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if not isinstance(item, Entry):
            return
        if item.kind in ("dir", "parent"):
            self._cwd = item.payload
            self._refresh_list()
        else:
            self._do_toggle_pull(item)

    def action_toggle_pull(self) -> None:
        item = self.query_one(ListView).highlighted_child
        if not isinstance(item, Entry):
            return
        if item.kind == "dir":
            self._do_toggle_pull_dir(item)
        elif item.kind == "file":
            self._do_toggle_pull(item)

    def action_toggle_delete(self) -> None:
        item = self.query_one(ListView).highlighted_child
        if not isinstance(item, Entry):
            return
        if item.kind == "dir":
            self._do_toggle_delete_dir(item)
        elif item.kind == "file":
            self._do_toggle_delete(item)

    # ── toggle logic ─────────────────────────────────────────────────────────

    def _do_toggle_pull(self, item: Entry) -> None:
        key = item.payload["original_path"]
        if key in self._pull_sel:
            self._pull_sel.discard(key)
        else:
            self._pull_sel.add(key)
            self._del_sel.discard(key)   # mutual exclusion
        self._repaint_item(item)
        self._repaint_all_dirs()
        self._update_status()

    def _do_toggle_delete(self, item: Entry) -> None:
        key = item.payload["original_path"]
        if key in self._del_sel:
            self._del_sel.discard(key)
        else:
            self._del_sel.add(key)
            self._pull_sel.discard(key)  # mutual exclusion
        self._repaint_item(item)
        self._repaint_all_dirs()
        self._update_status()

    def _repaint_all_items(self) -> None:
        for item in self.query_one(ListView).children:
            if isinstance(item, Entry) and item.kind in ("file", "dir"):
                self._repaint_item(item)

    def _do_toggle_pull_dir(self, item: Entry) -> None:
        files = self._files_under(item.payload)
        keys  = {p["original_path"] for p in files}
        if keys and keys.issubset(self._pull_sel):
            self._pull_sel -= keys
        else:
            self._pull_sel |= keys
            self._del_sel  -= keys
        self._repaint_all_items()
        self._update_status()

    def _do_toggle_delete_dir(self, item: Entry) -> None:
        files = self._files_under(item.payload)
        keys  = {p["original_path"] for p in files}
        if keys and keys.issubset(self._del_sel):
            self._del_sel -= keys
        else:
            self._del_sel  |= keys
            self._pull_sel -= keys
        self._repaint_all_items()
        self._update_status()

    # ── exit actions ─────────────────────────────────────────────────────────

    def action_confirm_pull(self) -> None:
        if not self._pull_sel:
            return
        self.chosen_pull = [p for p in self._all_files()
                            if p["original_path"] in self._pull_sel]
        self.exit()

    def action_confirm_delete(self) -> None:
        if not self._del_sel:
            return
        self.chosen_delete = [p for p in self._all_files()
                               if p["original_path"] in self._del_sel]
        self.exit()

    def action_quit_app(self) -> None:
        self.exit()


def browse(tree: dict) -> tuple[list, list]:
    """Run the browser and return (to_pull, to_delete)."""
    app = BrowserApp(tree)
    app.run()
    return app.chosen_pull, app.chosen_delete
