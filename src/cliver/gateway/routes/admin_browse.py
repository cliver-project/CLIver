"""File browser route for the admin portal."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


def get_browse_routes(require_auth: Callable) -> list:
    """Return file browser API route."""

    @require_auth
    async def handle_browse_files(request: Request):
        home = Path.home().resolve()
        dir_path = request.query_params.get("dir", "")
        file_filter = request.query_params.get("filter", "")
        if not dir_path:
            dir_path = str(home)
        target = Path(dir_path).expanduser().resolve()

        try:
            target.relative_to(home)
        except ValueError:
            return JSONResponse({"error": "Access restricted to home directory"}, status_code=403)

        if not target.is_dir():
            return JSONResponse({"error": "Not a directory", "path": str(target)}, status_code=400)

        filter_exts = set()
        if file_filter == "image":
            filter_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp"}
        elif file_filter == "audio":
            filter_exts = {".mp3", ".wav", ".ogg", ".aac", ".flac", ".m4a"}
        elif file_filter == "video":
            filter_exts = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

        items = []
        try:
            parent = target.parent.resolve()
            try:
                parent.relative_to(home)
                if parent != target:
                    items.append({"name": "..", "path": str(parent), "type": "dir"})
            except ValueError:
                pass
            for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    items.append({"name": entry.name, "path": str(entry), "type": "dir"})
                elif entry.is_file():
                    if filter_exts and entry.suffix.lower() not in filter_exts:
                        continue
                    items.append({"name": entry.name, "path": str(entry), "type": "file", "size": entry.stat().st_size})
        except PermissionError:
            return JSONResponse({"error": "Permission denied", "path": str(target)}, status_code=403)

        return JSONResponse({"path": str(target), "items": items})

    return [Route("/admin/api/browse", handle_browse_files)]
