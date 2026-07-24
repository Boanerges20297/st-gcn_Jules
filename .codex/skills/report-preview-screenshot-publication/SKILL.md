---
name: report-preview-screenshot-publication
description: >
  Use this skill when Report Preview must export snapshot artifacts to the
  sibling screenshot-report_preview application, or when that export fails
  during JSON writing, synchronization, commit, or push. Covers the verified
  Windows flow from artifact generation through remote hash confirmation.
license: MIT
metadata:
  author: Codex
  version: "1.0"
---

# Report Preview screenshot publication

Use this procedure for the full contract: generate artifacts, copy only static
data, commit only `public/data`, push `main`, and verify the remote commit.

**Failure pattern:** Windows raises `[Errno 22] Invalid argument` on changing
artifact names, or the dashboard reports success locally without a confirmed
commit and push.

**Verified by:** the full application functions generated and copied 13
artifacts, created commit `d8eabd89bdcb772be4a19cd0c7b8e75ea12d8a55`,
pushed it, and `HEAD` matched `origin/main`.

## Procedure

- [ ] Confirm the dashboard publish button calls
  `exportSnapshotToScreenshotApp(true)` and sends `publish_repo: true`.
- [ ] Inspect `app.py` functions `_sync_static_snapshot_to_screenshot_app()` and
  `_publish_screenshot_repo()`. Keep export, copy, commit, and push in that order.
- [ ] Keep JSON writes in `scripts/export_static_snapshot.py::_write_json()`
  isolated with a unique same-directory temporary file. Replace atomically and
  retry transient Windows errors 13 and 22; never share a fixed `.tmp` filename.
- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py scripts\export_static_snapshot.py
.\.venv\Scripts\python.exe -m unittest tests.test_export_static_snapshot
```

- [ ] Execute the actual application flow, not a synthetic copy:

```powershell
@'
import json
import os
import app

target_data = os.path.join(app.STATIC_SCREENSHOT_REPO_DIR, 'public', 'data')
sync_info = app._sync_static_snapshot_to_screenshot_app(target_data)
publish_info = app._publish_screenshot_repo(sync_info['target_repo_dir'])
print(json.dumps({
    'copied_count': len(sync_info.get('copied_files', [])),
    'publish': publish_info,
}, ensure_ascii=False, indent=2))
'@ | .\.venv\Scripts\python.exe -
```

- [ ] Require `published`, `commit_created`, and `push_executed` to be true.
- [ ] Verify the pushed state:

```powershell
git -C ..\screenshot-report_preview fetch origin main
git -C ..\screenshot-report_preview rev-parse HEAD
git -C ..\screenshot-report_preview rev-parse origin/main
git -C ..\screenshot-report_preview status --short
```

- [ ] If the served process predates the source fix, restart only this
  workspace's `app.py` process and require
  `/api/export_static_snapshot/status` to return `idle` with `error: null`.

## Gotchas

- The publish button is `btn-export-screenshot-publish`; a local-only export is
  not equivalent to the requested workflow.
- `_publish_screenshot_repo()` must stage only `public/data`. Preserve unrelated
  edits in the screenshot repository.
- Do not require `git pull --rebase` before this scoped publication. It can be
  blocked by unrelated local work.
- Git identity belongs in the screenshot repository's local configuration;
  never put credentials or tokens in this skill.
- A passing writer unit test is necessary but insufficient. The final check is
  equality between local `HEAD` and `origin/main`.

## What didn't work

- Direct `Path.write_text()` to the final artifact: readers and Windows file
  handling can expose or reject the final path during publication.
- One fixed temporary filename such as `micronodes.geojson.tmp`: concurrent or
  stale processes can collide on it, moving the failure to another artifact.
- Testing only `explainability.json`: the next full run failed on
  `micronodes.geojson`; validate both large artifacts and the complete publish.
- Editing source without restarting an older served process: the dashboard kept
  the old module and stale error status until the app was restarted.
