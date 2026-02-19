import os
import shutil
import tempfile
import json

# Make sure repo root is in path
ROOT = os.getcwd()
import sys
sys.path.append(ROOT)

import app


def run_dryrun():
    # Create a temp workspace root that mimics the repo layout
    tmp_root = tempfile.mkdtemp(prefix='stgcn_dryrun_')
    try:
        data_dir = os.path.join(tmp_root, 'data')
        os.makedirs(data_dir, exist_ok=True)

        src_path = os.path.join(ROOT, 'data', 'exogenous_events.json')
        dst_path = os.path.join(data_dir, 'exogenous_events.json')
        shutil.copy2(src_path, dst_path)

        # Point app to the temp root and run archive
        original_base = getattr(app, 'BASE_DIR', None)
        app.BASE_DIR = tmp_root

        print('Temporary workspace prepared at:', tmp_root)
        print('Running archive_old_exogenous_events() (dry-run target) ...')
        # Call the function (it will operate only on the temp workspace)
        app.archive_old_exogenous_events()

        # Report created archives
        archives_dir = os.path.join(data_dir, 'archives')
        if os.path.exists(archives_dir):
            files = sorted(os.listdir(archives_dir))
            print('\nArchives created:')
            for f in files:
                p = os.path.join(archives_dir, f)
                try:
                    with open(p, 'r', encoding='utf-8') as fh:
                        arr = json.load(fh)
                    print(f" - {f}: {len(arr)} events")
                except Exception as e:
                    print(f" - {f}: (could not read) {e}")
        else:
            print('\nNo archives created (nothing older than cutoff).')

        # Show resulting remaining events count in temp main file
        try:
            with open(dst_path, 'r', encoding='utf-8') as mf:
                remaining = json.load(mf)
            print(f"\nRemaining events in temp exogenous_events.json: {len(remaining)}")
        except Exception as e:
            print('Could not read temp main events file:', e)

    finally:
        # cleanup
        app.BASE_DIR = original_base
        shutil.rmtree(tmp_root)
        print('\nTemporary workspace removed.')


if __name__ == '__main__':
    run_dryrun()
