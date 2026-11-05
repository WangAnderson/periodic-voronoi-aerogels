"""Run in Python/Spyder, or package this entry point with PyInstaller."""
import os
import sys

os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        from aerogel_studio.worker import main as worker_main
        return worker_main([sys.argv[2]])
    from aerogel_studio.app import run
    return run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
