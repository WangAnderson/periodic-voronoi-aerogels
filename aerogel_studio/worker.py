"""GUI worker entry: file-based events also work in windowed frozen builds."""
from pathlib import Path
import json
import os
import sys
import traceback


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    request = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from .backend import process_command

    def emit(event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False) + "\n"
        if request.get("events_path"):
            with Path(request["events_path"]).open("a", encoding="utf-8") as stream:
                stream.write(line)
        elif sys.stdout is not None:
            print("STUDIO:" + line, end="", flush=True)

    try:
        result = process_command(request, lambda message: emit({"event": "progress", "message": str(message)}))
        emit({"event": "success", "result": result})
        return 0
    except Exception as error:
        emit({"event": "error", "message": str(error), "detail": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
