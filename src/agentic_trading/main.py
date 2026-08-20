import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_src = str(Path(__file__).resolve().parents[1])
    if project_src not in sys.path:
        sys.path.insert(0, project_src)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from agentic_trading.cli import main as cli_main

        cli_main(sys.argv[2:])
    else:
        dashboard_app = Path(__file__).resolve().parent / "gui" / "dashboard.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_app)], check=True)


if __name__ == "__main__":
    main()
