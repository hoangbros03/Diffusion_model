from pathlib import Path

def check_and_create_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Create path: {str(path)}")
    else:
        print(f"Path {str(path)} is already existed")