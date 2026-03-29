import ctypes
import os
import sys


def _resolve_cudnn_dll_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    venv_root = os.path.join(repo_root, ".venv")

    candidate_paths = [
        os.path.join(
            venv_root,
            "Lib",
            "site-packages",
            "nvidia",
            "cudnn",
            "bin",
            "cudnn_engines_precompiled64_9.dll",
        ),
        os.path.join(
            venv_root,
            "Lib",
            "site-packages",
            "nvidia",
            "cu13",
            "bin",
            "x86_64",
            "cudnn_engines_precompiled64_9.dll",
        ),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    return candidate_paths[0]


def test_load():
    dll_path = _resolve_cudnn_dll_path()
    bin_dir = os.path.dirname(dll_path)

    print(f"Testing DLL: {dll_path}")
    print(f"Exists: {os.path.exists(dll_path)}")

    if not os.path.exists(bin_dir):
        print(f"SKIP: DLL directory not found: {bin_dir}")
        return

    # Add the bin dir to DLL search
    os.add_dll_directory(bin_dir)

    try:
        ctypes.WinDLL(dll_path)
        print("Success: Loaded with os.add_dll_directory")
    except Exception as e:
        print(f"Failed with os.add_dll_directory: {e}")

    # Check dependencies (heuristic)
    deps = ["cudnn_ops_infer64_9.dll", "cudnn_cnn_infer64_9.dll", "zlibwapi.dll"]
    for dep in deps:
        found = False
        # Check in same dir
        if os.path.exists(os.path.join(bin_dir, dep)):
            print(f"Found {dep} in same dir")
            found = True
        else:
            # Check in path
            for p in os.environ["PATH"].split(os.pathsep):
                if os.path.exists(os.path.join(p, dep)):
                    print(f"Found {dep} in PATH: {p}")
                    found = True
                    break
        if not found:
            print(f"MISSING: {dep}")


if __name__ == "__main__":
    test_load()
