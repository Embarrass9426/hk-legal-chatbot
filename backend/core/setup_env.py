import os
import sys
import ctypes
import glob as _glob


def _setup_linux_trt_libs():
    """
    Pre-load TensorRT shared libraries on Linux/WSL so that onnxruntime
    can find the TensorRT execution provider without a wrapper script.
    Must be called BEFORE importing onnxruntime.
    """
    trt_dir = None

    venv_prefix = getattr(sys, "prefix", None)
    if venv_prefix:
        candidate = os.path.join(
            venv_prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
            "tensorrt_libs",
        )
        if os.path.isdir(candidate):
            trt_dir = candidate

    if trt_dir is None:
        try:
            import site

            for sp in site.getsitepackages():
                candidate = os.path.join(sp, "tensorrt_libs")
                if os.path.isdir(candidate):
                    trt_dir = candidate
                    break
        except Exception:
            pass

    if trt_dir is None:
        print("[setup_env] tensorrt_libs directory not found; skipping TRT preload.")
        return

    print(f"[setup_env] Found tensorrt_libs: {trt_dir}")

    search_paths = [trt_dir]

    venv_prefix = getattr(sys, "prefix", "")
    if venv_prefix:
        nvidia_base = os.path.join(
            venv_prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
            "nvidia",
        )
        if os.path.isdir(nvidia_base):
            for root, dirs, _files in os.walk(nvidia_base):
                if os.path.basename(root) == "lib":
                    search_paths.append(root)

    dedup_paths = []
    for p in search_paths:
        if p and p not in dedup_paths:
            dedup_paths.append(p)

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    existing = [p for p in ld_path.split(":") if p]
    merged = dedup_paths + [p for p in existing if p not in dedup_paths]
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    patterns = [
        "libnvinfer*.so*",
        "libnvonnxparser*.so*",
        "libcudart*.so*",
        "libcublas*.so*",
        "libcudnn*.so*",
    ]
    so_files = []
    for path in dedup_paths:
        for pattern in patterns:
            so_files.extend(_glob.glob(os.path.join(path, pattern)))

    so_files = sorted(set(so_files))
    loaded = 0
    for so_path in so_files:
        try:
            ctypes.CDLL(
                so_path, mode=ctypes.RTLD_GLOBAL
            )  # RTLD_GLOBAL: onnxruntime needs TRT symbols visible
            loaded += 1
        except OSError as e:
            print(f"[setup_env] Could not load {os.path.basename(so_path)}: {e}")

    if loaded:
        print(f"[setup_env] Pre-loaded {loaded} TensorRT libraries from {trt_dir}")
    else:
        print("[setup_env] WARNING: No TensorRT .so files could be loaded.")


def setup_cuda_dlls():
    """
    Sets up CUDA/TensorRT library paths so onnxruntime and torch find their providers.
    On Windows: adds DLL directories. On Linux/WSL: pre-loads TensorRT .so files.
    Must be called BEFORE importing torch or onnxruntime.
    """
    if sys.platform != "win32":
        _setup_linux_trt_libs()
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".venv"))

    if not os.path.exists(venv_root):
        # Check backend/.venv
        backend_venv = os.path.abspath(os.path.join(script_dir, "..", ".venv"))
        if os.path.exists(backend_venv):
            venv_root = backend_venv
        else:
            # Fallback to current directory logic
            venv_root = os.path.abspath(os.path.join(os.getcwd(), ".venv"))

    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    nvidia_base = os.path.join(venv_root, "Lib", "site-packages", "nvidia")
    onnxruntime_capi = os.path.join(
        venv_root,
        "Lib",
        "site-packages",
        "onnxruntime",
        "capi",
    )

    def _add_dll_dir(path: str, label: str):
        if not path or not os.path.isdir(path):
            return
        print(f"Adding {label}: {path}")
        try:
            os.add_dll_directory(path)
        except Exception:
            pass
        if path not in os.environ["PATH"]:
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

    # Prioritize Torch DLLs (they are generally the most compatible)
    if os.path.exists(torch_lib):
        _add_dll_dir(torch_lib, "Torch DLLs")

    if os.path.exists(onnxruntime_capi):
        _add_dll_dir(onnxruntime_capi, "ONNX Runtime DLLs")

    # Add ALL Nvidia bin folders found in site-packages
    if os.path.exists(nvidia_base):
        seen_paths = set()
        for root, dirs, files in os.walk(nvidia_base):
            for folder in ("bin", "lib", "x64", "x86_64"):
                if folder in dirs:
                    candidate = os.path.join(root, folder)
                    if candidate not in seen_paths:
                        seen_paths.add(candidate)
                        _add_dll_dir(candidate, "Nvidia DLLs")

            has_dll = any(name.lower().endswith(".dll") for name in files)
            if has_dll and root not in seen_paths:
                seen_paths.add(root)
                _add_dll_dir(root, "Nvidia DLLs")

    # Extra check for zlibwapi.dll which Paddle often misses
    zlib_path = os.path.join(torch_lib, "zlibwapi.dll")
    if os.path.exists(zlib_path):
        # Already added torch_lib, but just ensure it is in PATH
        if torch_lib not in os.environ["PATH"]:
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]

    # Add TensorRT DLLs
    trt_libs = os.path.join(venv_root, "Lib", "site-packages", "tensorrt_libs")
    if os.path.exists(trt_libs):
        _add_dll_dir(trt_libs, "TensorRT DLLs")


if __name__ == "__main__":
    setup_cuda_dlls()
    try:
        import torch

        print(f"Success: Torch (CUDA: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"Setup failed: {e}")
