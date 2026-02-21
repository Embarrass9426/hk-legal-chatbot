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

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if trt_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = trt_dir + (":" + ld_path if ld_path else "")

    so_files = sorted(_glob.glob(os.path.join(trt_dir, "libnvinfer*.so*")))
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

    # Prioritize Torch DLLs (they are generally the most compatible)
    if os.path.exists(torch_lib):
        print(f"Adding Torch DLLs: {torch_lib}")
        try:
            os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
        except Exception:
            pass

    # Add ALL Nvidia bin folders found in site-packages
    if os.path.exists(nvidia_base):
        # We need to find all subfolders containing 'bin'
        for root, dirs, files in os.walk(nvidia_base):
            if "bin" in dirs:
                bin_path = os.path.join(root, "bin")
                print(f"Adding Nvidia DLLs: {bin_path}")
                try:
                    os.add_dll_directory(bin_path)
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                except Exception:
                    pass

    # Extra check for zlibwapi.dll which Paddle often misses
    zlib_path = os.path.join(torch_lib, "zlibwapi.dll")
    if os.path.exists(zlib_path):
        # Already added torch_lib, but just ensure it is in PATH
        if torch_lib not in os.environ["PATH"]:
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]

    # Add TensorRT DLLs
    trt_libs = os.path.join(venv_root, "Lib", "site-packages", "tensorrt_libs")
    if os.path.exists(trt_libs):
        print(f"Adding TensorRT DLLs: {trt_libs}")
        try:
            os.add_dll_directory(trt_libs)
            os.environ["PATH"] = trt_libs + os.pathsep + os.environ["PATH"]
        except Exception:
            pass


if __name__ == "__main__":
    setup_cuda_dlls()
    try:
        import torch

        print(f"Success: Torch (CUDA: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"Setup failed: {e}")
