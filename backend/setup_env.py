import os
import sys

def setup_cuda_dlls():
    """
    Sets up the necessary DLL directories for Torch, PaddlePaddle, and Unstructured on Windows.
    Solves [WinError 127] by prioritizing specific DLL paths.
    """
    if sys.platform != 'win32':
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_root = os.path.abspath(os.path.join(script_dir, "..", ".venv"))
    if not os.path.exists(venv_root):
        # Fallback to current directory logic
        venv_root = os.path.abspath(os.path.join(os.getcwd(), ".venv"))

    torch_lib = os.path.join(venv_root, "Lib", "site-packages", "torch", "lib")
    nvidia_base = os.path.join(venv_root, "Lib", "site-packages", "nvidia")

    # Prioritize Torch DLLs (they are generally the most compatible)
    if os.path.exists(torch_lib):
        print(f"Adding Torch DLLs: {torch_lib}")
        try:
            os.add_dll_directory(torch_lib)
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
        except Exception: pass

    # Add ALL Nvidia bin folders found in site-packages
    if os.path.exists(nvidia_base):
        # We need to find all subfolders containing 'bin'
        for root, dirs, files in os.walk(nvidia_base):
            if 'bin' in dirs:
                bin_path = os.path.join(root, 'bin')
                print(f"Adding Nvidia DLLs: {bin_path}")
                try:
                    os.add_dll_directory(bin_path)
                    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                except Exception: pass
    
    # Extra check for zlibwapi.dll which Paddle often misses
    zlib_path = os.path.join(torch_lib, "zlibwapi.dll")
    if os.path.exists(zlib_path):
        # Already added torch_lib, but just ensure it is in PATH
        if torch_lib not in os.environ['PATH']:
             os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']

    # Add TensorRT DLLs
    trt_libs = os.path.join(venv_root, "Lib", "site-packages", "tensorrt_libs")
    if os.path.exists(trt_libs):
        print(f"Adding TensorRT DLLs: {trt_libs}")
        try:
            os.add_dll_directory(trt_libs)
            os.environ['PATH'] = trt_libs + os.pathsep + os.environ['PATH']
        except Exception: pass

if __name__ == "__main__":
    setup_cuda_dlls()
    try:
        import torch
        print(f"Success: Torch (CUDA: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"Setup failed: {e}")
