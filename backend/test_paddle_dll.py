import os
import ctypes
import sys

def test_load_dlls():
    venv_site_packages = r"C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\.venv\Lib\site-packages"
    
    # Order matters sometimes
    dirs_to_add = [
        os.path.join(venv_site_packages, "nvidia", "cuda_runtime", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cublas", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(venv_site_packages, "torch", "lib"),
    ]
    
    for d in dirs_to_add:
        if os.path.exists(d):
            print(f"Adding DLL directory: {d}")
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

    target_dll = os.path.join(venv_site_packages, "nvidia", "cudnn", "bin", "cudnn_engines_precompiled64_9.dll")
    
    print(f"Attempting to load: {target_dll}")
    try:
        lib = ctypes.WinDLL(target_dll)
        print("Successfully loaded DLL with ctypes!")
    except Exception as e:
        print(f"Failed to load DLL with ctypes: {e}")
        
    print("\nImporting paddle...")
    try:
        import paddle
        print("Paddle imported successfully!")
        print(f"Paddle CUDA: {paddle.device.is_compiled_with_cuda()}")
    except Exception as e:
        print(f"Paddle import failed: {e}")

if __name__ == "__main__":
    test_load_dlls()
