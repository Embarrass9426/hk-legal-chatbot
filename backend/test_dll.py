import ctypes
import os
import sys

def test_load():
    dll_path = r'C:\Users\Embarrass\Desktop\vscode\hk-legal-chatbot\.venv\Lib\site-packages\nvidia\cudnn\bin\cudnn_engines_precompiled64_9.dll'
    bin_dir = os.path.dirname(dll_path)
    
    print(f"Testing DLL: {dll_path}")
    print(f"Exists: {os.path.exists(dll_path)}")
    
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
