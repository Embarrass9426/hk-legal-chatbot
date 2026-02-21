# Installation Instructions for WSL2 (Linux)

You selected to **Switch to WSL manually**.

Since I am running in a Windows terminal (`MINGW64_NT`), I cannot directly modify your WSL environment. 
You must **open your WSL terminal (Ubuntu/Debian)** and run the following commands yourself.

## 1. Clean Environment (Run inside WSL)

```bash
# Uninstall existing packages to avoid conflicts
pip uninstall -y onnxruntime onnxruntime-gpu tensorrt

# Install Linux-compatible GPU packages
# This URL provides the Linux builds for CUDA 12
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## 2. Verify Installation (Run inside WSL)

```bash
# Check installed versions
pip list | grep -E "onnx|tensorrt|nvidia|cuda"
```

## 3. Configure Libraries (Add to ~/.bashrc in WSL)

```bash
# Add CUDA libraries to path (if not already present)
# This enables onnxruntime to find the WSL2 GPU drivers mapped from Windows
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 4. Run Ingestion (Run inside WSL)

```bash
cd backend/scripts
python ingest_pdfs.py --force-embed
```

---

**Why this is necessary:**
- Windows uses `onnxruntime-gpu-win_amd64.whl` (which I installed earlier)
- WSL requires `onnxruntime-gpu-linux_x86_64.whl`
- Mixing them will cause "Module not found" or "DLL load failed" errors.

Please execute these commands in your WSL terminal and let me know if you encounter any issues.
