#!/bin/bash
set -e

# Ensure we are in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Error: Please activate your virtual environment first."
    exit 1
fi

echo "Installing TensorRT and ONNX Runtime GPU..."
pip install --upgrade pip
pip install tensorrt==10.0.1 onnxruntime-gpu

# Find where tensorrt libs are installed
TRT_LIB_PATH=$(python3 -c "import site; import os; print(os.path.join(site.getsitepackages()[0], 'tensorrt_libs'))")

if [ -d "$TRT_LIB_PATH" ]; then
    echo "Found TensorRT libs at: $TRT_LIB_PATH"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_LIB_PATH
    echo "Exported LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    
    # Create a helper script to run python with the correct environment
    echo "#!/bin/bash" > backend/run_with_trt.sh
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$TRT_LIB_PATH" >> backend/run_with_trt.sh
    echo "exec \"\$@\"" >> backend/run_with_trt.sh
    chmod +x backend/run_with_trt.sh
    echo "Created backend/run_with_trt.sh wrapper script."
else
    echo "Warning: Could not find tensorrt_libs directory."
fi

echo "Verifying installation..."
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
python3 -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__} (Device: {onnxruntime.get_device()})')"
