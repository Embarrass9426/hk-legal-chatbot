import os
import json
import sys
import torch
from unstructured.partition.pdf import partition_pdf

# 1. Verify GPU Availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
if cuda_available:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Add local Poppler to PATH
poppler_path = os.path.join(os.getcwd(), "backend", "bin", "poppler", "poppler-24.08.0", "Library", "bin")
if os.path.exists(poppler_path):
    os.environ["PATH"] += os.pathsep + poppler_path
    print(f"Added Poppler to PATH: {poppler_path}")

# Add TensorRT and CUDA DLLs from venv to PATH
venv_site_packages = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
gpu_paths = [
    os.path.join(venv_site_packages, "tensorrt_libs"),
    os.path.join(venv_site_packages, "nvidia", "cu13", "bin", "x86_64"),
    os.path.join(venv_site_packages, "nvidia", "cuda_runtime", "bin"),
]
for path in gpu_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        print(f"Added to PATH: {path}")

# Add Tesseract to PATH (Common default installation paths)
tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR",
    r"C:\Users\Embarrass\AppData\Local\Tesseract-OCR"
]
for path in tesseract_paths:
    if os.path.exists(path):
        os.environ["PATH"] += os.pathsep + path
        print(f"Added Tesseract to PATH: {path}")
        break

def test_parse_specifics(file_path, target_page=None, target_section_keyword=None):
    print(f"\n--- Parsing: {os.path.basename(file_path)} ---")
    
    try:
        # GPU acceleration enabled for RTX 5060 Ti (Blackwell)
        model_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {model_device}")
        
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=["eng"],
            model_device=model_device,
            pdf_image_dpi=200
        )
    except Exception as e:
        print(f"HI_RES Failed: {e}")
        print("Falling back to 'fast' strategy to show content...")
        elements = partition_pdf(filename=file_path, strategy="fast")
        
    parsed_data = []
    for el in elements:
        el_dict = el.to_dict()
        parsed_data.append({
            "type": el_dict.get("type"),
            "text": el_dict.get("text"),
            "page_number": el_dict.get("metadata", {}).get("page_number")
        })

    # Filter for target page if requested
    if target_page:
        print(f"\n--- Page {target_page} Content ---")
        page_elements = [el for el in parsed_data if el["page_number"] == target_page]
        for el in page_elements:
            print(f"[{el['type']}] {el['text'][:200]}...")

    # Filter for section keyword (Basic search for test)
    if target_section_keyword:
        print(f"\n--- Section with keyword '{target_section_keyword}' ---")
        found_indices = []
        for i, el in enumerate(parsed_data):
            if target_section_keyword.lower() in el["text"].lower():
                found_indices.append(i)
        
        if found_indices:
            # Pick a "good" match - often headers are repeated, we want one that looks like a section start
            # For Cap 282 Section 5, it usually starts with "5. Employer's liability"
            best_idx = found_indices[0]
            for idx in found_indices:
                if "employer" in parsed_data[idx]["text"].lower():
                    best_idx = idx
                    break
            
            print(f"Showing content around match {best_idx} (Page {parsed_data[best_idx]['page_number']}):")
            for i in range(max(0, best_idx - 1), min(best_idx + 8, len(parsed_data))):
                print(f"[{parsed_data[i]['type']}] {parsed_data[i]['text']}")
        else:
            print(f"Could not find section containing '{target_section_keyword}'")

if __name__ == "__main__":
    pdf_dir = os.path.join(os.getcwd(), "backend", "data", "pdfs")
    
    # Cap 1C specifics: Page 4
    cap1c_path = os.path.join(pdf_dir, "cap1C.pdf")
    if os.path.exists(cap1c_path):
        test_parse_specifics(cap1c_path, target_page=4)
        
    # Cap 282 specifics: Show Section 5
    cap282_path = os.path.join(pdf_dir, "cap282.pdf")
    if os.path.exists(cap282_path):
        test_parse_specifics(cap282_path, target_section_keyword="Section 5")
