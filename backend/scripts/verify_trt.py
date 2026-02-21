import os
import sys

# Ensure project root is in sys.path (scripts -> backend -> project root = 3 levels)
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.core import setup_env

setup_env.setup_cuda_dlls()

from backend.services.vector_store import BoostedYuanEmbeddings


def verify_trt():
    print("[INFO] Verifying TensorRT Execution Provider...")

    # 1. Initialize Model (triggers engine build)
    start_time = time.time()
    # Go up one level: scripts -> backend -> models
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "yuan-onnx-trt"
    )

    try:
        embeddings = BoostedYuanEmbeddings(model_path)
        print(f"Model initialized in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return

    # 2. Check Providers
    providers = embeddings.model.model.get_providers()
    print(f"[INFO] Active Providers: {providers}")

    if "TensorrtExecutionProvider" not in providers:
        print("WARNING: TensorrtExecutionProvider is NOT active!")
    else:
        print("TensorrtExecutionProvider is ACTIVE")

    # 3. Generate Embedding
    test_text = "The quick brown fox jumps over the lazy dog."
    print(f"Generating embedding for: '{test_text}'")

    try:
        start_embed = time.time()
        vector = embeddings.embed_query(test_text)
        duration = time.time() - start_embed

        print(f"Embedding generated in {duration:.4f} seconds")
        print(f"Vector dimension: {len(vector)}")

        if len(vector) == 1024:
            print("Dimension check passed (1024)")
        else:
            print(f"Dimension check failed! Expected 1024, got {len(vector)}")

    except Exception as e:
        print(f"Embedding generation failed: {e}")


if __name__ == "__main__":
    verify_trt()
