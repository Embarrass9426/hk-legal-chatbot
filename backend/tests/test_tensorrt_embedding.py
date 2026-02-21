"""
Test script to verify TensorRT embedding generation works correctly.
"""

import sys
import os

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from services.embedding_service import EmbeddingService


def main():
    print("=" * 60)
    print("TensorRT Embedding Test")
    print("=" * 60)

    # Initialize service (loads model with TensorRT)
    print("\n[1] Initializing EmbeddingService...")
    svc = EmbeddingService()

    # Check active providers
    print("\n[2] Checking active providers...")
    providers = svc.model.model.get_providers()
    print(f"   Active Providers: {providers}")

    if "TensorrtExecutionProvider" not in providers:
        print("   [FAIL] TensorRT not in active providers!")
        return 1
    print("   [PASS] TensorRT is active")

    # Generate embeddings for test texts
    print("\n[3] Generating embeddings for sample texts...")
    test_texts = [
        "Section 1. General Provisions",
        "This ordinance shall apply to all persons within Hong Kong",
        "Interpretation: In this Ordinance, unless the context otherwise requires",
    ]

    try:
        embeddings = svc.embed_documents(test_texts)
        print(f"   Generated {len(embeddings)} embeddings")

        # Verify embeddings are non-zero
        import numpy as np

        for i, emb in enumerate(embeddings):
            emb_array = np.array(emb)
            norm = np.linalg.norm(emb_array)
            print(f"   Text {i + 1}: embedding norm = {norm:.6f}")

            if norm < 1e-6:
                print(f"   [FAIL] Embedding {i + 1} is nearly zero!")
                return 1

        print("   [PASS] All embeddings are non-zero")

    except Exception as e:
        print(f"   [FAIL] Embedding generation failed: {e}")
        return 1

    print("\n" + "=" * 60)
    print("[ALL TESTS PASSED]")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
