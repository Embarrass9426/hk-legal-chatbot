import argparse
import os
import sys
from typing import cast

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core import setup_env

setup_env.setup_cuda_dlls()


def _resolve_pdf_path(cap: str) -> str:
    backend_pdf_dir = os.path.join(project_root, "backend", "data", "pdfs")
    filename = f"cap{cap}.pdf"
    return os.path.join(backend_pdf_dir, filename)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump raw Unstructured element text for a single PDF page"
    )
    _ = parser.add_argument("--cap", default="232", help="Cap number, e.g. 232 or 232A")
    _ = parser.add_argument("--page", type=int, default=48, help="1-based page number")
    _ = parser.add_argument(
        "--strategy",
        default="fast",
        choices=["fast", "hi_res", "ocr_only", "auto"],
        help="partition_pdf strategy",
    )
    _ = parser.add_argument(
        "--output",
        default="",
        help="Optional output text file path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()
    cap = cast(str, args.cap)
    page = cast(int, args.page)
    strategy = cast(str, args.strategy)
    output = cast(str, args.output)

    try:
        from unstructured.partition.pdf import partition_pdf
    except ModuleNotFoundError as exc:
        print(
            "ERROR: Missing dependency for unstructured PDF parsing: "
            + f"{exc}. Use the project venv interpreter (e.g. .\\.venv\\Scripts\\python.exe) "
            + "after installing backend requirements."
        )
        return 3

    pdf_path = _resolve_pdf_path(cap)
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}")
        return 2

    elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy,
        include_page_breaks=False,
        infer_table_structure=True,
        languages=["eng"],
    )

    page_elements: list[tuple[int, str, str]] = []
    for idx, element in enumerate(elements, start=1):
        metadata = getattr(element, "metadata", None)
        page_no = getattr(metadata, "page_number", None)
        if page_no != page:
            continue

        category = str(getattr(element, "category", "Unknown"))
        text = str(getattr(element, "text", "") or "").strip()
        if not text:
            continue

        page_elements.append((idx, category, text))

    if not page_elements:
        print(f"No non-empty Unstructured elements found for cap{cap}.pdf page {page}.")
        return 1

    blocks = [
        "# Raw Unstructured text dump\n"
        + f"cap={cap} page={page} strategy={strategy}\n"
        + f"pdf={pdf_path}\n"
    ]

    for idx, category, text in page_elements:
        blocks.append(f"--- element_index={idx} category={category} ---\n{text}\n")

    output_text = "\n".join(blocks).rstrip() + "\n"

    if output:
        output_path = os.path.abspath(output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            _ = file.write(output_text)
        print(f"Wrote raw text to: {output_path}")
    else:
        print(output_text)

    print(f"Total elements on page {page}: {len(page_elements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
