import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pymupdf4llm

logger = logging.getLogger(__name__)
PDF_PARSE_ENCODING = "utf-8"


def _cell_to_str(cell: Any) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _rows_to_markdown_table(rows: List[List[Any]]) -> str:
    """Convert extracted table rows to Markdown so the downstream LLM can inspect table content."""
    clean_rows = [[_cell_to_str(cell) for cell in row] for row in rows if row]
    if not clean_rows:
        return ""

    # Normalize row width.
    width = max(len(row) for row in clean_rows)
    clean_rows = [row + [""] * (width - len(row)) for row in clean_rows]

    header = clean_rows[0]
    separator = ["---"] * width
    body = clean_rows[1:]

    def fmt(row: List[str]) -> str:
        escaped = [cell.replace("|", "\\|") for cell in row]
        return "| " + " | ".join(escaped) + " |"

    lines = [fmt(header), fmt(separator)] + [fmt(row) for row in body]
    return "\n".join(lines)


def _safe_table_extract(table: Any) -> List[List[Any]]:
    try:
        rows = table.extract()
        if isinstance(rows, list):
            return rows
    except Exception as exc:
        logger.debug("Table extraction failed: %s", exc)
    return []


def _extract_pdf_layout_elements(pdf_path: str) -> Dict[str, Any]:
    """
    Extract recoverable tables and raster-image locations for page-level provenance.
    Raster-image content is not interpreted by this function.
    """
    layout: Dict[str, Any] = {"pdf": Path(pdf_path).name, "pages": []}
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        logger.warning("PyMuPDF layout audit unavailable: %s", exc)
        layout["error"] = f"PyMuPDF layout audit unavailable: {exc}"
        return layout

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Could not open PDF for layout audit: %s", exc)
        layout["error"] = f"Could not open PDF for layout audit: {exc}"
        return layout

    try:
        for page_idx, page in enumerate(doc, start=1):
            page_record: Dict[str, Any] = {
                "page": page_idx,
                "image_count": 0,
                "images": [],
                "table_count": 0,
                "tables": [],
            }

            # Figure/raster-image audit.
            try:
                images = page.get_images(full=True)
                page_record["image_count"] = len(images)
                for image_idx, img in enumerate(images, start=1):
                    xref = img[0]
                    rects = []
                    try:
                        rects = [list(map(float, rect)) for rect in page.get_image_rects(xref)]
                    except Exception:
                        rects = []
                    page_record["images"].append({
                        "image_id": image_idx,
                        "xref": xref,
                        "bbox": rects[0] if rects else None,
                        "source_location": f"Page {page_idx}, Figure/Image candidate {image_idx}",
                    })
            except Exception as exc:
                logger.debug("Image audit failed on page %d: %s", page_idx, exc)

            # Table audit with table content when PyMuPDF can recover rows.
            try:
                finder = page.find_tables() if hasattr(page, "find_tables") else None
                tables = getattr(finder, "tables", []) if finder is not None else []
                page_record["table_count"] = len(tables)
                for table_idx, table in enumerate(tables, start=1):
                    rows = _safe_table_extract(table)
                    bbox = getattr(table, "bbox", None)
                    page_record["tables"].append({
                        "table_id": table_idx,
                        "bbox": list(map(float, bbox)) if bbox else None,
                        "n_rows": len(rows),
                        "n_cols": max((len(row) for row in rows), default=0),
                        "source_location": f"Page {page_idx}, Layout-Audit Table {table_idx}",
                        "markdown": _rows_to_markdown_table(rows),
                    })
            except Exception as exc:
                logger.debug("Table audit failed on page %d: %s", page_idx, exc)

            layout["pages"].append(page_record)
    finally:
        doc.close()

    return layout


def _layout_audit_to_markdown(layout: Dict[str, Any]) -> str:
    """Serialize the layout audit into Markdown appended to parsed full text."""
    if not layout or layout.get("error"):
        return "\n\n## M3 PDF Layout Audit\n" + json.dumps(layout, ensure_ascii=False, indent=2)

    lines = [
        "\n\n## M3 PDF Layout Audit: Figures and Tables",
        "This section is generated programmatically during PDF parsing. It preserves page-level table/figure evidence locations for downstream source-traceable extraction. It does not invent visual content when image text is not machine-extractable.",
    ]
    for page in layout.get("pages", []):
        image_count = page.get("image_count", 0)
        table_count = page.get("table_count", 0)
        if not image_count and not table_count:
            continue
        lines.append(f"\n### Page {page.get('page')} layout elements")
        if image_count:
            lines.append(f"- Detected raster figure/image candidates: {image_count}")
            for img in page.get("images", []):
                lines.append(
                    f"  - {img.get('source_location')}; bbox={img.get('bbox')}"
                )
        if table_count:
            lines.append(f"- Detected table candidates: {table_count}")
            for table in page.get("tables", []):
                lines.append(
                    f"\n#### {table.get('source_location')} "
                    f"(rows={table.get('n_rows')}, cols={table.get('n_cols')}, bbox={table.get('bbox')})"
                )
                markdown_table = table.get("markdown")
                if markdown_table:
                    lines.append(markdown_table)
                else:
                    lines.append("<!-- Table structure detected, but cell text could not be reliably extracted. -->")
    if len(lines) == 2:
        lines.append("\nNo additional figure or table candidates were detected by the layout audit.")
    return "\n".join(lines)


def parse_pdf_to_markdown(pdf_path: str, include_layout_audit: bool = True) -> str:
    pdf_path_obj = Path(pdf_path)
    try:
        logger.info(f"Parsing PDF: {pdf_path_obj.name} (Path: {pdf_path_obj.absolute()})")
        md_content = pymupdf4llm.to_markdown(pdf_path)

        if include_layout_audit:
            logger.info("Running page-level PDF layout audit for tables and figures...")
            layout = _extract_pdf_layout_elements(pdf_path)
            md_content = md_content + _layout_audit_to_markdown(layout)
            logger.info(
                "PDF layout audit appended: %d pages inspected.",
                len(layout.get("pages", [])) if isinstance(layout, dict) else 0,
            )

        logger.info(f"PDF parsed successfully, total chars: {len(md_content)}")
        return md_content
    except Exception as e:
        logger.error(f"PDF Parsing Error for {pdf_path_obj.name}: {str(e)}", exc_info=True)
        raise
