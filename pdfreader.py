import pymupdf4llm
import pathlib

md_text = pymupdf4llm.to_markdown("data\BNS.pdf")

pathlib.Path("output.txt").write_bytes(md_text.encode("utf8"))