"""Tiny local fallback for pandas.DataFrame.to_markdown in offline demos.

This is not the full third-party tabulate package. It provides the small subset
needed by the generated research report when the optional dependency is absent.
"""

__version__ = "0.9.0"


def tabulate(tabular_data, headers=(), tablefmt="pipe", showindex=False, **kwargs):
    rows = [list(row) for row in tabular_data]
    header = list(headers) if headers not in (None, "keys") else []
    if not header and rows:
        header = [f"col_{i}" for i in range(len(rows[0]))]
    def cell(value):
        return str(value).replace("|", " ")
    lines = []
    if header:
        lines.append("| " + " | ".join(cell(h) for h in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)
