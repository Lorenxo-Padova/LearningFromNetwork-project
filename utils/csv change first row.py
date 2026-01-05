from pathlib import Path

path = Path("data/clean_vitagraph_small.csv")

with path.open("r", encoding="utf-8") as f:
    lines = f.readlines()

# Add Source,Target as first line (header)
lines.insert(0, "Source,Target\n")
# Add #heterogeneous as the very first line
lines.insert(0, "#heterogeneous\n")

with path.open("w", encoding="utf-8") as f:
    f.writelines(lines)
    