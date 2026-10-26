"""
Full repair of Desktop projet-cbir/src/cbir/search.py:
- Remove BOM if present
- Remove all duplicated/corrupted _DESCRIPTOR_KEYS garbage
- Insert one clean block
- Verify syntax
"""
import ast

TARGET = r"C:\Users\USER\Desktop\projet-cbir\src\cbir\search.py"

with open(TARGET, "r", encoding="utf-8-sig", errors="replace") as f:  # utf-8-sig strips BOM
    content = f.read()

# Find the start of _DESCRIPTOR_KEYS and def _optimized_search
start = content.find("_DESCRIPTOR_KEYS")
end = content.find("\ndef _optimized_search")
if start < 0 or end < 0 or end <= start:
    print(f"ERROR: start={start}, end={end}")
    raise SystemExit(1)

print(f"Replacing chars {start} to {end}")

NEW_BLOCK = '''_DESCRIPTOR_KEYS = {
    "color":   {"hsv_grid3_8_4_4"},
    "hsv":     {"hsv_grid3_8_4_4"},
    "texture": {"lbp_grid3_P8_R1"},
    "lbp":     {"lbp_grid3_P8_R1"},
    "shape":   {"hog_o9_c16x16_b2x2"},
    "hog":     {"hog_o9_c16x16_b2x2"},
    "mpeg7":   {"cld_6_3_3", "ehd_4x4_e5", "scd_16_4_4"},
    "cld":     {"cld_6_3_3"},
    "ehd":     {"ehd_4x4_e5"},
    "scd":     {"scd_16_4_4"},
}


'''

content = content[:start] + NEW_BLOCK + content[end+1:]

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(content)

print("File written.")

try:
    ast.parse(content)
    print("Syntax OK!")
except SyntaxError as e:
    print(f"Syntax Error at line {e.lineno}: {e.msg}")
    # Show context
    lines = content.splitlines()
    for i in range(max(0, e.lineno - 3), min(len(lines), e.lineno + 3)):
        print(f"  {i+1}: {lines[i]}")
