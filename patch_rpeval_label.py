from pathlib import Path

p = Path("frontiermem/external_data.py")
s = p.read_text(encoding="utf-8")

anchor = '        "supportive": "SUPPORT",\n'

label1 = "\u652f\u6301\u6027\u504f\u597d"
label2 = "\u652f\u652f\u6301\u6301\u6027\u6027\u504f\u504f\u597d\u597d"

addition = (
    f'        "{label1}": "SUPPORT",\n'
    f'        "{label2}": "SUPPORT",\n'
)

if label1 not in s or label2 not in s:
    if anchor not in s:
        raise RuntimeError("Could not find supportive mapping anchor")
    s = s.replace(anchor, anchor + addition)

p.write_text(s, encoding="utf-8")

print("patched_standard =", label1 in s)
print("patched_doubled  =", label2 in s)
