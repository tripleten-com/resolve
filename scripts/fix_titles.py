"""Quote every `title:` value in the knowledge-base front matter so YAML parses titles that contain colons."""
import glob
import json
import re

fixed = []
for path in glob.glob("data/kb/*/*.md"):
    text = open(path, encoding="utf-8").read()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    front = m.group(1)
    lines = []
    for line in front.split("\n"):
        if line.startswith("title:"):
            value = line[len("title:"):].strip()
            if not (value.startswith('"') or value.startswith("'")):
                line = "title: " + json.dumps(value)
        lines.append(line)
    new_front = "\n".join(lines)
    if new_front != front:
        open(path, "w", encoding="utf-8").write(text.replace(front, new_front, 1))
        fixed.append(path)
print(len(fixed), "files fixed")
for p in fixed:
    print("  ", p)

gen = "scripts/gen_kb.py"
src = open(gen, encoding="utf-8").read()
old = "f\"title: {item['title']}\\n\""
if old in src:
    src = src.replace(old, "f\"title: {json.dumps(item['title'])}\\n\"")
    if "import json" not in src:
        src = "import json\n" + src
    open(gen, "w", encoding="utf-8").write(src)
    print("generator patched")
else:
    print("generator: title line not found; check manually")
