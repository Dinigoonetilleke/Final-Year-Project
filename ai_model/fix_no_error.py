rows = []

with open("clean_no_error.csv", "r", encoding="utf-8", errors="ignore") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        # skip header
        if line.lower() == "text,label":
            continue

        # split from the LAST comma only
        if "," in line:
            text, label = line.rsplit(",", 1)
            text = text.strip().strip('"')
            label = label.strip()

            if label == "No_Error":
                rows.append((text, label))

with open("clean_no_error_fixed.csv", "w", encoding="utf-8") as file:
    file.write("text,label\n")
    for text, label in rows:
        text = text.replace('"', "")
        file.write(f'"{text}",{label}\n')

print("Fixed rows:", len(rows))
print("Saved clean_no_error_fixed.csv")