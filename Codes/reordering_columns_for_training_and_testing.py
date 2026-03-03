input_file = "or-token-pos-morph-with-vibh-chunk-17022026.txt"       # original tab-separated file (no header)
output_file = "17022026-train-file.txt"  # output file

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.rstrip("\n")
        if not line.strip():
            fout.write("\n")
            continue

        cols = line.split("\t")

        # Expected original order:
        # 0 token, 1 pos, 2 lcat, 3 gender, 4 number, 5 person, 6 case, 7 vibhakti, 8 chunk
        # New order:
        # token, lcat, gender, number, person, case, vibhakti, pos, chunk
        if len(cols) >= 9:
            new_cols = [cols[0], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[1], cols[8]]
            fout.write("\t".join(new_cols) + "\n")
        else:
            # Keep malformed/short lines as-is
            fout.write(line + "\n")

print(f"Done. Reordered file saved as: {output_file}")
