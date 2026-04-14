input_file = "test.txt"     # your original file
output_file = "test01.txt"   # new file

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        cols = line.strip().split("\t")
        if cols:                     # ensure line is not empty
            f_out.write(cols[0] + "\n")