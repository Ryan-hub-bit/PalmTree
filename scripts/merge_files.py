import os

# def merge_files(input_folder, output_file="dfg_train.txt"):
def merge_files(input_folder, output_file="cfg_train.txt"):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(input_folder):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                # Skip the output file itself if it’s inside the same folder
                if fpath == os.path.abspath(output_file):
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")  # separate files with newline
    print(f"✅ All files merged into {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python merge_files.py <folder_path>")
    else:
        merge_files(sys.argv[1])
