#!/usr/bin/env python3
import sys

def remove_empty_lines(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # count and filter
    empty_count = sum(1 for line in lines if line == "\n")
    cleaned = [line for line in lines if line != "\n"]

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(cleaned)

    print(f"Removed {empty_count} empty lines from {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_empty_lines.py <file>")
        sys.exit(1)
    remove_empty_lines(sys.argv[1])

