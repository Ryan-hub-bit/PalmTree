import json

def count_tokens(instr_str):
    return len(instr_str.split())

def main():
    path = '/data/kun/jtrans/addressaware/func_blocks_addr.json'
    with open(path, 'r') as f:
        data = json.load(f)

    total = 0
    over_512 = 0

    for func in data.values():
        if 'instructions' in func:
            instr_str = func['instructions']
        elif 'tokens' in func:
            instr_str = func['tokens']
        else:
            continue

        token_count = count_tokens(instr_str)
        total += 1
        if token_count > 512:
            over_512 += 1

    if total == 0:
        print("No functions found.")
    else:
        percent = 100.0 * over_512 / total
        print(f"Functions with >512 tokens: {over_512}/{total} ({percent:.2f}%)")

if __name__ == '__main__':
    main()