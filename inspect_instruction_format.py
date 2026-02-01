import json

def main():
    path = '/data/kun/jtrans/addressaware/func_blocks_addr.json'
    with open(path, 'r') as f:
        data = json.load(f)

    # Find the first function with 'instructions' field
    for func_id, func in data.items():
        if 'instructions' in func:
            print(f"Function ID: {func_id}")
            print("Sample 'instructions' value:")
            print(func['instructions'])
            break
    else:
        print("No 'instructions' field found in any function.")

if __name__ == "__main__":
    main()
