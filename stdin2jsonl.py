
import sys
import json

def main():
    for idx, line in enumerate(sys.stdin, 1):
        line = line.rstrip("\r\n")
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 3:
            print(f"Skipping line #{idx} (does not have 3 columns): {line}", file=sys.stderr)
            continue
        obj = {
            "source": parts[0],
            "hypothesis": parts[1],
            "reference": parts[2]
        }
        print(json.dumps(obj, ensure_ascii=False))

if __name__ == "__main__":
    main()
