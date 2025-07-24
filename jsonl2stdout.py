
import sys
import json

def main():
    # Read all lines from stdin
    lines = [json.loads(line) for line in sys.stdin if line.strip()]

    # Sort lines using the float value of the "prediction" key
    lines.sort(key=lambda x: float(x["prediction"]))

    # Print the required columns, tab-separated
    for item in lines:
        print(f"{item['prediction']}\t{item['source']}\t{item['reference']}\t{item['hypothesis']}")

if __name__ == "__main__":
    main()
