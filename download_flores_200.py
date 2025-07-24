
import sys

from datasets import load_dataset

def main():
    src_lang = sys.argv[1] # e.g., "eng_Latn"
    tgt_lang = sys.argv[2] # e.g., "spa_Latn"
    split = sys.argv[3] if len(sys.argv) > 3 else "dev"

    dataset = load_dataset("facebook/flores", "all", split=split, trust_remote_code=True)
    data = dataset

    src_key = f"sentence_{src_lang}"
    tgt_key = f"sentence_{tgt_lang}"

    if src_key not in data.column_names or tgt_key not in data.column_names:
        print(f"Error: One of the provided language codes is invalid or not in FLORES-200.")
        sys.exit(1)

    for example in data:
        src_sentence = example[src_key].replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        tgt_sentence = example[tgt_key].replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
        print(f"{src_sentence}\t{tgt_sentence}")

if __name__ == "__main__":
    main()
