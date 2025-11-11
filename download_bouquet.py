
import sys

from datasets import load_dataset
import pandas as pd

def main():
    src_lang = sys.argv[1] # e.g., "eng_Latn"
    tgt_lang = sys.argv[2] # e.g., "spa_Latn"
    split = sys.argv[3] if len(sys.argv) > 3 else "dev"

    dataset = load_dataset("facebook/bouquet", "sentence_level", split=split, trust_remote_code=True)
    data = dataset.to_pandas()

    src2tgt = pd.merge(
        data.loc[data["src_lang"].eq(src_lang)].drop(["tgt_lang", "tgt_text"], axis=1),
        data.loc[data["src_lang"].eq(tgt_lang), ["src_lang", "src_text", "uniq_id"]].rename({"src_lang": "tgt_lang", "src_text": "tgt_text"}, axis=1),
        on="uniq_id",
    )

    for src, tgt in zip(src2tgt["src_text"], src2tgt["tgt_text"]):
        print(f"{src}\t{tgt}")

if __name__ == "__main__":
    main()
