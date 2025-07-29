
import os
import sys
import math

from comet import download_model, load_from_checkpoint

def eval(model, source, translation, reference, batch_size=8, gpus=1, zero_score_empty=False):
    source = [source] if isinstance(source, str) else source
    translation = [translation] if isinstance(translation, str) else translation
    reference = [reference] if isinstance(reference, str) else reference

    assert isinstance(source, list), f"{type(source)}: {source}"
    assert isinstance(translation, list), f"{type(translation)}: {translation}"
    assert isinstance(reference, list), f"{type(reference)}: {reference}"

    data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(source, translation, reference)]
    score_zero_idxs = {idx for idx, (s, t, r) in enumerate(zip(source, translation, reference)) if s.strip() != '' and t.strip() == '' and r.strip() != ''} if zero_score_empty else {}

    assert len(data) == len(source) == len(translation) == len(reference), f"All input lists must have the same length: {len(data)}, {len(source)}, {len(translation)}, {len(reference)}"

    devices = [int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(',')] if "CUDA_VISIBLE_DEVICES" in os.environ and gpus > 0 else None
    scores = model.predict(data, batch_size=batch_size, gpus=gpus, accelerator="cpu" if gpus == 0 else "auto", devices=devices)

    assert len(scores["scores"]) == len(data), f"Scores length must match input data length: {len(scores['scores'])} vs {len(data)}: {scores['scores']} vs {data}"

    avg = sum(scores["scores"]) / len(scores["scores"]) if len(scores["scores"]) > 0 else 0.0

    assert math.isclose(avg, scores["system_score"]), f"Average score {avg} does not match system score {scores['system_score']}"

    # Remove zero scores
    scores = [s for i, s in enumerate(scores) if i not in score_zero_idxs] if len(score_zero_idxs) > 0 else scores["scores"]
    avg = (sum(scores) / len(scores) if len(scores) > 0 else 0.0) if len(score_zero_idxs) > 0 else avg

    return avg, scores

def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print_all_scores = bool(sys.argv[2]) if len(sys.argv) > 2 else False

    # Load COMET 22 DA
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    sources, translations, references = [], [], []

    for idx, row in enumerate(sys.stdin, 1):
        row = row.rstrip("\r\n").split("\t")

        assert len(row) == 3, f"Input must contain 3 columns: source, translation, and reference: {len(row)} columns found in line {idx}: {row}"

        sources.append(row[0])
        translations.append(row[1])
        references.append(row[2])

    avg, scores = eval(model, source, translation, reference, batch_size=8, gpus=1)

    if print_all_scores:
        for score in scores:
            print(score)

    #print(f"System score: {scores['system_score']}")
    print(f"System score: {avg}") # we use avg instead of system_score for taking into account the zero scores

if __name__ == "__main__":
    main()
