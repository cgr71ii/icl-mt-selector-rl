
import os

import numpy as np

def get_embeddings(name, sentences, lang, filename=None, data_dir=None, suffix_name="data"):
    # Code adapted from https://github.com/ArmelRandy/ICL-MT/blob/fbef2aeec4f04e2dd63f2f726f946c143874bcf4/miscellaneous/embedding.py

    assert isinstance(sentences, list), f"sentences must be a list, got {type(sentences)}: {sentences}"
    assert len(sentences) > 0, "sentences must contain at least one sentence"

    handle_files = data_dir is not None and filename is not None
    embeddings = None
    embeddings_exist = False
    final_path = None

    if handle_files:
        output_path = os.path.join(data_dir, filename)

        os.makedirs(os.path.join(output_path, name), exist_ok=True)

        final_path = os.path.join(output_path, f"{name}/{suffix_name}.bin")

        if os.path.exists(final_path):
            print(f"{final_path} already exist!")

            embeddings_exist = True
            embeddings = np.fromfile(final_path, dtype=np.float32, count=-1).reshape(len(sentences), -1)

    if embeddings is None:
        if name == "SONAR":
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            model_name_or_path = "text_sonar_basic_encoder"
            t2vec_model = TextToEmbeddingModelPipeline(
                encoder=model_name_or_path, tokenizer=model_name_or_path
            )
            embeddings = t2vec_model.predict(sentences, source_lang=lang)
            embeddings = embeddings.detach().numpy()

        else:
            raise Exception(f"Embeddings not supported: {name}")

    assert embeddings is not None, "Embeddings could not be computed"

    if handle_files and not embeddings_exist:
        embeddings.tofile(final_path)

    return embeddings
