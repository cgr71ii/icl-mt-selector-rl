
import os

import utils

import torch
import numpy as np
from sklearn.metrics import pairwise_distances

def get_embeddings(name, sentences, lang, filename=None, data_dir=None, suffix_name="data", max_seq_len=512, device=None, batch_size=8, model=None, return_model=False):
    # Code adapted from https://github.com/ArmelRandy/ICL-MT/blob/fbef2aeec4f04e2dd63f2f726f946c143874bcf4/miscellaneous/embedding.py
    # max_seq_len does make sense for computing the embeddings: https://github.com/facebookresearch/SONAR/blob/3a95f405d86e2d51ba23154c8a413df34949f1c3/sonar/inference_pipelines/text.py#L277

    assert isinstance(sentences, list), f"sentences must be a list, got {type(sentences)}: {sentences}"
    assert len(sentences) > 0, "sentences must contain at least one sentence"

    if device is None:
        if utils.use_cuda():
            device = "cuda"
        else:
            device = "cpu"

    if isinstance(device, str):
        device = torch.device(device)

    assert isinstance(device, torch.device), type(device)

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
            t2vec_model = TextToEmbeddingModelPipeline(encoder=model_name_or_path, tokenizer=model_name_or_path, device=device) if model is None else model
            model = t2vec_model
            embeddings = t2vec_model.predict(sentences, source_lang=lang, max_seq_len=max_seq_len, batch_size=batch_size) # https://github.com/facebookresearch/SONAR/blob/3a95f405d86e2d51ba23154c8a413df34949f1c3/sonar/inference_pipelines/text.py#L211
            embeddings = embeddings.detach().cpu().numpy()
        else:
            raise Exception(f"Embeddings not supported: {name}")

    assert embeddings is not None, "Embeddings could not be computed"
    assert isinstance(embeddings, np.ndarray), f"Embeddings must be a numpy array, got {type(embeddings)}: {embeddings}"
    assert len(embeddings.shape) == 2, f"Embeddings must be a 2D numpy array, got {len(embeddings.shape)}D: {embeddings.shape}"
    assert embeddings.shape[0] == len(sentences), f"Embeddings first dimension must be equal to the number of sentences: {embeddings.shape[0]} vs {len(sentences)}"

    if handle_files and not embeddings_exist:
        embeddings.tofile(final_path)

    if return_model:
        return embeddings, model

    return embeddings

def get_similarity(embeddings1, embeddings2, metric="cosine"):
    assert len(embeddings1.shape) == 2, f"embeddings1 must be 2D, got {len(embeddings1.shape)}D: {embeddings1.shape}"
    assert len(embeddings2.shape) == 2, f"embeddings2 must be 2D, got {len(embeddings2.shape)}D: {embeddings2.shape}"
    assert embeddings1.shape[1] == embeddings2.shape[1], f"embeddings1 and embeddings2 must have the same last dimension size, got {embeddings1.shape[1]} vs {embeddings2.shape[1]}"

    distance = pairwise_distances(embeddings1, embeddings2, metric=metric) # cosine_distance if metric=="cosine"

    if metric == "cosine":
        similarity = 1.0 - distance # 1 - cosine_distance = cosine_similarity

        assert np.all(similarity <= 2.0) and np.all(similarity >= 0.0), f"Cosine similarity must be between 0 and 2, got {similarity}"
    else:
        raise Exception(f"Metric not supported: {metric}")

    return similarity # greater is more similar
