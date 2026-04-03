
import os

import utils

import torch
import numpy as np
from sklearn.metrics import pairwise_distances

flores_lang_mapping = { # https://huggingface.co/spaces/UNESCO/nllb/blob/main/flores.py
    "Acehnese (Arabic script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta’izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar (Arabic script)": "bjn_Arab",
    "Banjar (Latin script)": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri (Arabic script)": "knc_Arab",
    "Central Kanuri (Latin script)": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Kasai": "lua_Latn",
    "Ganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Mizo": "lus_Latn",
    "Standard Latvian": "lvs_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau (Latin script)": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Plateau Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Meitei (Bengali script)": "mni_Beng",
    "Halh Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Maori": "mri_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "West Central Oromo": "gaz_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Eastern Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Western Persian": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Southern Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Beng",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Southern Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Tosk Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tamasheq (Latin script)": "taq_Latn",
    "Tamasheq (Tifinagh script)": "taq_Tfng",
    "Tok Pisin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Uyghur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Northern Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Eastern Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Standard Malay": "zsm_Latn",
    "Zulu": "zul_Latn",
}

def get_embeddings(name, sentences, lang, filename=None, data_dir=None, suffix_name="data", max_seq_len=512, device=None, batch_size=8, model=None, return_model=False, numpy=True):
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
            assert lang in flores_lang_mapping, f"Language not supported for SONAR embeddings: {lang}"

            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            model_name_or_path = "text_sonar_basic_encoder"
            t2vec_model = TextToEmbeddingModelPipeline(encoder=model_name_or_path, tokenizer=model_name_or_path, device=device) if model is None else model
            model = t2vec_model
            _lang = flores_lang_mapping[lang]
            embeddings = t2vec_model.predict(sentences, source_lang=_lang, max_seq_len=max_seq_len, batch_size=batch_size) # https://github.com/facebookresearch/SONAR/blob/3a95f405d86e2d51ba23154c8a413df34949f1c3/sonar/inference_pipelines/text.py#L211
            embeddings = embeddings.detach().cpu()
        else:
            raise Exception(f"Embeddings not supported: {name}")

    assert embeddings is not None, "Embeddings could not be computed"

    if numpy:
        embeddings = embeddings.numpy()

        assert isinstance(embeddings, np.ndarray), f"Embeddings must be a numpy array, got {type(embeddings)}: {embeddings}"
    else:
        assert isinstance(embeddings, torch.Tensor), f"Embeddings must be a torch Tensor, got {type(embeddings)}: {embeddings}"

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

        assert np.all(similarity <= 1.0) and np.all(similarity >= -1.0), f"Cosine similarity must be between 0 and 2, got {similarity}"
    else:
        raise Exception(f"Metric not supported: {metric}")

    return similarity # greater is more similar
