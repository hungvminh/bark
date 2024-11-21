# Detailed Explanation of the Bark Codebase

## `bark/api.py`

This file contains functions for generating semantic arrays from text, converting semantic tokens to waveforms, saving prompts, and generating audio.

### Functions

#### `text_to_semantic`

```python
def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic
```

This function generates a semantic array from the input text. It takes the following parameters:
- `text`: The text to be turned into audio.
- `history_prompt`: An optional history choice for audio cloning.
- `temp`: The generation temperature (1.0 more diverse, 0.0 more conservative).
- `silent`: A boolean to disable the progress bar.

The function returns a numpy semantic array to be fed into `semantic_to_waveform`.

#### `semantic_to_waveform`

```python
def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = .7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr
```

This function generates an audio array from the semantic input. It takes the following parameters:
- `semantic_tokens`: The semantic token output from `text_to_semantic`.
- `history_prompt`: An optional history choice for audio cloning.
- `temp`: The generation temperature (1.0 more diverse, 0.0 more conservative).
- `silent`: A boolean to disable the progress bar.
- `output_full`: A boolean to return the full generation to be used as a history prompt.

The function returns a numpy audio array at a sample frequency of 24khz.

#### `save_as_prompt`

```python
def save_as_prompt(filepath, full_generation):
    """Save the full generation as a prompt file.

    Args:
        filepath: path to save the prompt file
        full_generation: dictionary containing semantic, coarse, and fine prompts

    Returns:
        None
    """
    assert(filepath.endswith(".npz"))
    assert(isinstance(full_generation, dict))
    assert("semantic_prompt" in full_generation)
    assert("coarse_prompt" in full_generation)
    assert("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)
```

This function saves the full generation as a prompt file. It takes the following parameters:
- `filepath`: The path to save the prompt file.
- `full_generation`: A dictionary containing semantic, coarse, and fine prompts.

The function does not return any value.

#### `generate_audio`

```python
def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr
```

This function generates an audio array from the input text. It takes the following parameters:
- `text`: The text to be turned into audio.
- `history_prompt`: An optional history choice for audio cloning.
- `text_temp`: The generation temperature for text (1.0 more diverse, 0.0 more conservative).
- `waveform_temp`: The generation temperature for waveform (1.0 more diverse, 0.0 more conservative).
- `silent`: A boolean to disable the progress bar.
- `output_full`: A boolean to return the full generation to be used as a history prompt.

The function returns a numpy audio array at a sample frequency of 24khz.

## `bark/generation.py`

This file contains the main generation functionality, including model loading, tokenization, and audio code generation.

### Functions

#### `_grab_best_device`

```python
def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and use_gpu and GLOBAL_ENABLE_MPS:
        device = "mps"
    else:
        device = "cpu"
    return device
```

This function determines the best available device (GPU, MPS, or CPU) for model execution. It takes the following parameter:
- `use_gpu`: A boolean indicating whether to use GPU if available.

The function returns the best available device as a string.

#### `_get_ckpt_path`

```python
def _get_ckpt_path(model_type, use_small=False):
    key = model_type
    if use_small or USE_SMALL_MODELS:
        key += "_small"
    return os.path.join(CACHE_DIR, REMOTE_MODEL_PATHS[key]["file_name"])
```

This function constructs the checkpoint path for a given model type. It takes the following parameters:
- `model_type`: The type of the model (e.g., "text", "coarse", "fine").
- `use_small`: A boolean indicating whether to use the small model.

The function returns the checkpoint path as a string.

#### `_download`

```python
def _download(from_hf_path, file_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=CACHE_DIR)
```

This function downloads a model checkpoint from the Hugging Face Hub. It takes the following parameters:
- `from_hf_path`: The repository ID on the Hugging Face Hub.
- `file_name`: The name of the file to download.

The function does not return any value.

#### `InferenceContext`

```python
class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark
```

This class provides a context manager for setting and restoring the cuDNN benchmark mode. It takes the following parameter:
- `benchmark`: A boolean indicating whether to enable cuDNN benchmarking.

The class does not return any value.

#### `_inference_mode`

```python
@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield
```

This function provides a context manager for inference mode, which includes disabling gradient computation and enabling mixed precision. It does not take any parameters.

The function does not return any value.

#### `_clear_cuda_cache`

```python
def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

This function clears the CUDA cache and synchronizes the device. It does not take any parameters.

The function does not return any value.

#### `clean_models`

```python
def clean_models(model_key=None):
    """
    Clean the models from memory.

    Args:
        model_key (str, optional): The key of the model to clean. If None, clean all models.
    """
    global models
    model_keys = [model_key] if model_key is not None else list(models.keys())
    for k in model_keys:
        if k in models:
            del models[k]
    _clear_cuda_cache()
    gc.collect()
```

This function cleans the models from memory. It takes the following parameter:
- `model_key`: An optional key of the model to clean. If None, it cleans all models.

The function does not return any value.

#### `_load_model`

```python
def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    """
    Load a model from a checkpoint.

    Args:
        ckpt_path (str): The path to the checkpoint file.
        device (str): The device to load the model on.
        use_small (bool, optional): Whether to use the small model. Defaults to False.
        model_type (str, optional): The type of the model. Defaults to "text".

    Returns:
        model: The loaded model.
    """
    if model_type == "text":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "coarse":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "fine":
        ConfigClass = FineGPTConfig
        ModelClass = FineGPT
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    model_info = REMOTE_MODEL_PATHS[model_key]
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        _download(model_info["repo_id"], model_info["file_name"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    gptconf = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(gptconf)
    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    model.eval()
    model.to(device)
    del checkpoint, state_dict
    _clear_cuda_cache()
    if model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        return {
            "model": model,
            "tokenizer": tokenizer,
        }
    return model
```

This function loads a model from a checkpoint. It takes the following parameters:
- `ckpt_path`: The path to the checkpoint file.
- `device`: The device to load the model on.
- `use_small`: An optional boolean indicating whether to use the small model. Defaults to False.
- `model_type`: An optional string indicating the type of the model. Defaults to "text".

The function returns the loaded model.

#### `_load_codec_model`

```python
def _load_codec_model(device):
    """
    Load the codec model.

    Args:
        device (str): The device to load the model on.

    Returns:
        model: The loaded codec model.
    """
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    _clear_cuda_cache()
    return model
```

This function loads the codec model. It takes the following parameter:
- `device`: The device to load the model on.

The function returns the loaded codec model.

#### `load_model`

```python
def load_model(use_gpu=True, use_small=False, force_reload=False, model_type="text"):
    """
    Load a model.

    Args:
        use_gpu (bool, optional): Whether to use GPU. Defaults to True.
        use_small (bool, optional): Whether to use the small model. Defaults to False.
        force_reload (bool, optional): Whether to force reload the model. Defaults to False.
        model_type (str, optional): The type of the model. Defaults to "text".

    Returns:
        model: The loaded model.
    """
    _load_model_f = funcy.partial(_load_model, model_type=model_type, use_small=use_small)
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    model_key = f"{model_type}"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        ckpt_path = _get_ckpt_path(model_type, use_small=use_small)
        clean_models(model_key=model_key)
        model = _load_model_f(ckpt_path, device)
        models[model_key] = model
    if model_type == "text":
        models[model_key]["model"].to(device)
    else:
        models[model_key].to(device)
    return models[model_key]
```

This function loads a model. It takes the following parameters:
- `use_gpu`: An optional boolean indicating whether to use GPU. Defaults to True.
- `use_small`: An optional boolean indicating whether to use the small model. Defaults to False.
- `force_reload`: An optional boolean indicating whether to force reload the model. Defaults to False.
- `model_type`: An optional string indicating the type of the model. Defaults to "text".

The function returns the loaded model.

#### `load_codec_model`

```python
def load_codec_model(use_gpu=True, force_reload=False):
    """
    Load the codec model.

    Args:
        use_gpu (bool, optional): Whether to use GPU. Defaults to True.
        force_reload (bool, optional): Whether to force reload the model. Defaults to False.

    Returns:
        model: The loaded codec model.
    """
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    if device == "mps":
        # encodec doesn't support mps
        device = "cpu"
    model_key = "codec"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)
        model = _load_codec_model(device)
        models[model_key] = model
    models[model_key].to(device)
    return models[model_key]
```

This function loads the codec model. It takes the following parameters:
- `use_gpu`: An optional boolean indicating whether to use GPU. Defaults to True.
- `force_reload`: An optional boolean indicating whether to force reload the model. Defaults to False.

The function returns the loaded codec model.

#### `preload_models`

```python
def preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
):
    """
    Preload all the necessary models for the pipeline.

    Args:
        text_use_gpu (bool, optional): Whether to use GPU for the text model. Defaults to True.
        text_use_small (bool, optional): Whether to use the small text model. Defaults to False.
        coarse_use_gpu (bool, optional): Whether to use GPU for the coarse model. Defaults to True.
        coarse_use_small (bool, optional): Whether to use the small coarse model. Defaults to False.
        fine_use_gpu (bool, optional): Whether to use GPU for the fine model. Defaults to True.
        fine_use_small (bool, optional): Whether to use the small fine model. Defaults to False.
        codec_use_gpu (bool, optional): Whether to use GPU for the codec model. Defaults to True.
        force_reload (bool, optional): Whether to force reload the models. Defaults to False.
    """
    if _grab_best_device() == "cpu" and (
        text_use_gpu or coarse_use_gpu or fine_use_gpu or codec_use_gpu
    ):
        logger.warning("No GPU being used. Careful, inference might be very slow!")
    _ = load_model(
        model_type="text", use_gpu=text_use_gpu, use_small=text_use_small, force_reload=force_reload
    )
    _ = load_model(
        model_type="coarse",
        use_gpu=coarse_use_gpu,
        use_small=coarse_use_small,
        force_reload=force_reload,
    )
    _ = load_model(
        model_type="fine", use_gpu=fine_use_gpu, use_small=fine_use_small, force_reload=force_reload
    )
    _ = load_codec_model(use_gpu=codec_use_gpu, force_reload=force_reload)
```

This function preloads all the necessary models for the pipeline. It takes the following parameters:
- `text_use_gpu`: An optional boolean indicating whether to use GPU for the text model. Defaults to True.
- `text_use_small`: An optional boolean indicating whether to use the small text model. Defaults to False.
- `coarse_use_gpu`: An optional boolean indicating whether to use GPU for the coarse model. Defaults to True.
- `coarse_use_small`: An optional boolean indicating whether to use the small coarse model. Defaults to False.
- `fine_use_gpu`: An optional boolean indicating whether to use GPU for the fine model. Defaults to True.
- `fine_use_small`: An optional boolean indicating whether to use the small fine model. Defaults to False.
- `codec_use_gpu`: An optional boolean indicating whether to use GPU for the codec model. Defaults to True.
- `force_reload`: An optional boolean indicating whether to force reload the models. Defaults to False.

The function does not return any value.

#### `_tokenize`

```python
def _tokenize(tokenizer, text):
    """
    Tokenize the input text.

    Args:
        tokenizer (BertTokenizer): The tokenizer to use.
        text (str): The input text.

    Returns:
        list: The tokenized text.
    """
    return tokenizer.encode(text, add_special_tokens=False)
```

This function tokenizes the input text. It takes the following parameters:
- `tokenizer`: The tokenizer to use.
- `text`: The input text.

The function returns the tokenized text as a list.

#### `_detokenize`

```python
def _detokenize(tokenizer, enc_text):
    """
    Detokenize the encoded text.

    Args:
        tokenizer (BertTokenizer): The tokenizer to use.
        enc_text (list): The encoded text.

    Returns:
        str: The detokenized text.
    """
    return tokenizer.decode(enc_text)
```

This function detokenizes the encoded text. It takes the following parameters:
- `tokenizer`: The tokenizer to use.
- `enc_text`: The encoded text.

The function returns the detokenized text as a string.

#### `_normalize_whitespace`

```python
def _normalize_whitespace(text):
    """
    Normalize whitespace in the input text.

    Args:
        text (str): The input text.

    Returns:
        str: The normalized text.
    """
    return re.sub(r"\s+", " ", text).strip()
```

This function normalizes whitespace in the input text. It takes the following parameter:
- `text`: The input text.

The function returns the normalized text as a string.

#### `_load_history_prompt`

```python
def _load_history_prompt(history_prompt_input):
    """
    Load the history prompt.

    Args:
        history_prompt_input (str or dict): The history prompt input.

    Returns:
        dict: The loaded history prompt.
    """
    if isinstance(history_prompt_input, str) and history_prompt_input.endswith(".npz"):
        history_prompt = np.load(history_prompt_input)
    elif isinstance(history_prompt_input, str):
        # make sure this works on non-ubuntu
        history_prompt_input = os.path.join(*history_prompt_input.split("/"))
        if history_prompt_input not in ALLOWED_PROMPTS:
            raise ValueError("history prompt not found")
        history_prompt = np.load(
            os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt_input}.npz")
        )
    elif isinstance(history_prompt_input, dict):
        assert("semantic_prompt" in history_prompt_input)
        assert("coarse_prompt" in history_prompt_input)
        assert("fine_prompt" in history_prompt_input)
        history_prompt = history_prompt_input
    else:
        raise ValueError("history prompt format unrecognized")
    return history_prompt
```

This function loads the history prompt. It takes the following parameter:
- `history_prompt_input`: The history prompt input, which can be a string or a dictionary.

The function returns the loaded history prompt as a dictionary.

#### `generate_text_semantic`

```python
def generate_text_semantic(
    text,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    min_eos_p=0.2,
    max_gen_duration_s=None,
    allow_early_stop=True,
    use_kv_caching=False,
):
    """
    Generate semantic tokens from text.

    Args:
        text (str): The input text.
        history_prompt (str or dict, optional): The history prompt. Defaults to None.
        temp (float, optional): The temperature for sampling. Defaults to 0.7.
        top_k (int, optional): The top-k sampling parameter. Defaults to None.
        top_p (float, optional): The top-p sampling parameter. Defaults to None.
        silent (bool, optional): Whether to disable progress bar. Defaults to False.
        min_eos_p (float, optional): The minimum probability for early stopping. Defaults to 0.2.
        max_gen_duration_s (float, optional): The maximum generation duration in seconds. Defaults to None.
        allow_early_stop (bool, optional): Whether to allow early stopping. Defaults to True.
        use_kv_caching (bool, optional): Whether to use key-value caching. Defaults to False.

    Returns:
        np.ndarray: The generated semantic tokens.
    """
    assert isinstance(text, str)
    text = _normalize_whitespace(text)
    assert len(text.strip()) > 0
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        semantic_history = history_prompt["semantic_prompt"]
        assert (
            isinstance(semantic_history, np.ndarray)
            and len(semantic_history.shape) == 1
            and len(semantic_history) > 0
            and semantic_history.min() >= 0
            and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
        )
    else:
        semantic_history = None
    # load models if not yet exist
    global models
    global models_devices
    if "text" not in models:
        preload_models()
    model_container = models["text"]
    model = model_container["model"]
    tokenizer = model_container["tokenizer"]
    encoded_text = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if OFFLOAD_CPU:
        model.to(models_devices["text"])
    device = next(model.parameters()).device
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = torch.from_numpy(
        np.hstack([
            encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
        ]).astype(np.int64)
    )[None]
    assert x.shape[1] == 256 + 256 + 1
    with _inference_mode():
        x = x.to(device)
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )
            if top_p is not None:
                # faster to convert to numpy
                original_device = relevant_logits.device
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(original_device)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = F.softmax(relevant_logits / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
            if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(n - pbar_state)
                break
            x = torch.cat((x, item_next[None]), dim=1)
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                pbar.update(n - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.update(n - pbar_state)
                break
            del logits, relevant_logits, probs, item_next

            if n > pbar_state:
                if n > pbar.total:
                    pbar.total = n
                pbar.update(n - pbar_state)
            pbar_state = n
        pbar.total = n
        pbar.refresh()
        pbar.close()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]
    if OFFLOAD_CPU:
        model.to("cpu")
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    _clear_cuda_cache()
    return out
```

This function generates semantic tokens from text. It takes the following parameters:
- `text`: The input text.
- `history_prompt`: An optional history prompt. Defaults to None.
- `temp`: An optional temperature for sampling. Defaults to 0.7.
- `top_k`: An optional top-k sampling parameter. Defaults to None.
- `top_p`: An optional top-p sampling parameter. Defaults to None.
- `silent`: An optional boolean to disable the progress bar. Defaults to False.
- `min_eos_p`: An optional minimum probability for early stopping. Defaults to 0.2.
- `max_gen_duration_s`: An optional maximum generation duration in seconds. Defaults to None.
- `allow_early_stop`: An optional boolean to allow early stopping. Defaults to True.
- `use_kv_caching`: An optional boolean to use key-value caching. Defaults to False.

The function returns the generated semantic tokens as a numpy array.

#### `_flatten_codebooks`

```python
def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    """
    Flatten the codebooks.

    Args:
        arr (np.ndarray): The input array.
        offset_size (int, optional): The offset size. Defaults to CODEBOOK_SIZE.

    Returns:
        np.ndarray: The flattened codebooks.
    """
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr
```

This function flattens the codebooks. It takes the following parameters:
- `arr`: The input array.
- `offset_size`: An optional offset size. Defaults to CODEBOOK_SIZE.

The function returns the flattened codebooks as a numpy array.

#### `generate_coarse`

```python
def generate_coarse(
    x_semantic,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    max_coarse_history=630,  # min 60 (faster), max 630 (more context)
    sliding_window_len=60,
    use_kv_caching=False,
):
    """
    Generate coarse audio codes from semantic tokens.

    Args:
        x_semantic (np.ndarray): The input semantic tokens.
        history_prompt (str or dict, optional): The history prompt. Defaults to None.
        temp (float, optional): The temperature for sampling. Defaults to 0.7.
        top_k (int, optional): The top-k sampling parameter. Defaults to None.
        top_p (float, optional): The top-p sampling parameter. Defaults to None.
        silent (bool, optional): Whether to disable progress bar. Defaults to False.
        max_coarse_history (int, optional): The maximum coarse history length. Defaults to 630.
        sliding_window_len (int, optional): The sliding window length. Defaults to 60.
        use_kv_caching (bool, optional): Whether to use key-value caching. Defaults to False.

    Returns:
        np.ndarray: The generated coarse audio codes.
    """
    assert (
        isinstance(x_semantic, np.ndarray)
        and len(x_semantic.shape) == 1
        and len(x_semantic) > 0
        and x_semantic.min() >= 0
        and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_semantic_history = history_prompt["semantic_prompt"]
        x_coarse_history = history_prompt["coarse_prompt"]
        assert (
            isinstance(x_semantic_history, np.ndarray)
            and len(x_semantic_history.shape) == 1
            and len(x_semantic_history) > 0
            and x_semantic_history.min() >= 0
            and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            and isinstance(x_coarse_history, np.ndarray)
            and len(x_coarse_history.shape) == 2
            and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
            and x_coarse_history.shape[-1] >= 0
            and x_coarse_history.min() >= 0
            and x_coarse_history.max() <= CODEBOOK_SIZE - 1
            and (
                round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
            )
        )
        x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)
    # load models if not yet exist
    global models
    global models_devices
    if "coarse" not in models:
        preload_models()
    model = models["coarse"]
    if OFFLOAD_CPU:
        model.to(models_devices["coarse"])
    device = next(model.parameters()).device
    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    with _inference_mode():
        x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
        x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :256]
            x_in = F.pad(
                x_in,
                (0, 256 - x_in.shape[-1]),
                "constant",
                COARSE_SEMANTIC_PAD_TOKEN,
            )
            x_in = torch.hstack(
                [
                    x_in,
                    torch.tensor([COARSE_INFER_TOKEN])[None].to(device),
                    x_coarse_in[:, -max_coarse_history:],
                ]
            )
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [-1]]
                else:
                    x_input = x_in

                logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                logit_start_idx = (
                    SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = (
                    SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                )
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                if top_p is not None:
                    # faster to convert to numpy
                    original_device = relevant_logits.device
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(original_device)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temp, dim=-1)
                item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
                item_next += logit_start_idx
                x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                x_in = torch.cat((x_in, item_next[None]), dim=1)
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history) :]
    del x_coarse_in
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
    _clear_cuda_cache()
    return gen_coarse_audio_arr
```

This function generates coarse audio codes from semantic tokens. It takes the following parameters:
- `x_semantic`: The input semantic tokens.
- `history_prompt`: An optional history prompt. Defaults to None.
- `temp`: An optional temperature for sampling. Defaults to 0.7.
- `top_k`: An optional top-k sampling parameter. Defaults to None.
- `top_p`: An optional top-p sampling parameter. Defaults to None.
