import json
import os
from typing import Optional

from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "aquila": AquilaConfig,
    "baichuan": BaiChuanConfig,
    "chatglm": ChatGLMConfig,
    "cpm_dragonfly": CPMDragonflyConfig,
    "cpm_mistral": CPMMistralConfig,
    "mpt": MPTConfig,
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "yi": YiConfig,
}


def get_config(model: str,
               trust_remote_code: bool,
               revision: Optional[str] = None) -> PretrainedConfig:
    try:
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision)
        except ValueError as e:
            if (not trust_remote_code and
                    "requires you to execute the configuration file" in str(e)):
                err_msg = (
                    "Failed to load the model config. If the model is a custom "
                    "model not yet available in the HuggingFace transformers "
                    "library, consider setting `trust_remote_code=True` in LLM "
                    "or using the `--trust-remote-code` flag in the CLI.")
                raise RuntimeError(err_msg) from e
            else:
                raise e
    #except KeyError as e:
    except:
        if os.path.exists(model):
            config = {}
            with open(f"{model}/config.json", 'r') as fin:
                config = json.load(fin)
            model_type = config.get("model_type", "")
            if model_type in _CONFIG_REGISTRY:
                config_class = _CONFIG_REGISTRY[model_type]
                config = config_class.from_pretrained(model, revision=revision)
        else:
            raise e
    return config
