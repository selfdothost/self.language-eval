"""Model implementations for language_eval.

Models are lazily loaded via the registry system to improve startup performance.

Usage
-----
For programmatic access, use the registry:

    from language_eval.api.registry import get_model
    model_cls = get_model("hf")
    model = model_cls(pretrained="gpt2")

For direct imports (e.g., subclassing), use explicit module paths:

    from language_eval.models.huggingface import HFLM
    from language_eval.models.vllm_causallms import VLLM

Adding New Models
-----------------
1. Create your model class in a new file under language_eval/models/
2. Use the @register_model decorator on your class
3. Add an entry to MODEL_MAPPING below for lazy discovery
"""

MODEL_MAPPING = {
    "anthropic-chat": "language_eval.models.anthropic_llms:AnthropicChat",
    "anthropic-chat-completions": "language_eval.models.anthropic_llms:AnthropicChat",
    "anthropic-completions": "language_eval.models.anthropic_llms:AnthropicLM",
    "dummy": "language_eval.models.dummy:DummyLM",
    "ggml": "language_eval.models.gguf:GGUFLM",
    "gguf": "language_eval.models.gguf:GGUFLM",
    "hf": "language_eval.models.huggingface:HFLM",
    "hf-audiolm-qwen": "language_eval.models.hf_audiolm:HFAudioLM",
    "hf-auto": "language_eval.models.huggingface:HFLM",
    "hf-mistral3": "language_eval.models.mistral3:Mistral3LM",
    "hf-multimodal": "language_eval.models.hf_vlms:HFMultimodalLM",
    "huggingface": "language_eval.models.huggingface:HFLM",
    "ipex": "language_eval.models.optimum_ipex:IPEXForCausalLM",
    "local-chat-completions": "language_eval.models.openai_completions:LocalChatCompletion",
    "local-completions": "language_eval.models.openai_completions:LocalCompletionsAPI",
    "mamba_ssm": "language_eval.models.mamba_lm:MambaLMWrapper",
    "megatron_lm": "language_eval.models.megatron_lm:MegatronLMEval",
    "nemo_lm": "language_eval.models.nemo_lm:NeMoLM",
    "neuronx": "language_eval.models.neuron_optimum:NeuronModelForCausalLM",
    "openai-chat-completions": "language_eval.models.openai_completions:OpenAIChatCompletion",
    "openai-completions": "language_eval.models.openai_completions:OpenAICompletionsAPI",
    "openvino": "language_eval.models.optimum_lm:OptimumLM",
    "habana": "language_eval.models.optimum_habana:HabanaLM",
    "sglang": "language_eval.models.sglang_causallms:SGLangLM",
    "sglang-generate": "language_eval.models.sglang_generate_API:SGLANGGENERATEAPI",
    "steered": "language_eval.models.hf_steered:SteeredModel",
    "textsynth": "language_eval.models.textsynth:TextSynthLM",
    "vllm": "language_eval.models.vllm_causallms:VLLM",
    "vllm-vlm": "language_eval.models.vllm_vlms:VLLM_VLM",
    "watsonx_llm": "language_eval.models.ibm_watsonx_ai:WatsonxLLM",
    "winml": "language_eval.models.winml:WindowsML",
}


def _register_all_models():
    """Register all known models lazily in the registry."""
    from language_eval.api.registry import model_registry

    for name, path in MODEL_MAPPING.items():
        # Only register if not already present (avoids conflicts when modules are imported)
        if name not in model_registry:
            # Register the lazy placeholder
            model_registry.register(name, target=path)


# Call registration on module import
_register_all_models()

__all__ = ["MODEL_MAPPING"]