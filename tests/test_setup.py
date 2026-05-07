from transformers import AutoConfig, AutoTokenizer

from configs.config import config


def test_tokenizers_share_vocab():
    tok_draft = AutoTokenizer.from_pretrained(config.draft_name)
    tok_target = AutoTokenizer.from_pretrained(config.target_name)

    assert tok_draft.get_vocab() == tok_target.get_vocab(), "vocab mismatch"
    assert tok_draft.bos_token_id == tok_target.bos_token_id
    assert tok_draft.eos_token_id == tok_target.eos_token_id


def test_configs_agree_on_vocab_size():
    cfg_draft = AutoConfig.from_pretrained(config.draft_name)
    cfg_target = AutoConfig.from_pretrained(config.target_name)

    assert cfg_draft.vocab_size == cfg_target.vocab_size

    print(
        f"\ndraft : {cfg_draft.num_hidden_layers}L "
        f"hidden={cfg_draft.hidden_size} kv_heads={cfg_draft.num_key_value_heads}"
    )
    print(
        f"target: {cfg_target.num_hidden_layers}L "
        f"hidden={cfg_target.hidden_size} kv_heads={cfg_target.num_key_value_heads}"
    )


def test_tokenizer_roundtrip():
    tok = AutoTokenizer.from_pretrained(config.target_name)
    text = "Speculative decoding accelerates LLM inference."
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids)
    assert decoded.strip() == text
