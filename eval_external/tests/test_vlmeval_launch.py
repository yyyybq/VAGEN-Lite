"""The SDPA override changes only the attention argument, and only while it is active."""

from eval_external.qa_transfer import vlmeval_launch as L


class _Base:
    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls, path, kw


class _Child(_Base):
    pass


def test_attention_override_is_scoped(monkeypatch):
    monkeypatch.setattr(L, "MODEL_CLASSES", (_Child,))
    with L.attention_override("sdpa"):
        cls, path, kw = _Child.from_pretrained("p", torch_dtype="auto", attn_implementation="flash_attention_2")
        assert (cls, path, kw) == (_Child, "p", {"torch_dtype": "auto", "attn_implementation": "sdpa"})
        assert _Child.from_pretrained("p", attn_implementation="eager")[2]["attn_implementation"] == "eager"
    assert "from_pretrained" not in _Child.__dict__
    assert _Child.from_pretrained("p", attn_implementation="flash_attention_2")[2]["attn_implementation"] == \
        "flash_attention_2"
    assert L.vlmeval.vlm.Qwen2VLChatSDPA is L.Qwen2VLChatSDPA
