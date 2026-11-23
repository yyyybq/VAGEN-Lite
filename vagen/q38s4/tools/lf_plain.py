"""C line (q38 s4), smoke S4: the existing SFT trainer (vagen.q38.sft.lf_launcher, LLaMA-Factory's own trainer,
collator and model loading) with only the D4 frame sampling of lf_plugin.S4VideoPlugin, so it reads the same
``|s4|`` videos as vagen.q38s4.lf_nfp.  S4 runs it and lf_nfp with both NFP weights 0 on the same tokenized QA data
and compares the per-step losses (vagen.q38s4.tools.smoke compare).

    accelerate launch ... -m vagen.q38s4.tools.lf_plain RUN/train.yaml   (RUN/run_plain.sh, smoke plain)
"""
from __future__ import annotations


def install() -> None:
    from llamafactory.train.sft import workflow as W

    from vagen.q38s4.lf_nfp import make_get_template

    W.get_template_and_fix_tokenizer = make_get_template(W.get_template_and_fix_tokenizer)


def main() -> None:
    install()
    from vagen.q38.sft import lf_launcher

    lf_launcher.main()


if __name__ == "__main__":
    main()
