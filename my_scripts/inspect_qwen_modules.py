from pathlib import Path
import re

p = Path("my_hparams/ROME_qwen2.5_0.5b.yaml")
s = p.read_text(encoding="utf-8")

s = re.sub(
    r'layer_module_tmp:\s*["\']?.*?["\']?\s*$',
    'layer_module_tmp: "model.layers.{}.mlp"',
    s,
    flags=re.MULTILINE,
)

s = re.sub(
    r'attn_module_tmp:\s*["\']?.*?["\']?\s*$',
    'attn_module_tmp: "model.layers.{}.self_attn"',
    s,
    flags=re.MULTILINE,
)

s = re.sub(
    r'mlp_module_tmp:\s*["\']?.*?["\']?\s*$',
    'mlp_module_tmp: "model.layers.{}.mlp"',
    s,
    flags=re.MULTILINE,
)

s = re.sub(
    r'rewrite_module_tmp:\s*["\']?.*?["\']?\s*$',
    'rewrite_module_tmp: "model.layers.{}.mlp.down_proj"',
    s,
    flags=re.MULTILINE,
)

s = re.sub(
    r'ln_f_module:\s*["\']?.*?["\']?\s*$',
    'ln_f_module: "model.norm"',
    s,
    flags=re.MULTILINE,
)

s = re.sub(
    r'lm_head_module:\s*["\']?.*?["\']?\s*$',
    'lm_head_module: "lm_head"',
    s,
    flags=re.MULTILINE,
)

p.write_text(s, encoding="utf-8")
print("patched ROME yaml")