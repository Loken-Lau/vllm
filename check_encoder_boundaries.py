"""Compare the real encoder against its pinned original forward."""
import ast
import json
import pathlib
import subprocess
import tempfile
import textwrap

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel, cleanup_dist_env_and_memory
import vllm.model_executor.models.qwen3_omni_moe_thinker as module

root = pathlib.Path(__file__).resolve().parent
source = subprocess.check_output(['git', 'show', 'f3871c3cbf8a608f7ee833ec4c2c5277f77bee41:vllm/model_executor/models/qwen3_omni_moe_thinker.py'], text=True)
cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3OmniMoeAudioEncoder')
forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
namespace = dict(vars(module))
exec(compile(textwrap.dedent(ast.get_source_segment(source, forward)), 'pinned_baseline_forward', 'exec'), namespace)
baseline = namespace['forward']
torch.manual_seed(61608)
records = []
with tempfile.NamedTemporaryFile() as f, set_current_vllm_config(VllmConfig()):
    init_distributed_environment(world_size=1, rank=0, distributed_init_method=f'file://{f.name}', local_rank=0, backend='nccl')
    initialize_model_parallel(1, 1)
    config = Qwen3OmniMoeAudioEncoderConfig(num_mel_bins=8, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=64, d_model=32, max_source_positions=16, n_window=50, n_window_infer=800, output_dim=16, conv_chunksize=2, downsample_hidden_size=4)
    for dtype in [torch.float32, torch.bfloat16]:
        encoder = module.Qwen3OmniMoeAudioEncoder(config).to(device='cuda', dtype=dtype).eval()
        for parameter in encoder.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        for lengths in [[1], [7], [8], [9], [99], [100], [101], [200], [201], [1,100], [101,201], [100,200], [799,801]]:
            features = torch.randn(8, sum(lengths), device='cuda', dtype=dtype)
            feature_lens = torch.tensor(lengths, device='cuda')
            output_lens = torch.tensor([(l//100)*13+((l%100)+7)//8 for l in lengths], device='cuda')
            with torch.inference_mode():
                expected = baseline(encoder, features, feature_lens, output_lens)
                actual = encoder(features, feature_lens, output_lens)
            record = {'dtype': str(dtype), 'lengths': lengths, 'shape': list(actual.shape), 'equal': torch.equal(actual, expected), 'max_abs_diff': (actual-expected).abs().max().item()}
            records.append(record)
            assert actual.shape[0] == sum((l//100)*13+((l%100)+7)//8 for l in lengths)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cleanup_dist_env_and_memory()
(root/'encoder-boundaries.json').write_text(json.dumps(records, indent=2)+'\n')
print(json.dumps(records), flush=True)
