"""Independently recompute word errors and verify checkpoint integrity."""
import hashlib
import json
import pathlib

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

p = pathlib.Path(__file__).resolve().parent
root = p.parents[1]
x = json.loads((p / 'dataset-evaluation.json').read_text())
normalizer_file = root / '.contributions/infra-screen/whisper-normalizer.json'
n = EnglishTextNormalizer(json.loads(normalizer_file.read_text()))


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        row = [i]
        for j, target in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j - 1] + (word != target)))
        previous = row
    return previous[-1]


references = [n(s['reference']).split() for s in x['samples']]
words = sum(map(len, references))
result = {'reference_words': words, 'cases': {}, 'checkpoint': {}}
for batch, case in x['cases'].items():
    out = {}
    for variant in ['baseline', 'fixed']:
        v = case[variant]
        hypotheses = [n(r['text'].split('<asr_text>', 1)[-1]).split()
                      for r in v['outputs']]
        assert all(h == r['normalized'].split()
                   for h, r in zip(hypotheses, v['outputs'], strict=True))
        edits = sum(distance(a, b)
                    for a, b in zip(references, hypotheses, strict=True))
        assert edits / words == v['wer']
        assert edits == v['substitutions'] + v['deletions'] + v['insertions']
        out[variant] = {'edits': edits, 'words': words, 'wer': edits / words}
    out['normalized_pairs_equal'] = sum(
        a['normalized'] == b['normalized']
        for a, b in zip(case['baseline']['outputs'], case['fixed']['outputs'], strict=True))
    result['cases'][batch] = out
metadata = json.loads((p / 'asr-model-metadata.json').read_text())
for item in metadata['siblings']:
    if item['rfilename'].endswith('.safetensors'):
        path = root / '.models/Qwen3-ASR-0.6B' / item['rfilename']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == item['lfs']['sha256']
        result['checkpoint'][item['rfilename']] = digest
(p / 'independent-wer-audit.json').write_text(json.dumps(result, indent=2) + chr(10))
print(json.dumps(result, indent=2))
