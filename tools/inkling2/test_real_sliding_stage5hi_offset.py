"""Stage-5HI checkpoint query offset: official HF conversion adds one to source index."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from run_real_sliding_stage5hi import mandatory_sliding_sources, readiness_from_spans


class TestQueryOffset(unittest.TestCase):
    def test_sliding_source_indices(self):
        config = json.loads((HERE.parents[1] / 'tests/models/inkling2/fixtures/config.json').read_text())
        for physical in range(5):
            needed = mandatory_sliding_sources(physical, config)
            found = [n for n in needed if 'attn_res.attn_queries.' in n]
            expected = [] if physical == 0 else [f'model.llm.attn_res.attn_queries.{physical - 1}']
            self.assertEqual(found, expected)
            if expected:
                self.assertEqual(needed[expected[0]], ((2304,),))

    def test_full_layer_5_three_queries_is_not_loaded_as_sliding_4(self):
        config = json.loads((HERE.parents[1] / 'tests/models/inkling2/fixtures/config.json').read_text())
        t = config['text_config']; h = t['hidden_size']
        expected = {'model.llm.embed.weight': ((t['vocab_size'],h),),
                    'model.llm.embed_norm.weight': ((h,),)}
        for layer in range(5):
            expected.update(mandatory_sliding_sources(layer, config))
        # This is a checkpoint where layer 5's three attention queries occupy
        # source index 4.  Stage 5HI must neither load nor validate it for layer 4.
        spans = {k: SimpleNamespace(dtype='F32', shape=options[0]) for k, options in expected.items()}
        spans['model.llm.attn_res.attn_queries.4'] = SimpleNamespace(dtype='F32', shape=(3,h))
        report = readiness_from_spans(spans, config, layers=5)
        self.assertEqual(report['layers_supported'], 5)
        self.assertEqual(report['first_visit_cache_slots'], [0,1,2,3,4])
        self.assertNotIn('model.llm.attn_res.attn_queries.4', expected)

    def test_missing_correct_query_fails_closed(self):
        config = json.loads((HERE.parents[1] / 'tests/models/inkling2/fixtures/config.json').read_text())
        t=config['text_config'];h=t['hidden_size']
        expected = {'model.llm.embed.weight': ((t['vocab_size'],h),),
                    'model.llm.embed_norm.weight': ((h,),)}
        for layer in range(5): expected.update(mandatory_sliding_sources(layer, config))
        spans = {k: SimpleNamespace(dtype='F32', shape=opts[0]) for k, opts in expected.items()}
        del spans['model.llm.attn_res.attn_queries.3']
        spans['model.llm.attn_res.attn_queries.4'] = SimpleNamespace(dtype='F32', shape=(3,h))
        with self.assertRaisesRegex(ValueError, 'attn_queries.3'):
            readiness_from_spans(spans, config, layers=5)


if __name__ == '__main__':
    unittest.main()
