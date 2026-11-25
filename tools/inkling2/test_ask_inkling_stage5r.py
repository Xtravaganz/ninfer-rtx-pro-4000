"""Offline contract tests for experimental text-prompt wrapper (no GPU required)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ask_inkling_stage5r import tokenize_question, render_tokens, stop_ids

class DummyTokenizer:
    eos_token_id=10
    def __init__(self):
        self.calls=[]
    def encode(self, text, add_special_tokens=False):
        self.calls.append(('raw',text,add_special_tokens))
        return [1000,1001]
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(('chat',messages,tokenize,add_generation_prompt))
        return [200001,1000,200002]
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        self.calls.append(('decode', ids, skip_special_tokens,clean_up_tokenization_spaces))
        return 'OK'

class TestAskStage5R(unittest.TestCase):
    def test_chat_template_is_used_without_silent_raw_fallback(self):
        tok=DummyTokenizer()
        self.assertEqual(tokenize_question(tok,'Was ist 2+2?'),[200001,1000,200002])
        self.assertEqual(tok.calls, [('chat',[{'role':'user','content':'Was ist 2+2?'}],True,True)])

    def test_explicit_raw_mode(self):
        tok=DummyTokenizer()
        self.assertEqual(tokenize_question(tok, 'Hallo', raw=True), [1000,1001])
        self.assertEqual(tok.calls, [('raw','Hallo',False)])

    def test_empty_question_rejected(self):
        with self.assertRaisesRegex(ValueError, 'question'):
            tokenize_question(DummyTokenizer(),'  ')

    def test_invalid_special_ids_rejected(self):
        tok=DummyTokenizer()
        tok.apply_chat_template=lambda *a,**kw:[-1]
        with self.assertRaisesRegex(ValueError, 'invalid'):
            tokenize_question(tok,'x')
        tok.apply_chat_template=lambda *a,**kw:[201024]
        with self.assertRaises(ValueError): tokenize_question(tok,'x')
        tok.apply_chat_template=lambda *a,**kw:[True]
        with self.assertRaises(ValueError): tokenize_question(tok,'x')
        tok.apply_chat_template=lambda *a,**kw:[]
        with self.assertRaises(ValueError): tokenize_question(tok,'x')

    def test_stop_ids_merge_tokenizer_and_text_config(self):
        tok=DummyTokenizer()
        self.assertEqual(stop_ids(tok,{'eos_token_id':[200006,200007]}),{10,200006,200007})
        self.assertEqual(stop_ids(tok,{}),{10})

    def test_decode_generated_tokens_not_prompt(self):
        tok=DummyTokenizer()
        self.assertEqual(render_tokens(tok,[4,5,6]),'OK')
        self.assertEqual(tok.calls[-1],('decode',[4,5,6],True,False))

    def test_invalid_input_contract(self):
        tok=DummyTokenizer()
        tok.apply_chat_template=lambda *a,**kw:'unexpected string'
        with self.assertRaises(ValueError): tokenize_question(tok,'x')

    def test_help_runs_without_torch_or_transformers(self):
        process=subprocess.run([sys.executable, str(HERE/'ask_inkling_stage5r.py'),'--help'],
                               capture_output=True,text=True,check=False)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertIn('--question',process.stdout)
        self.assertIn('--tokenize-only',process.stdout)


    def test_nested_batch_list_normalizes(self):
        tok=DummyTokenizer()
        tok.apply_chat_template=lambda *args, **kwargs:[[200001,1000,200002]]
        self.assertEqual(tokenize_question(tok,'x'),[200001,1000,200002])

    def test_mapping_input_ids_normalizes(self):
        tok=DummyTokenizer()
        tok.apply_chat_template=lambda *args, **kwargs:{'input_ids': [200001, 1000, 200002]}
        self.assertEqual(tokenize_question(tok,'x'),[200001,1000,200002])

    def test_invalid_out_of_range_reports_exact_position(self):
        tok=DummyTokenizer()
        tok.apply_chat_template=lambda *args, **kwargs:[200001, 201024, 1000]
        with self.assertRaisesRegex(ValueError, r'input_ids\[1\]=201024'):
            tokenize_question(tok,'x')

    def test_ambiguous_multibatch_is_rejected(self):
        tok=DummyTokenizer()
        tok.apply_chat_template=lambda *args, **kwargs:[[1000],[1001]]
        with self.assertRaisesRegex(ValueError, r'input_ids\[0\]'):
            tokenize_question(tok,'x')

    def test_mapping_without_input_ids_is_rejected(self):
        tok=DummyTokenizer()
        tok.apply_chat_template=lambda *args, **kwargs:{'attention_mask': [1]}
        with self.assertRaisesRegex(ValueError, 'without input_ids'):
            tokenize_question(tok,'x')

    def test_array_like_return(self):
        tok=DummyTokenizer()
        class FakeArray:
            def tolist(self): return [[1000,1001]]
        tok.apply_chat_template=lambda *args,**kwargs:FakeArray()
        self.assertEqual(tokenize_question(tok,'x'),[1000,1001])

if __name__=='__main__': unittest.main()
