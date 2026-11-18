"""Choosing a quantized build for the Kaggle hardware, from Hugging Face file lists."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm_select as s

GB = 10**9
T4X2 = {'gpus': [15360, 15360], 'ram_gib': 31.3, 'gpu_gbps': 300, 'cpu_gbps': 20}
CPU_SESSION = {'gpus': [], 'ram_gib': 31.3, 'cpu_gbps': 20}
# Qwen3-32B: 64 layers, 8 KV heads of 128 -> 256 KiB of fp16 KV cache per token.
QWEN3_32B = {'num_hidden_layers': 64, 'num_attention_heads': 64, 'num_key_value_heads': 8, 'head_dim': 128, 'hidden_size': 5120}
QWEN3_30B_A3B = {'num_hidden_layers': 48, 'num_attention_heads': 32, 'num_key_value_heads': 4, 'head_dim': 128,
                 'num_experts': 128, 'num_experts_per_tok': 8}


def files(*pairs):
    return [{'type': 'file', 'path': path, 'size': size} for path, size in pairs]


QWEN3_32B_FILES = files(
    ('Qwen3-32B-Q4_K_M.gguf', 19.8 * GB), ('Qwen3-32B-Q5_K_M.gguf', 23.2 * GB), ('Qwen3-32B-Q6_K.gguf', 26.9 * GB),
    ('Q8_0/Qwen3-32B-Q8_0-00001-of-00002.gguf', 20.0 * GB), ('Q8_0/Qwen3-32B-Q8_0-00002-of-00002.gguf', 14.8 * GB),
    ('Qwen3-32B-IQ2_XXS.gguf', 9.0 * GB), ('mmproj-F16.gguf', 1.3 * GB), ('README.md', 4000), ('imatrix.gguf', 2 * GB))


class Parsing(unittest.TestCase):
    def test_builds_group_split_files_and_skip_projectors(self):
        builds = s.gguf_builds(QWEN3_32B_FILES)
        quants = sorted(b['quant'] for b in builds.values())
        self.assertEqual(quants, ['IQ2_XXS', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'])
        q8 = next(b for b in builds.values() if b['quant'] == 'Q8_0')
        self.assertEqual(q8['first'], 'Q8_0/Qwen3-32B-Q8_0-00001-of-00002.gguf')
        self.assertEqual(len(q8['files']), 2)
        self.assertAlmostEqual(q8['bytes'], 34.8 * GB, delta=1)

    def test_same_quant_from_two_files_stays_separate(self):
        builds = s.gguf_builds(files(('m-Q4_K_M.gguf', 5 * GB), ('m-UD-Q4_K_M.gguf', 5.2 * GB)))
        self.assertEqual(len(builds), 2)

    def test_lfs_sizes_and_hub_sibling_names(self):
        builds = s.gguf_builds([{'rfilename': 'a-Q8_0.gguf', 'lfs': {'size': 9 * GB}}])
        self.assertEqual(next(iter(builds.values()))['bytes'], 9 * GB)

    def test_kv_cache_and_mixture_of_experts(self):
        self.assertEqual(s.kv_bytes_per_token(QWEN3_32B), 2 * 64 * 8 * 128 * 2)
        self.assertEqual(s.kv_bytes_per_token({'text_config': QWEN3_32B}), 262144)
        self.assertIsNone(s.kv_bytes_per_token({}))
        self.assertEqual(s.active_fraction(QWEN3_32B), 1.0)
        self.assertLess(s.active_fraction(QWEN3_30B_A3B), 0.2)


class Choosing(unittest.TestCase):
    def test_dual_t4_takes_the_best_quant_that_fits_on_the_gpus(self):
        chosen, options = s.choose(s.gguf_builds(QWEN3_32B_FILES), T4X2, QWEN3_32B, 8192)
        self.assertEqual(chosen['quant'], 'Q5_K_M')
        self.assertEqual(chosen['placement'], 'gpu')
        placements = {o['quant']: o['placement'] for o in options}
        self.assertEqual(placements['Q6_K'], 'hybrid')
        self.assertNotIn('Q8_0', [o['quant'] for o in options if o['placement'] == 'gpu'])

    def test_longer_context_leaves_less_room_for_weights(self):
        chosen, _ = s.choose(s.gguf_builds(QWEN3_32B_FILES), T4X2, QWEN3_32B, 32768)
        self.assertEqual(chosen['quant'], 'Q4_K_M')

    def test_low_bit_builds_only_when_nothing_better_fits(self):
        small = {'gpus': [12288], 'ram_gib': 16, 'gpu_gbps': 300, 'cpu_gbps': 20}
        chosen, _ = s.choose(s.gguf_builds(QWEN3_32B_FILES), small, QWEN3_32B, 4096)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen['quant'], 'IQ2_XXS')

    def test_cpu_session_prefers_a_mixture_of_experts_speed(self):
        moe = files(('Qwen3-30B-A3B-Q4_K_M.gguf', 18.6 * GB), ('Qwen3-30B-A3B-Q8_0.gguf', 32.5 * GB))
        chosen, options = s.choose(s.gguf_builds(moe), CPU_SESSION, QWEN3_30B_A3B, 8192)
        self.assertEqual(chosen['quant'], 'Q4_K_M')
        self.assertEqual(chosen['placement'], 'cpu')
        self.assertGreater(chosen['tok_per_s'], 2)
        self.assertEqual([o['quant'] for o in options], ['Q4_K_M'])  # Q8_0 needs more RAM than the session has

    def test_nothing_fits(self):
        huge = files(('big-Q4_K_M.gguf', 400 * GB))
        self.assertEqual(s.choose(s.gguf_builds(huge), T4X2, {}, 8192), (None, []))


class Repositories(unittest.TestCase):
    def test_official_then_trusted_then_downloads(self):
        repos = ['someone/Qwen3-32B-GGUF', 'bartowski/Qwen_Qwen3-32B-GGUF', 'Qwen/Qwen3-32B-GGUF', 'unsloth/Qwen3-32B-GGUF', 'x/Other-GGUF']
        ranked = s.rank_repos('Qwen3-32B', repos)
        self.assertEqual(ranked[0], 'Qwen/Qwen3-32B-GGUF')
        # Trusted converters next, in download order (the search's order).
        self.assertEqual(ranked[1:3], ['bartowski/Qwen_Qwen3-32B-GGUF', 'unsloth/Qwen3-32B-GGUF'])
        self.assertEqual(ranked[3], 'someone/Qwen3-32B-GGUF')
        self.assertEqual(ranked[-1], 'x/Other-GGUF')

    def test_find_uses_the_hub_and_returns_the_choice(self):
        calls = []
        def get(url, timeout=30):
            calls.append(url)
            if '/api/models?' in url:
                return [{'id': 'someone/Qwen3-32B-GGUF'}, {'id': 'Qwen/Qwen3-32B-GGUF'}]
            if url.endswith('/tree/main?recursive=true'):
                return QWEN3_32B_FILES
            if url.endswith('config.json'):
                return QWEN3_32B
            raise AssertionError(url)
        original = s._get
        s._get = get
        try:
            best, considered = s.find('Qwen3-32B', T4X2)
        finally:
            s._get = original
        self.assertEqual(best['repo'], 'Qwen/Qwen3-32B-GGUF')
        self.assertEqual(best['quant'], 'Q5_K_M')
        self.assertEqual(best['kv_bytes_per_token'], 262144)
        self.assertIn('search=Qwen3-32B', calls[0])


if __name__ == '__main__':
    unittest.main()
