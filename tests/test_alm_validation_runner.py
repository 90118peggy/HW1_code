import unittest
from scripts.evaluate_alm_validation import CLASSES, metric, prompts
from scripts.smoke_qwen2_audio_v3 import parse_output


class ValidationProtocolTests(unittest.TestCase):
    def test_parser_accepts_only_supported_exact_rankings(self):
        cases = [('["US", "UK", "Germany"]', 'json'),
                 ("['US', 'UK', 'Germany']", 'python_list'),
                 (r'[\"US\", \"UK\", \"Germany\"]', 'escaped_quotes'),
                 ('```json\n["US", "UK", "Germany"]\n```', 'json_code_fence')]
        for raw, syntax in cases:
            with self.subTest(syntax=syntax):
                result = parse_output(raw, CLASSES['dataset_B'])
                self.assertTrue(result['valid_output'])
                self.assertEqual(result['top3'], ['US', 'UK', 'Germany'])
                self.assertEqual(result['output_syntax'], syntax)
                self.assertEqual(result['raw_json_compliant'], syntax == 'json')

    def test_parser_rejects_duplicates_unknown_labels_extra_prose_wrong_length(self):
        for raw in ['["US", "US", "UK"]', '["US", "UK", "France"]',
                    '["US", "UK"]', '["US", "UK", "Germany", "Italy"]',
                    'Answer: ["US", "UK", "Germany"]', '["us", "UK", "Germany"]',
                    '{"top3":["US", "UK", "Germany"]}', '__import__("os").getcwd()']:
            with self.subTest(raw=raw):
                self.assertFalse(parse_output(raw, CLASSES['dataset_B'])['valid_output'])

    def test_invalid_is_in_full_denominator_and_matrix(self):
        rows = []
        for label, raw in [('US', '["US", "UK", "Germany"]'),
                           ('UK', '["US", "UK", "Germany"]'), ('Italy', 'invalid')]:
            answer = parse_output(raw, CLASSES['dataset_B'])
            rows.append({'label': label, 'initial': answer})
        result = metric(rows, CLASSES['dataset_B'], 'initial')
        self.assertAlmostEqual(result['top1'], 1/3)
        self.assertAlmostEqual(result['top3'], 2/3)
        self.assertEqual(result['invalid_outputs'], 1)
        self.assertEqual(sum(map(sum, result['confusion_matrix_with_invalid'])), 3)
        self.assertEqual(result['confusion_matrix_with_invalid'][2][-1], 1)

    def test_prompts_use_only_fixed_task_information(self):
        for dataset in CLASSES:
            value = prompts(dataset)
            self.assertEqual(set(value), {'direct', 'acoustic'})
            for text in value.values():
                self.assertTrue(all(label in text for label in CLASSES[dataset]))
                self.assertIn('no explanation or other text', text)


if __name__ == '__main__':
    unittest.main()

