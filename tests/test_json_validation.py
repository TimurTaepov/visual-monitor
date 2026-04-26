import unittest

from vlm_evals.eval.json_validation import extract_json_object, parse_and_validate, parse_json_output


class JsonValidationTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        raw = 'Here:\n```json\n{"answer": true}\n```'
        self.assertEqual(extract_json_object(raw), '{"answer": true}')

    def test_parse_json_output(self):
        parsed, valid, error = parse_json_output('prefix {"answer": true} suffix')
        self.assertTrue(valid)
        self.assertIsNone(error)
        self.assertEqual(parsed, {"answer": True})

    def test_schema_validation_rejects_extra_fields(self):
        result = parse_and_validate(
            '{"answer": true, "confidence": 0.9, "evidence": "visible", "requires_review": false, "extra": 1}',
            "object_presence_schema",
        )
        self.assertTrue(result["valid_json"])
        self.assertFalse(result["schema_valid"])


if __name__ == "__main__":
    unittest.main()

