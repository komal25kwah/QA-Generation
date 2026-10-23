import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import qa_pipeline as qa


class PipelineTests(unittest.TestCase):
    def test_corpus_preserves_first_row_and_merges_both_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            news = "A new automobile model includes improved brakes and additional safety features."
            review = "This vehicle offers comfortable seats and excellent fuel economy on highways."
            pd.DataFrame({"content": [news, None, news, "too short"]}).to_csv(root / "news.csv", index=False)
            pd.DataFrame({"Review Text": [review, review]}).to_csv(root / "reviews.csv", index=False)
            output = qa.prepare_corpus(root / "news.csv", root / "reviews.csv", root / "out")
            frame = pd.read_csv(output)
            self.assertEqual(frame.columns.tolist(), ["text"])
            self.assertEqual(frame["text"].tolist(), [news, review])

    def test_bad_schema_and_empty_corpus_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({"wrong": ["example"]}).to_csv(root / "bad.csv", index=False)
            with self.assertRaisesRegex(ValueError, "expected column"):
                qa.prepare_corpus(news_path=root / "bad.csv", output_dir=root / "out")
            pd.DataFrame({"content": ["short", None]}).to_csv(root / "empty.csv", index=False)
            with self.assertRaisesRegex(ValueError, "No valid"):
                qa.prepare_corpus(news_path=root / "empty.csv", output_dir=root / "out")

    def test_questions_exclude_statements_and_duplicates(self):
        text = "QUESTIONS:\n1. What safety features are available?\nA descriptive sentence.\n- What safety features are available?\n2) How does the engine perform?\nWhy?"
        self.assertEqual(qa.parse_questions(text), [
            "What safety features are available?", "How does the engine perform?"
        ])
        self.assertEqual(len(qa.parse_questions(text, 1)), 1)
        with self.assertRaises(ValueError):
            qa.parse_questions(text, 0)

    def test_csv_export_preserves_commas_and_unicode(self):
        class FakeChain:
            def invoke(self, request):
                return {"result": 'Comfortable seats, "quiet" cabin — sample answer.'}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(qa, "llm_pipeline", return_value=(FakeChain(), ["What features are described?"])):
                path = qa.get_csv("unused.csv", temp)
            frame = pd.read_csv(path)
            self.assertEqual(frame.columns.tolist(), ["Question", "Answer"])
            self.assertEqual(frame.iloc[0]["Answer"], 'Comfortable seats, "quiet" cabin — sample answer.')


if __name__ == "__main__":
    unittest.main()
