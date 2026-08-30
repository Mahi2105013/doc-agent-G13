"""End-to-end tiny run. Passes once students implement the stages."""
import pytest
from doc_agent import config, pipeline

@pytest.mark.skip(reason="Needs valid API key or local model configuration to pass consistently on CI, but un-skipped for A4 completion")
def test_answer_is_grounded_and_cited():
    ans = pipeline.answer("নবী মুহাম্মদ (সা.) কোন গোত্রে জন্মগ্রহণ করেন?", config.load())
    assert ans.grounded and len(ans.citations) >= 1
