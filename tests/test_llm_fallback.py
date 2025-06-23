import sys, pathlib, json
from unittest.mock import patch
import os

# Add repository root to path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import pipeline


def test_gpt4o_fallback_parsing():
    class DummyResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    dummy_response = DummyResp(json.dumps({"electricity_kwh": 250, "carbon_kgco2e": 100}))

    with patch.object(pipeline.config, "OPENAI_API_KEY", "sk-test"):
        with patch("pipeline.openai.chat.completions.create", return_value=dummy_response):
            tmp = pathlib.Path("dummy.png")
            tmp.write_bytes(b"fake")
            try:
                fields = pipeline.gpt4o_fallback(tmp)
            finally:
                tmp.unlink()
            assert fields == {"electricity_kwh": 250, "carbon_kgco2e": 100}

