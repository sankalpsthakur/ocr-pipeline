import pytest
import sys, pathlib

# Add repository root to path so tests can import modules directly
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import extractor


def test_energy_with_comma():
    text = 'Total usage: 1,234 kWh'
    result = extractor.extract_fields(text)
    assert result['electricity_kwh'] == 1234


def test_carbon_with_subscript():
    text = 'Emission: Kg\u202fCO\u2082e\u202f567'
    result = extractor.extract_fields(text)
    assert result['carbon_kgco2e'] == 567
