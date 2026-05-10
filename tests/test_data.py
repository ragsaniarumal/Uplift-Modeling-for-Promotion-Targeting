import pandas as pd
import pytest
from pricesense.data import validate


def test_validate_accepts_expected_schema():
    row = {"ID": 1, "Promotion": "Yes", "purchase": 0, **{f"V{i}": 1 for i in range(1, 8)}}
    assert len(validate(pd.DataFrame([row]))) == 1


def test_validate_rejects_bad_treatment():
    row = {"ID": 1, "Promotion": "Maybe", "purchase": 0, **{f"V{i}": 1 for i in range(1, 8)}}
    with pytest.raises(ValueError):
        validate(pd.DataFrame([row]))
