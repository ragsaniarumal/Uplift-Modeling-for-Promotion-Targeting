import pandas as pd
from pricesense.experiment import ab_summary


def sample_df():
    rows=[]
    for i in range(50):
        rows.append({"ID":i,"Promotion":"Yes","purchase":1 if i<10 else 0,**{f"V{j}":1 for j in range(1,8)}})
    for i in range(50,100):
        rows.append({"ID":i,"Promotion":"No","purchase":1 if i<55 else 0,**{f"V{j}":1 for j in range(1,8)}})
    return pd.DataFrame(rows)


def test_ab_summary_computes_incremental_response():
    s=ab_summary(sample_df())
    assert abs(s["incremental_response_rate"]-0.10)<1e-12


def test_blanket_nir_uses_promotion_cost():
    s=ab_summary(sample_df())
    assert abs(s["blanket_expected_nir"]-85.0)<1e-12
