import pandas as pd
from pricesense.models import split_experiment


def test_split_is_deterministic():
    rows=[]
    for i in range(500):
        rows.append({"ID":i,"Promotion":"Yes" if i%2 else "No","purchase":1 if i%37==0 else 0,
                     **{f"V{j}": (i+j)%4 for j in range(1,8)}})
    d=pd.DataFrame(rows)
    a=split_experiment(d)
    b=split_experiment(d)
    assert a.test.ID.tolist()==b.test.ID.tolist()
