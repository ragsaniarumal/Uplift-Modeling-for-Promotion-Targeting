import numpy as np
import pandas as pd
from pricesense.models import policy_result, split_experiment


def make_data(n=400):
    rng=np.random.default_rng(1)
    rows=[]
    for i in range(n):
        treatment=i%2
        x=rng.normal()
        purchase=int(rng.random() < (0.05 + 0.10*treatment*(x>0)))
        rows.append({"ID":i,"Promotion":"Yes" if treatment else "No","purchase":purchase,
                     "V1":int(i%4),"V2":x,"V3":rng.normal(),"V4":1,"V5":2,"V6":3,"V7":1})
    return pd.DataFrame(rows)


def test_split_preserves_all_rows():
    d=make_data()
    s=split_experiment(d)
    assert len(s.train)+len(s.validation)+len(s.test)==len(d)


def test_policy_targets_requested_fraction():
    d=make_data(200)
    d["treatment"]=(d.Promotion=="Yes").astype(int)
    score=np.linspace(0,1,len(d))
    r=policy_result(d,score,0.5)
    assert r["targeted_customers"]==100
