import pandas as pd
from src.preprocessing import build_invoice_features
from src.reliability import quality_dimensions

def test_quality_index_range():
    d=pd.DataFrame({"Quantity":[1,2],"UnitPrice":[10,20],"InvoiceDate":pd.to_datetime(["2020-01-01","2020-01-02"])})
    assert 0 <= quality_dimensions(d)["observed_quality_index"] <= 100

def test_invoice_aggregation():
    d=pd.DataFrame({"InvoiceNo":["A","A","B"],"Quantity":[1,2,1],"UnitPrice":[10,5,20],
      "Revenue":[10,10,20],"InvoiceDate":pd.to_datetime(["2020-01-01"]*3),
      "CustomerID":[1,1,2],"Country":["UK"]*3})
    assert len(build_invoice_features(d)) == 2
