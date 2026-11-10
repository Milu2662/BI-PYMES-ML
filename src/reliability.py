import numpy as np

def quality_dimensions(df):
    n=max(len(df),1); completeness=1-df.isna().sum().sum()/(n*max(df.shape[1],1))
    uniqueness=1-df.duplicated().sum()/n
    issue=0; checks=0
    for col, bad in [("Quantity",lambda s:s.isna()|(s==0)),("UnitPrice",lambda s:s.isna()|(s<0)),("InvoiceDate",lambda s:s.isna())]:
        if col in df: issue += int(bad(df[col]).sum()); checks += len(df)
    validity=1-issue/max(checks,1)
    dims={"completeness":float(np.clip(completeness,0,1)),
          "uniqueness":float(np.clip(uniqueness,0,1)),
          "rule_based_validity":float(np.clip(validity,0,1))}
    dims["observed_quality_index"]=100*(.4*dims["completeness"]+.3*dims["uniqueness"]+.3*dims["rule_based_validity"])
    return dims
