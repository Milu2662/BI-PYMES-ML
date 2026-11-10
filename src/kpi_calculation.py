import pandas as pd
import numpy as np

def calculate_kpis(lines):
    return {"records":int(len(lines)),"invoices":int(lines.InvoiceNo.nunique()),
      "customers":int(lines.CustomerID.nunique()),"recorded_revenue":float(lines.Revenue.sum()),
      "quantity_sum":float(lines.Quantity.sum()),"mean_unit_price":float(lines.UnitPrice.mean()),
      "cancellation_lines":int(lines.InvoiceNo.astype("string").str.startswith("C",na=False).sum())}

def kpi_sensitivity(lines, invoice_scores, flag_col="if_anomaly"):
    full=calculate_kpis(lines)
    flagged=set(invoice_scores.loc[invoice_scores[flag_col]==1,"InvoiceNo"].astype(str))
    remaining=lines.loc[~lines.InvoiceNo.astype(str).isin(flagged)]
    reviewed=calculate_kpis(remaining); rows=[]
    for k,v in full.items():
        w=reviewed[k]; delta=v-w
        rows.append({"kpi":k,"all_records":v,"excluding_flagged_invoices":w,
                     "absolute_difference":delta,
                     "difference_pct_of_all":delta/abs(v)*100 if v else np.nan})
    return pd.DataFrame(rows)
