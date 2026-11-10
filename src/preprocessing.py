from pathlib import Path
import zipfile, requests
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

def download_online_retail_ii(raw_dir):
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    zp = raw_dir / "online_retail_ii.zip"
    if not zp.exists():
        r = requests.get(UCI_ZIP_URL, timeout=180); r.raise_for_status(); zp.write_bytes(r.content)
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not names: raise FileNotFoundError("No se encontró Excel en el ZIP de UCI.")
        for n in names:
            target = raw_dir / Path(n).name
            if not target.exists(): target.write_bytes(z.read(n))
    return raw_dir / Path(names[0]).name

def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "") for c in df.columns]
    aliases = {"Invoice":"InvoiceNo","InvoiceNo":"InvoiceNo","Price":"UnitPrice","UnitPrice":"UnitPrice",
               "Customer_ID":"CustomerID","CustomerID":"CustomerID"}
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    required = ["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Faltan columnas {missing}; columnas recibidas: {list(df.columns)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for c in ["Quantity","UnitPrice","CustomerID"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df

def load_retail_excel(path):
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for name, frame in sheets.items():
        frame = frame.copy(); frame["SourceSheet"] = str(name); frames.append(frame)
    if not frames: raise ValueError("El Excel no contiene hojas.")
    return standardize_columns(pd.concat(frames, ignore_index=True))

def quality_profile(df):
    profile = pd.DataFrame({"dtype":df.dtypes.astype(str),"missing_count":df.isna().sum(),
        "missing_pct":(df.isna().mean()*100).round(3),"unique_count":df.nunique(dropna=True)})
    checks = {"rows":int(len(df)),"columns":int(df.shape[1]),"exact_duplicates":int(df.duplicated().sum()),
        "quantity_negative":int((df.Quantity<0).sum()),"quantity_zero":int((df.Quantity==0).sum()),
        "price_negative":int((df.UnitPrice<0).sum()),"invalid_dates":int(df.InvoiceDate.isna().sum()),
        "cancellation_invoices":int(df.InvoiceNo.astype("string").str.startswith("C",na=False).sum())}
    return profile, checks

def build_invoice_features(df):
    d=df.copy()
    d["is_cancel"]=d.InvoiceNo.astype("string").str.startswith("C",na=False).astype(int)
    out=d.groupby("InvoiceNo",dropna=False).agg(
        line_count=("InvoiceNo","size"), total_quantity=("Quantity","sum"),
        mean_quantity=("Quantity","mean"), mean_unit_price=("UnitPrice","mean"),
        total_revenue=("Revenue","sum"), revenue_abs=("Revenue",lambda x:x.abs().sum()),
        min_unit_price=("UnitPrice","min"), max_unit_price=("UnitPrice","max"),
        cancellation_line_share=("is_cancel","mean"), invoice_date=("InvoiceDate","min"),
        customer_id=("CustomerID","first"), country=("Country","first")).reset_index()
    out["invoice_month"]=out.invoice_date.dt.to_period("M").astype("string")
    return out
