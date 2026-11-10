from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.preprocessing import download_online_retail_ii,load_retail_excel,build_invoice_features
from src.anomaly_detection import fit_isolation_forest
from src.reliability import quality_dimensions
from src.kpi_calculation import calculate_kpis

st.set_page_config(page_title="SME-DataTrust BI",layout="wide")
st.title("SME-DataTrust BI")
st.caption("Prototipo BI con perfilamiento de calidad y detección de anomalías.")
uploaded=st.file_uploader("Carga Excel compatible con Online Retail II",type=["xlsx","xls"])
if uploaded:
    path=ROOT/"data/raw/uploaded.xlsx"; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(uploaded.getvalue())
    df=load_retail_excel(path)
else:
    cache=ROOT/"data/processed/retail_lines.pkl"
    if not cache.exists():
        if st.button("Descargar Online Retail II desde UCI"):
            with st.spinner("Descargando..."):
                xlsx=download_online_retail_ii(ROOT/"data/raw")
                df=load_retail_excel(xlsx); df.to_pickle(cache)
            st.success("Descarga completada. Actualiza la página.")
        st.info("Carga un Excel o descarga el dataset desde el botón.")
        st.stop()
    df=pd.read_pickle(cache)
quality=quality_dimensions(df); kpi=calculate_kpis(df)
a,b,c,d=st.columns(4)
a.metric("Registros",f"{len(df):,}"); b.metric("Facturas",f"{kpi['invoices']:,}")
c.metric("Clientes",f"{kpi['customers']:,}"); d.metric("Calidad observable",f"{quality['observed_quality_index']:.1f}/100")
with st.expander("Dimensiones de calidad"): st.json(quality)
invoice=build_invoice_features(df)
cont=st.slider("Contaminación configurada",.01,.15,.03,.01)
_,scored,_=fit_isolation_forest(invoice,contamination=cont)
st.metric("Facturas señaladas",f"{int(scored.if_anomaly.sum()):,}",f"{scored.if_anomaly.mean()*100:.2f}%")
st.plotly_chart(px.histogram(scored,x="if_score",color="if_anomaly",title="Puntajes de anomalía"),use_container_width=True)
st.warning("Una anomalía estadística es una señal para revisión, no prueba de error. No se eliminan registros automáticamente.")
st.dataframe(scored[scored.if_anomaly==1].sort_values("if_score",ascending=False).head(500),use_container_width=True)
st.download_button("Descargar resultados CSV",scored.to_csv(index=False).encode(), "anomaly_results.csv","text/csv")
