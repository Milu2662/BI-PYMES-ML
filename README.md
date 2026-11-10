# SME-DataTrust BI

Prototipo de investigación de Business Intelligence con perfilamiento de calidad de datos, detección de anomalías transaccionales mediante Machine Learning y análisis de sensibilidad de KPI.

## Dataset
Online Retail II, UCI Machine Learning Repository. DOI: 10.24432/C5CG6D.
Ficha oficial: https://archive.ics.uci.edu/dataset/502/online+retail+ii
El dataset se descarga al ejecutar el Notebook 01; no se incluye en este ZIP por su tamaño. Representa un minorista británico, no una muestra representativa de PYMES colombianas.

## Ejecución
1. Sube la carpeta del proyecto a GitHub.
2. Abre los notebooks desde GitHub en Google Colab.
3. Ejecuta del 01 al 06 en orden y en la misma sesión.
4. Para el dashboard local: `pip install -r requirements.txt` y `streamlit run dashboard/app.py`.

## Estructura
- `notebook/`: seis notebooks reproducibles.
- `src/`: módulos reutilizables.
- `data/`: datos originales, procesados y experimentales (no versionar archivos grandes).
- `results/`: métricas y figuras.
- `dashboard/`: interfaz Streamlit.
- `paper/`: plantilla LaTeX IEEE.

## Precauciones científicas
Una anomalía estadística no es necesariamente un error. No eliminar automáticamente los registros señalados. Las etiquetas de anomalías sintéticas sirven para evaluar detección de alteraciones inyectadas, no para afirmar detección perfecta de errores reales. La comparación de KPI al excluir facturas marcadas es un análisis de sensibilidad, no una corrección validada. La originalidad debe establecerse mediante revisión sistemática de literatura; no se presupone en este repositorio.
# BI-PYMES-ML
