import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Dashboard Emas — FTS & MARIMA", page_icon="🪙", layout="wide")
st.markdown("""
<style>
:root{
  --navy-0: #07102a;
  --navy-1: #0b1736;
  --gold:#f7c948;
  --muted:#cbd5e1;
  --card-bg: rgba(255,255,255,0.03);
}
.stApp, html, body {
  background: linear-gradient(180deg, var(--navy-0) 0%, var(--navy-1) 100%) !important;
  color: #f1f5f9 !important;
}
.block-container {padding: 28px 36px 48px 36px; max-width: 1400px; margin: auto;}
.hero {
  background: linear-gradient(90deg, rgba(247,201,72,0.07), rgba(255,255,255,0.03));
  padding: 24px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 24px;
}
.kpi {
  background: var(--card-bg); border-radius: 12px; padding: 14px 16px;
  border-left: 6px solid var(--gold);
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
  margin: 8px;
  min-width: 260px; display: inline-block;
}
.kpi .title {font-size: 12px; color: var(--muted); text-transform: uppercase; font-weight: 700;}
.kpi .value {font-size: 20px; font-weight: 800; color: var(--gold);}
.small{color: var(--muted); font-size: 13px;}
.card-glass {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 12px; padding: 16px;
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  margin-top: 12px; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

def format_idr(v):
    try:
        return f"Rp {int(round(v)):,}".replace(",", ".")
    except:
        return "Rp N/A"

def trend_arrow(current, previous):
    try:
        if current > previous: return "📈 Naik"
        if current < previous: return "📉 Turun"
    except:
        pass
    return "Stabil"
# Load Data and Preprocessing
@st.cache_data
def load_and_prep(file_hist, file_fts, file_vecm, file_corr, file_eval):
    """Load dan bersihkan semua data."""
    try:
        df_hist = pd.read_csv(file_hist)
        df_hist['Waktu'] = pd.to_datetime(df_hist['Waktu'], errors='coerce', dayfirst=True, infer_datetime_format=True)
        df_hist['Waktu'] = df_hist['Waktu'].dt.to_period('M').dt.start_time

        df_fts = pd.read_csv(file_fts)
        if 'month' in df_fts.columns: df_fts.rename(columns={'month': 'Waktu'}, inplace=True)
        if 'forecast' in df_fts.columns: df_fts.rename(columns={'forecast': 'Forecast_FTS'}, inplace=True)
        # Ensure column for actual if present
        for col in ['actual', 'Harga Emas', 'Actual']:
            if col in df_fts.columns:
                if col != 'Harga Emas':
                    df_fts.rename(columns={col: 'Harga Emas_FTS_Actual'}, inplace=True)
                break
        df_fts['Waktu'] = pd.to_datetime(df_fts['Waktu'], errors='coerce').map(lambda d: d.replace(day=1) if pd.notnull(d) else d)
        df_fts = df_fts.sort_values('Waktu').drop_duplicates(subset='Waktu', keep='last')

        df_vecm = pd.read_csv(file_vecm)
        if 'Tanggal' in df_vecm.columns: df_vecm.rename(columns={'Tanggal': 'Waktu'}, inplace=True)
        col_pred = [c for c in df_vecm.columns if 'Prediksi' in c or 'Forecast' in c or 'forecast' in c]
        if col_pred: df_vecm.rename(columns={col_pred[0]: 'Forecast_VECM'}, inplace=True)
        df_vecm['Waktu'] = pd.to_datetime(df_vecm['Waktu'], errors='coerce').map(lambda d: d.replace(day=1) if pd.notnull(d) else d)
        df_vecm = df_vecm.sort_values('Waktu').drop_duplicates(subset='Waktu', keep='last')

        # correlation and evaluation files (may raise if missing)
        try:
            df_corr = pd.read_excel(file_corr)
        except Exception:
            df_corr = pd.DataFrame(columns=['Faktor', 'Korelasi'])

        try:
            df_eval = pd.read_csv(file_eval)
        except Exception:
            df_eval = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'MAPE (%)'])

        combined = (
            df_hist[['Waktu', 'Harga Emas']].merge(
                df_fts[['Waktu', 'Forecast_FTS']], on='Waktu', how='outer'
            ).merge(
                df_vecm[['Waktu', 'Forecast_VECM']], on='Waktu', how='outer'
            ).sort_values('Waktu')
        )

        return df_hist, df_fts, df_vecm, df_corr, df_eval, combined
    except Exception as e:
        st.error(f" Gagal memuat data: {e}")
        st.stop()

# --- path file (sesuaikan jika letak file berbeda) ---
PATH_HIST = "Data Forecast Emas.csv"
PATH_FTS_FULL = "forecast_gabungan_month_forecast.csv"
PATH_VECM_FULL = "Forecast VECM Full.csv"
PATH_CORR = "Korelasi.xlsx"
PATH_EVAL = "Evaluasi.csv"

df_hist, df_fts_full, df_vecm_full, df_corr, df_eval, combined = load_and_prep(
    PATH_HIST, PATH_FTS_FULL, PATH_VECM_FULL, PATH_CORR, PATH_EVAL
)

# Header
st.markdown('<div class="hero"><h2 style="color:var(--gold)"> Dashboard Peramalan Harga Emas</h2><div class="small">Visualisasi perbandingan model Fuzzy Time Series (FMTS) dan MARIMA (VECM).</div></div>', unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Card
if df_hist.empty:
    st.warning("Dataset historis kosong. Pastikan Data Forecast Emas.csv berisi data.")
    st.stop()

# protect indexing
if len(df_hist) >= 2:
    harga_terakhir = df_hist['Harga Emas'].iloc[-1]
    harga_terakhir_waktu = df_hist['Waktu'].iloc[-1]
    harga_sebelumnya = df_hist['Harga Emas'].iloc[-2]
else:
    harga_terakhir = df_hist['Harga Emas'].iloc[-1]
    harga_terakhir_waktu = df_hist['Waktu'].iloc[-1]
    harga_sebelumnya = harga_terakhir

pred_fmts_next = df_fts_full[df_fts_full['Waktu'] > harga_terakhir_waktu] if not df_fts_full.empty else pd.DataFrame()
pred_marima_next = df_vecm_full[df_vecm_full['Waktu'] > harga_terakhir_waktu] if not df_vecm_full.empty else pd.DataFrame()

if not pred_fmts_next.empty:
    pred_fmts_1 = pred_fmts_next['Forecast_FTS'].iloc[0]
    waktu_pred_fmts_1 = pred_fmts_next['Waktu'].iloc[0]
else:
    pred_fmts_1 = np.nan
    waktu_pred_fmts_1 = harga_terakhir_waktu

if not pred_marima_next.empty:
    pred_marima_1 = pred_marima_next['Forecast_VECM'].iloc[0]
    waktu_pred_marima_1 = pred_marima_next['Waktu'].iloc[0]
else:
    pred_marima_1 = np.nan
    waktu_pred_marima_1 = harga_terakhir_waktu

col1, col2, col3 = st.columns(3, gap="large")
col1.markdown(f"<div class='kpi'><div class='title'>Harga Terakhir ({harga_terakhir_waktu.strftime('%b %Y')})</div><div class='value'>{format_idr(harga_terakhir)}</div><div class='small'>{trend_arrow(harga_terakhir, harga_sebelumnya)}</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='kpi'><div class='title'>Prediksi FMTS ({waktu_pred_fmts_1.strftime('%b %Y')})</div><div class='value'>{format_idr(pred_fmts_1)}</div><div class='small'>{trend_arrow(pred_fmts_1, harga_terakhir)}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='kpi'><div class='title'>Prediksi MARIMA ({waktu_pred_marima_1.strftime('%b %Y')})</div><div class='value'>{format_idr(pred_marima_1)}</div><div class='small'>{trend_arrow(pred_marima_1, harga_terakhir)}</div></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Beranda", "FMTS Detail", "MARIMA Detail", "Live Prediction"])
def yaxis_range_for(*dfs, col_names=('Harga Emas', 'Forecast_FTS', 'Forecast_VECM')):
    vals = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for c in col_names:
            if c in df.columns:
                vals += list(pd.to_numeric(df[c], errors='coerce').dropna().values)
    if not vals:
        return dict(showgrid=True, zeroline=True, showline=True)
    mn = min(vals)
    mx = max(vals)
    if mn == mx:
        padding = mx * 0.05 if mx != 0 else 1
        mn -= padding
        mx += padding
    else:
        padding = (mx - mn) * 0.08
        mn -= padding
        mx += padding
    return dict(range=[mn, mx], showgrid=True, zeroline=True, showline=True)

# Tab 1
with tab1:
    st.markdown("### A. Grafik Gabungan: Historis & Prediksi")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['Waktu'], y=df_hist['Harga Emas'], name="Aktual", line=dict(color="#f7c948", width=3), line_shape="spline"))
    if 'Forecast_FTS' in df_fts_full.columns:
        fig.add_trace(go.Scatter(x=df_fts_full['Waktu'], y=df_fts_full['Forecast_FTS'], name="FMTS", line=dict(color="#60a5fa", width=2.5, dash="dot"), line_shape="spline"))
    if 'Forecast_VECM' in df_vecm_full.columns:
        fig.add_trace(go.Scatter(x=df_vecm_full['Waktu'], y=df_vecm_full['Forecast_VECM'], name="MARIMA", line=dict(color="#22c55e", width=2.5, dash="dot"), line_shape="spline"))
    # y-axis scale covering actual + both forecasts
    yax = yaxis_range_for(df_hist, df_fts_full, df_vecm_full)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=480,
                      xaxis_title="Waktu", yaxis_title="Harga Emas (Rp)",
                      yaxis=yax)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### B. Faktor yang Mempengaruhi Harga Emas")
    if not df_corr.empty:
        fig_corr = px.bar(df_corr.sort_values('Korelasi', ascending=True), x='Korelasi', y='Faktor', orientation='h', color='Korelasi', color_continuous_scale='RdYlBu_r', range_color=[-1,1])
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=350,
                               yaxis=dict(showgrid=True, zeroline=True, showline=True))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("File korelasi (Korelasi.xlsx) kosong atau tidak ditemukan.")

    st.markdown("### C. Insight & Rangkuman")
    try:
        mape_fmts = df_eval[df_eval['Model'] == 'Fuzzy']['MAPE (%)'].values[0]
        mape_marima = df_eval[df_eval['Model'] == 'Marima']['MAPE (%)'].values[0]
    except Exception:
        mape_fmts = "N/A"
        mape_marima = "N/A"
    st.markdown(f"""
    * Model **FMTS (Fuzzy Time Series)** memberikan hasil stabil dengan tingkat kesalahan MAPE **{mape_fmts}%**.
    * Model **MARIMA (VECM)** menunjukkan hasil lebih halus dan akurat dengan MAPE **{mape_marima}%**.
    * Faktor paling berpengaruh terhadap harga emas biasanya adalah **Kurs USD** dan **Suku Bunga** (lihat file korelasi).
    """)

# Tab 2
with tab2:
    st.markdown("### B. Detail FMTS (Fuzzy Time Series)")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df_hist['Waktu'], y=df_hist['Harga Emas'], name="Aktual", line=dict(color="#f7c948", width=2.5), line_shape="spline"))
    if 'Forecast_FTS' in df_fts_full.columns:
        fig_f.add_trace(go.Scatter(x=df_fts_full['Waktu'], y=df_fts_full['Forecast_FTS'], name="Prediksi FMTS", line=dict(color="#60a5fa", width=2.5, dash="dot"), line_shape="spline"))
    yax_f = yaxis_range_for(df_hist, df_fts_full, col_names=('Harga Emas','Forecast_FTS'))
    fig_f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=450, yaxis=yax_f)
    st.plotly_chart(fig_f, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Cara Membaca Grafik")
        st.markdown(f"""
        * Garis **kuning**: data aktual harga emas hingga {harga_terakhir_waktu.strftime('%B %Y')}.
        * Garis **biru titik-titik**: hasil prediksi FMTS (gabungan in-sample dan forecast).
        """)
    with col2:
        st.markdown("### Evaluasi Model FMTS")
        try:
            m = df_eval[df_eval['Model'] == 'Fuzzy'].iloc[0]
            st.metric("MAE", f"{m['MAE']:,.2f}")
            st.metric("RMSE", f"{m['RMSE']:,.2f}")
            st.metric("MAPE", f"{m['MAPE (%)']:.2f}%")
        except Exception:
            st.warning("Data evaluasi FMTS tidak tersedia.")

    st.markdown("### Insight :")
    st.markdown("""
    1. Akurasi Superior: Dengan nilai MAPE hanya 1.65%, model FMTS menunjukkan tingkat akurasi yang luar biasa tinggi (tertinggi di antara kedua model). Ini berarti rata-rata kesalahan perkiraan harga emas relatif sangat kecil.
    2. Pola Prediksi Step-Like: Peramalan jangka panjang (tahun 2025 akhir hingga 2026) memproyeksikan tren kenaikan yang signifikan. Pola pergerakannya khas, yaitu berbentuk seperti tangga (step-like) alih-alih kurva mulus. Pola ini adalah hasil alami dari proses diskretisasi dan kuantisasi yang digunakan dalam metode Fuzzy Time Series.
    3. Implikasi Investasi: Kenaikan yang diproyeksikan (di atas Rp 1.8 Juta) menunjukkan prospek harga emas yang sangat positif dalam jangka waktu prediksi.
    """)

# Tab 3
with tab3:
    st.markdown("### C. Detail MARIMA (VECM)")
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=df_hist['Waktu'], y=df_hist['Harga Emas'], name="Aktual", line=dict(color="#f7c948", width=2.5), line_shape="spline"))
    if 'Forecast_VECM' in df_vecm_full.columns:
        fig_m.add_trace(go.Scatter(x=df_vecm_full['Waktu'], y=df_vecm_full['Forecast_VECM'], name="Prediksi MARIMA", line=dict(color="#22c55e", width=2.5, dash="dot"), line_shape="spline"))
    yax_m = yaxis_range_for(df_hist, df_vecm_full, col_names=('Harga Emas','Forecast_VECM'))
    fig_m.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=450, yaxis=yax_m)
    st.plotly_chart(fig_m, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Cara Membaca Grafik")
        st.markdown("""
        * Garis **kuning**: data aktual harga emas.
        * Garis **hijau titik-titik**: hasil prediksi MARIMA (gabungan fit & forecast).
        """)
    with col2:
        st.markdown("### Evaluasi Model MARIMA")
        try:
            m = df_eval[df_eval['Model'] == 'Marima'].iloc[0]
            st.metric("MAE", f"{m['MAE']:,.2f}")
            st.metric("RMSE", f"{m['RMSE']:,.2f}")
            st.metric("MAPE", f"{m['MAPE (%)']:.2f}%")
        except Exception:
            st.warning("Data evaluasi MARIMA tidak tersedia.")

    st.markdown("### Insight :")
    st.markdown("""
    1. Akurasi Tinggi dan Responsif: Model MARIMA memiliki nilai MAPE 2.30%, yang tergolong akurat. Karena VECM (Vector Error Correction Model) adalah model multivariat, ia mampu menangkap hubungan jangka panjang (kointegrasi) antara harga emas dan faktor eksternal (seperti Kurs USD atau Suku Bunga), yang membuat prediksinya lebih kontekstual.
    2. Pola Prediksi Mulus: Prediksi jangka panjang (tahun 2025 akhir hingga 2026) menunjukkan kenaikan harga emas yang sangat kuat dan berkelanjutan dengan kurva yang mulus (tidak terputus). Pola ini lebih mewakili pergerakan harga komoditas di pasar keuangan yang cenderung bergerak secara kontinu.
    3. Implikasi Investasi: Proyeksi kenaikan harga emas oleh MARIMA bahkan mencapai level tertinggi dalam periode yang diramal (di atas Rp 2 Juta di tahun 2026), menguatkan pandangan tren bullish yang kuat.
    """)


# Tab 4
with tab4:
    st.markdown("### D. Live Prediction Interaktif")
    model_choice = st.radio("Pilih Model", ["FMTS", "MARIMA"], horizontal=True)
    df_pred = df_fts_full if model_choice == "FMTS" else df_vecm_full

    # ambil data masa depan
    future = df_pred[df_pred['Waktu'] > harga_terakhir_waktu] if not df_pred.empty else pd.DataFrame()
    if future.empty:
        st.info("Tidak ada data prediksi masa depan untuk model ini.")
    else:
        years = sorted(future['Waktu'].dt.year.unique())
        selected_year = st.selectbox("Pilih Tahun", years)
        months = future[future['Waktu'].dt.year == selected_year]
        month_options = {m.month: m.strftime('%B') for m in months['Waktu']}
        selected_month = st.selectbox("Pilih Bulan", month_options.keys(), format_func=lambda m: month_options[m])

        sel_row = months[months['Waktu'].dt.month == selected_month].iloc[0]
        pred_value = sel_row[ [c for c in sel_row.index if 'Forecast' in c or 'forecast' in c][0] ]
        change_abs = pred_value - harga_terakhir
        change_pct = (change_abs / harga_terakhir) * 100 if harga_terakhir != 0 else 0
        trend = trend_arrow(pred_value, harga_terakhir)

        st.markdown(f"""
        <div class='card-glass'>
        <h4 style='color:var(--gold);margin:0;'>Prediksi {model_choice} — {sel_row['Waktu'].strftime('%B %Y')}</h4>
        <div style='font-size:22px;font-weight:900;color:var(--gold);margin-top:8px'>{format_idr(pred_value)}</div>
        <div class='small'>Perubahan dari harga terakhir ({format_idr(harga_terakhir)}): 
        <b>{change_pct:+.2f}%</b> — {trend}</div>
        </div>
        """, unsafe_allow_html=True)

        # grafik pendek (12 bulan terakhir + prediksi)
        st.markdown("### Tren Jangka Pendek (12 bulan terakhir + prediksi)")
        df_short = pd.concat([df_hist.tail(12), future]).drop_duplicates(subset='Waktu', keep='last').sort_values('Waktu')
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(x=df_short['Waktu'], y=df_short['Harga Emas'], name="Aktual", line=dict(color="#f7c948", width=2.5), line_shape="spline"))
        # find forecast column in future
        forecast_col = [c for c in future.columns if 'Forecast' in c or 'forecast' in c]
        if forecast_col:
            fig_live.add_trace(go.Scatter(x=future['Waktu'], y=future[forecast_col[0]], name=f"Prediksi {model_choice}", line=dict(color="#60a5fa", width=2, dash="dot"), line_shape="spline"))
        yax_live = yaxis_range_for(df_short, future)
        fig_live.update_layout(template="plotly_dark", height=380, paper_bgcolor="rgba(0,0,0,0)", yaxis=yax_live)
        st.plotly_chart(fig_live, use_container_width=True)


st.markdown("<hr><div class='small' style='text-align:center'>© 2025 Dashboard Peramalan Harga Emas</div>", unsafe_allow_html=True)


