import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

st.set_page_config(page_title="NBA Clutch Dashboard", layout="wide")
st.title("🏀 NBA Clutch Score Dashboard (Linear Regression)")

st.markdown("""
Upload **two CSVs** with these columns (player-level clutch splits):
`PLAYER_NAME, FG_PCT, PTS, AST, TOV, STL, OREB, DREB, BLK, FG3M, GP, clutch_score`
""")

col_u1, col_u2 = st.columns(2)
with col_u1:
    f23 = st.file_uploader("Upload 2023–24 CSV", type="csv", key="csv_23")
with col_u2:
    f24 = st.file_uploader("Upload 2024–25 CSV", type="csv", key="csv_24")

run = st.button("Run Model & Build Dashboard")

def _validate_columns(df, season_label):
    required = {'PLAYER_NAME','FG_PCT','PTS','AST','TOV','STL','OREB','DREB','BLK','FG3M','GP','clutch_score'}
    missing = required - set(df.columns)
    if missing:
        st.error(f"{season_label}: missing columns: {sorted(list(missing))}")
        return False
    return True

if run:
    if (f23 is None) or (f24 is None):
        st.error("Please upload both CSVs first.")
        st.stop()

    df23 = pd.read_csv(f23)
    df24 = pd.read_csv(f24)
    if not _validate_columns(df23, "2023–24"): st.stop()
    if not _validate_columns(df24, "2024–25"): st.stop()

    # Merge by player to create training pairs: 23–24 features -> 24–25 clutch_score
    feats = ['FG_PCT','PTS','AST','TOV','STL','OREB','DREB','BLK','FG3M','GP']
    df_merged = pd.merge(
        df23[['PLAYER_NAME'] + feats + ['clutch_score']].rename(columns={'clutch_score':'clutch_score_23_24'}),
        df24[['PLAYER_NAME','clutch_score']].rename(columns={'clutch_score':'clutch_score_24_25'}),
        on='PLAYER_NAME',
        how='inner'
    )

    if df_merged.empty:
        st.error("No overlapping players between the two files. Check names/IDs.")
        st.stop()

    # Train linear regression on 23–24 features → 24–25 clutch
    X = df_merged[feats]
    y = df_merged['clutch_score_24_25']
    model = LinearRegression().fit(X, y)

    # Predict 24–25 from 23–24 features (predictive validity check)
    y_pred_24_25 = model.predict(X)

    # Predict 25–26 from 24–25 features (final forecast)
    X_future = df24[feats]
    pred_25_26 = model.predict(X_future)

    # ---- Metrics (Predictive Validity 23→24) ----
    r2 = r2_score(y, y_pred_24_25)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred_24_25)))
    rho, _ = spearmanr(y, y_pred_24_25)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² (23→24)", f"{r2:.3f}")
    m2.metric("RMSE (23→24)", f"{rmse:.3f}")
    m3.metric("Spearman ρ (23→24)", f"{rho:.3f}")

    # ---- Scatter: Actual vs Predicted (24–25) ----
    st.subheader("Predictive Validity: Actual vs Predicted (24–25)")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y, y_pred_24_25)
    mn, mx = float(min(y.min(), y_pred_24_25.min())), float(max(y.max(), y_pred_24_25.max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("Actual 24–25 Clutch")
    ax.set_ylabel("Predicted 24–25 Clutch")
    st.pyplot(fig)

    # ---- Leaderboards ----
    st.subheader("Leaderboards")

    rank23 = df23[['PLAYER_NAME','clutch_score']].copy()
    rank23['Rank_23_24'] = rank23['clutch_score'].rank(ascending=False, method='dense').astype(int)
    rank23 = rank23.sort_values('Rank_23_24').rename(columns={'clutch_score':'Clutch_23_24'})

    rank24 = df24[['PLAYER_NAME','clutch_score']].copy()
    rank24['Rank_24_25'] = rank24['clutch_score'].rank(ascending=False, method='dense').astype(int)
    rank24 = rank24.sort_values('Rank_24_25').rename(columns={'clutch_score':'Clutch_24_25'})

    forecast = df24[['PLAYER_NAME']].copy()
    forecast['Actual_24_25'] = df24['clutch_score'].values
    forecast['Predicted_25_26'] = pred_25_26
    forecast['Delta_vs_24_25'] = forecast['Predicted_25_26'] - forecast['Actual_24_25']
    pred25 = forecast[['PLAYER_NAME','Predicted_25_26']].copy()
    pred25['Rank_25_26_Pred'] = pred25['Predicted_25_26'].rank(ascending=False, method='dense').astype(int)
    pred25 = pred25.sort_values('Rank_25_26_Pred')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Actual 2023–24 (Top 15)**")
        st.dataframe(rank23.head(15), use_container_width=True)
    with c2:
        st.markdown("**Actual 2024–25 (Top 15)**")
        st.dataframe(rank24.head(15), use_container_width=True)
    with c3:
        st.markdown("**Predicted 2025–26 (Top 15)**")
        st.dataframe(pred25.head(15), use_container_width=True)

    # ---- Rank comparison ----
    st.subheader("Rank Comparison & Movement")
    compare = (
        rank23[['PLAYER_NAME','Clutch_23_24','Rank_23_24']]
        .merge(rank24[['PLAYER_NAME','Clutch_24_25','Rank_24_25']], on='PLAYER_NAME', how='outer')
        .merge(pred25[['PLAYER_NAME','Predicted_25_26','Rank_25_26_Pred']], on='PLAYER_NAME', how='outer')
    )
    compare['Rank_Delta_23_to_24'] = compare['Rank_23_24'] - compare['Rank_24_25']
    compare['Rank_Delta_24_to_25pred'] = compare['Rank_24_25'] - compare['Rank_25_26_Pred']
    compare = compare.sort_values('Rank_25_26_Pred', na_position='last')

    st.dataframe(compare, use_container_width=True)

    # ---- Biggest risers / fallers (by Delta vs 24–25) ----
    st.subheader("Projected Movers (25–26 vs 24–25)")
    movers_left, movers_right = st.columns(2)
    risers = forecast.sort_values('Delta_vs_24_25', ascending=False).head(10)
    fallers = forecast.sort_values('Delta_vs_24_25', ascending=True).head(10)
    with movers_left:
        st.markdown("**Top 10 Risers**")
        st.dataframe(risers[['PLAYER_NAME','Actual_24_25','Predicted_25_26','Delta_vs_24_25']], use_container_width=True)
    with movers_right:
        st.markdown("**Top 10 Fallers**")
        st.dataframe(fallers[['PLAYER_NAME','Actual_24_25','Predicted_25_26','Delta_vs_24_25']], use_container_width=True)

    # ---- Multi-season grouped bars for consistent top players ----
    st.subheader("Progression: 23–24 (Actual) → 24–25 (Actual & Pred) → 25–26 (Pred)")
    top_names = (
        df_merged[['PLAYER_NAME','clutch_score_24_25']]
        .sort_values('clutch_score_24_25', ascending=False)
        .head(12)['PLAYER_NAME'].tolist()
    )
    plot_df = (
        df_merged.set_index('PLAYER_NAME')
        .loc[top_names, ['clutch_score_23_24','clutch_score_24_25']]
        .reset_index()
        .merge(forecast[['PLAYER_NAME','Predicted_25_26']], on='PLAYER_NAME', how='left')
        .merge(pd.DataFrame({'PLAYER_NAME': df_merged['PLAYER_NAME'], 'Pred_24_25': y_pred_24_25}),
               on='PLAYER_NAME', how='left')
    )

    x = np.arange(len(plot_df))
    width = 0.2
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.bar(x - 0.3, plot_df['clutch_score_23_24'], width, label='Actual 23–24')
    ax2.bar(x - 0.1, plot_df['clutch_score_24_25'], width, label='Actual 24–25')
    ax2.bar(x + 0.1, plot_df['Pred_24_25'], width, label='Pred 24–25')
    ax2.bar(x + 0.3, plot_df['Predicted_25_26'], width, label='Pred 25–26')
    ax2.set_xticks(x)
    ax2.set_xticklabels(plot_df['PLAYER_NAME'], rotation=45, ha='right')
    ax2.set_ylabel("Clutch Score")
    ax2.legend()
    st.pyplot(fig2)

    # ---- Download predicted 25–26 leaderboard ----
    st.subheader("Download Predicted 2025–26 Leaderboard")
    out = pred25[['PLAYER_NAME','Predicted_25_26','Rank_25_26_Pred']].copy()
    st.download_button(
        "Download CSV",
        data=out.to_csv(index=False).encode('utf-8'),
        file_name="Predicted_Clutch_2025_26.csv",
        mime="text/csv"
    )
