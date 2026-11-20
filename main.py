import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st


# ──────────────────────────────────────────────
#  열 이름 표준화 (学校ごとに違う列名を統一)
# ──────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ──────────────────────────────────────────────
#  Plotly 글자·레이아웃 공통 설정
# ──────────────────────────────────────────────
def plotly_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def bar_chart(df, x_col, y_col, title, color_col=None, top_ids=None, bottom_ids=None):
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        return None
    hover = f"{x_col}: %{{x|%.2f}}<br>{y_col}: %{{y|%.2f}}"
    if color_col is not None and color_col in d.columns:
        colors = d[color_col].map(
            lambda v: "#2563eb"
            if top_ids is not None and str(v) in top_ids
            else "#dc2626"
            if bottom_ids is not None and str(v) in bottom_ids
            else "#94a3b8"
        )
        hover = f"{x_col}: %{{x|%.2f}}<br>{y_col}: %{{y|%.2f}}<br>{color_col}: %{{customdata[0]}}"
        fig = go.Figure(
            data=[
                go.Bar(
                    x=d[x_col],
                    y=d[y_col],
                    customdata=d[color_col].values[:, None],
                    marker_color=colors.tolist(),
                    hovertemplate=hover,
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=d[x_col],
                    y=d[y_col],
                    hovertemplate=hover,
                )
            ]
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=False,
    )
    return plotly_font(fig)


# ──────────────────────────────────────────────
#  데이터 읽기 (エンコード二段読み)
# ──────────────────────────────────────────────
@st.cache_data
def load_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ──────────────────────────────────────────────
#  Streamlit 앱
# ──────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
st.caption("CSV를 업로드하면 자동으로 분석됩니다. 학교명·열 이름은 하드코딩하지 않고 파일 내용만 읽습니다.")

env_file = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌱 생육 결과 데이터 CSV", type="csv")
growth_items = st.sidebar.multiselect(
    "📏 생육 측정 항목 (그래프에 표시할 항목)",
    ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"],
    default=["생중량", "잎 수", "지상부 길이"],
)

try:
    env_df = load_csv(env_file) if env_file is not None else pd.DataFrame()
    growth_df = load_csv(growth_file) if growth_file is not None else pd.DataFrame()
except Exception:
    env_df = pd.DataFrame()
    growth_df = pd.DataFrame()

# 列名標準化
if not env_df.empty:
    env_df = std_cols(env_df)
if not growth_df.empty:
    growth_df = std_cols(growth_df)

school_name = "우리 학교"
if env_file is not None:
    base = os.path.basename(env_file.name)
    school_name = base.split("_")[0].strip() or school_name
elif growth_file is not None:
    base = os.path.basename(growth_file.name)
    school_name = base.split("_")[0].strip() or school_name


# ── Tab 1: 실험 개요 ─────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 실험 개요", "🌡️ 환경 데이터", "🌱 생육 결과"])

with tab1:
    try:
        st.header(f"📋 {school_name} 극지식물 실험 개요")
        st.markdown("""
**연구 배경 및 목적**  
극지식물은 낮은 온도·건조한 환경에서도 살아남는 독특한 적응 전략을 가지고 있습니다.
이 대시보드에서는 우리 학교 실험에서 키운 극지식물 **개체마다 생육 결과가 얼마나 다른지**를 수치로 비교합니다.
평균·표준편차·변동계수(CV)·상관계수 등을 통해 "어떤 개체가 잘 자랐는지", "잎 수나 줄기 길이와 전체 무게는 얼마나 관련 있는지"를 살펴봅니다.

- **평균(mean)**: 여러 값을 더해 개수로 나눈 값. 대표값입니다.
- **표준편차(standard deviation)**: 값이 평균에서 얼마나 흩어져 있는지를 나타냅니다. 클수록 개체 간 차이가 큽니다.
- **변동계수(CV, coefficient of variation)**: 표준편차를 평균으로 나눈 값. 단위가 다른 항목끼리도 상대적 흩어짐을 비교할 수 있습니다.
- **상관계수(Pearson correlation)**: 두 변수가 함께 변하는 정도. +1에 가까우면 함께 증가, -1에 가까우면 한쪽이 늘 때 다른 쪽이 줄어듭니다.
""")

        # 데이터 요약 표
        total_growth = len(growth_df) if not growth_df.empty else 0
        total_env = len(env_df) if not env_df.empty else 0

        env_cols_avail = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
        growth_cols_avail = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]

        rows = []
        rows.append([school_name, total_growth, total_env, ", ".join(growth_cols_avail) if growth_cols_avail else "(없음)"])
        summary_tbl = pd.DataFrame(
            rows,
            columns=["학교명", "개체 수", "환경 데이터 행 수", "생육 측정 항목"],
        )
        st.subheader("📌 데이터 요약")
        st.dataframe(summary_tbl, width="stretch")

        # 평균 온도·습도 계산
        avg_temp = float(env_df["temperature"].dropna().mean()) if "temperature" in env_df.columns else 0.0
        avg_hum = float(env_df["humidity"].dropna().mean()) if "humidity" in env_df.columns else 0.0
        avg_fw = float(growth_df["생중량"].dropna().mean()) if "생중량" in growth_df.columns else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌱 총 개체 수", f"{total_growth} 개체")
        c2.metric("🌡️ 평균 온도", f"{avg_temp:.1f} °C")
        c3.metric("💧 평균 습도", f"{avg_hum:.1f} %")
        c4.metric("⚖️ 평균 생중량", f"{avg_fw:.1f} g")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── Tab 2: 환경 데이터 ──────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 사이드바에서 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터")

            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_num_cols:
                st.subheader("📊 환경 요약 (평균 · 최소 · 최대)")
                stats_dict = {}
                for c in env_num_cols:
                    s = env_df[c].dropna()
                    if len(s) > 0:
                        stats_dict[c] = [s.mean(), s.min(), s.max()]
                if stats_dict:
                    env_summary = pd.DataFrame(
                        stats_dict,
                        index=["평균", "최소", "최대"],
                    )
                    st.dataframe(env_summary, width="stretch")
                else:
                    st.info("환경 데이터에서 숫자로 읽을 수 있는 열이 없습니다.")
            else:
                st.info("환경 데이터에 temperature · humidity · ph · ec 열이 없습니다. 다른 형식의 CSV일 수 있습니다.")

            # 시계열 (일별 평균)
            env_dates = env_df.get("time")
            if env_dates is not None and pd.api.types.is_datetime64_any_dtype(env_dates) and env_dates.notna().sum() > 0:
                env_tmp = env_df.dropna(subset=["time"]).copy()
                env_tmp["date"] = env_tmp["time"].dt.date
                has_t = "temperature" in env_tmp.columns and env_tmp["temperature"].dropna().shape[0] >= 1
                has_h = "humidity" in env_tmp.columns and env_tmp["humidity"].dropna().shape[0] >= 1
                has_p = "ph" in env_tmp.columns and env_tmp["ph"].dropna().shape[0] >= 1
                has_e = "ec" in env_tmp.columns and env_tmp["ec"].dropna().shape[0] >= 1
                n_panels = sum([has_t, has_h, has_p, has_e])
                if n_panels > 0:
                    st.subheader("📈 날짜별 평균 변화 (일별)")
                    n_rows = 1 if n_panels <= 2 else 2
                    n_cols = 2 if n_panels >= 2 else 1
                    subplots_fig = make_subplots(rows=n_rows, cols=n_cols)
                    panel_idx = 0
                    panels = [(has_t, "temperature", "🌡️ 온도 (°C)"),
                              (has_h, "humidity", "💧 습도 (%)"),
                              (has_p, "ph", "🧪 pH"),
                              (has_e, "ec", "⚡ EC (mS/cm)")]
                    for available, col_name, label in panels:
                        if not available:
                            continue
                        panel_idx += 1
                        row_n = (panel_idx - 1) // n_cols + 1
                        col_n = (panel_idx - 1) % n_cols + 1
                        daily = env_tmp.groupby("date")[col_name].mean().dropna()
                        if len(daily) > 0:
                            subplots_fig.add_trace(
                                go.Scatter(
                                    x=list(daily.index),
                                    y=daily.values,
                                    mode="lines+markers",
                                    name=label,
                                    line=dict(width=2),
                                ),
                                row=row_n,
                                col=col_n,
                            )
                    subplots_fig.update_xaxes(title_text="날짜")
                    subplots_fig.update_yaxes(title_text="값")
                    subplots_fig.update_layout(height=300 + 150 * (n_rows - 1), showlegend=True)
                    st.plotly_chart(plotly_font(subplots_fig), width="stretch")
                else:
                    st.info("환경 데이터에 숫자로 읽을 수 있는 시계열 열이 없습니다.")
            else:
                st.info("환경 데이터에 시간(time) 열이 없거나 날짜로 읽히지 않았습니다.")

            with st.expander("📋 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 환경 데이터 다운로드 (UTF-8-SIG)",
                    data=csv_bytes,
                    file_name="환경데이터_다운로드.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ── Tab 3: 생육 결과 ────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("🌱 생육 결과 CSV를 사이드바에서 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header(f"🌱 {school_name} 생육 결과")

            # ── 핵심 결과 카드 ──
            fw_valid = growth_df["생중량"].dropna() if "생중량" in growth_df.columns else pd.Series(dtype=float)
            if len(fw_valid) == 0:
                st.info("생중량(지상부 질량 + 지하부 질량으로 계산된 값 포함)을 찾을 수 없습니다.")
            else:
                st.subheader("🥇 핵심 결과")
                avg_fw_val = float(fw_valid.mean())
                max_idx = fw_valid.idxmax()
                min_idx = fw_valid.idxmin()
                max_row = growth_df.loc[max_idx] if max_idx in growth_df.index else None
                min_row = growth_df.loc[min_idx] if min_idx in growth_df.index else None
                max_id = max_row.get("개체번호", "?") if max_row is not None else "?"
                min_id = min_row.get("개체번호", "?") if min_row is not None else "?"
                max_val = float(max_row["생중량"]) if max_row is not None and "생중량" in max_row else 0.0
                min_val = float(min_row["생중량"]) if min_row is not None and "생중량" in min_row else 0.0
                k1, k2, k3 = st.columns(3)
                k1.metric("📊 평균 생중량", f"{avg_fw_val:.1f} g")
                k2.metric("🏆 최대 생중량", f"{max_id} 번 — {max_val:.1f} g")
                k3.metric("📉 최소 생중량", f"{min_id} 번 — {min_val:.1f} g")

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📊 개체별 생육 비교")
            growth_x = growth_df.get("개체번호")
            if growth_x is not None and len(growth_x) > 0:
                for item in growth_items:
                    if item not in growth_df.columns:
                        continue
                    d = growth_df.dropna(subset=["개체번호", item]).copy()
                    if len(d) < 1:
                        continue
                    fig = bar_chart(
                        d,
                        x_col="개체번호",
                        y_col=item,
                        title=f"개체별 {item}",
                        color_col="개체번호",
                    )
                    if fig is not None:
                        st.plotly_chart(plotly_font(fig), width="stretch")
            else:
                st.info("개체번호 열이 없어 개체별 그래프를 그리지 못합니다.")

            # ── 생중량 분포 ──
            if "생중량" in growth_df.columns:
                fw_col = growth_df["생중량"].dropna()
                if len(fw_col) >= 2:
                    st.subheader("📦 생중량 분포")
                    violin_fig = go.Figure(
                        data=[
                            go.Violin(
                                y=fw_col.values,
                                box_visible=True,
                                fillcolor="#a78bfa",
                                line_color="#7c3aed",
                                name="생중량",
                            )
                        ]
                    )
                    violin_fig.update_layout(
                        title="생중량 분포 (박스 + 바이올린)",
                        yaxis_title="생중량 (g)",
                        xaxis_title="",
                        showlegend=False,
                    )
                    st.plotly_chart(plotly_font(violin_fig), width="stretch")
                else:
                    st.info("생중량 데이터가 2개 미만이라 분포를 표시하지 못합니다.")

            # ── 상관관계 분석 ──
            st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")
            if "생중량" in growth_df.columns:
                for item in ["잎 수", "지상부 길이"]:
                    if item not in growth_df.columns:
                        continue
                    corr_df = growth_df[[item, "생중량"]].dropna()
                    if len(corr_df) < 2:
                        st.info(f"{item}과 생중량 데이터가 부족하여 상관계수를 계산하지 못합니다.")
                        continue
                    x_vals = corr_df[item].values
                    y_vals = corr_df["생중량"].values
                    r, _ = stats.pearsonr(x_vals, y_vals)
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                    y_line = slope * x_line + intercept
                    fig = make_subplots(rows=1, cols=1)
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode="markers",
                            marker=dict(size=9, color="#60a5fa", opacity=0.8),
                            name="관측값",
                            hovertemplate=f"{item}: %{{x|%.2f}}<br>생중량: %{{y|%.2f}}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="#ef4444", width=2, dash="dot"),
                            name=f"추세선 (r={r:.2f})",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.update_xaxes(title_text=item)
                    fig.update_yaxes(title_text="생중량 (g)")
                    fig.update_layout(
                        title=f"{item} vs 생중량 (상관계수 r = {r:.2f})",
                        showlegend=True,
                        legend=dict(x=0.02, y=0.98),
                    )
                    st.plotly_chart(plotly_font(fig), width="stretch")
            else:
                st.info("생중량 열이 없어 상관관계 분석을 하지 못합니다.")

            # ── 상위 25% vs 하위 25% 비교 ──
            if "생중량" in growth_df.columns and len(fw_valid) >= 4:
                q75 = fw_valid.quantile(0.75)
                q25 = fw_valid.quantile(0.25)
                top_df = growth_df[fw_valid >= q75].dropna(subset=["생중량"])
                bottom_df = growth_df[fw_valid <= q25].dropna(subset=["생중량"])
                top_ids_set = set(top_df["개체번호"].astype(str)) if "개체번호" in top_df.columns else set()
                bottom_ids_set = set(bottom_df["개체번호"].astype(str)) if "개체번호" in bottom_df.columns else set()
                compare_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns and c in top_df.columns and c in bottom_df.columns]
                if compare_cols and len(top_df) >= 2 and len(bottom_df) >= 2:
                    st.subheader("🏅 생중량 상위 25% vs 하위 25% 비교")
                    rows = []
                    for c in compare_cols:
                        t = top_df[c].dropna()
                        b = bottom_df[c].dropna()
                        if len(t) > 0 and len(b) > 0:
                            rows.append([
                                c,
                                f"{t.mean():.2f} ± {t.std():.2f}",
                                f"{b.mean():.2f} ± {b.std():.2f}",
                                f"{(t.mean() - b.mean()):.2f}",
                            ])
                    if rows:
                        comp_tbl = pd.DataFrame(
                            rows,
                            columns=["측정 항목", "상위 25% (평균 ± SD)", "하위 25% (평균 ± SD)", "차이 (상위 − 하위)"],
                        )
                        st.dataframe(comp_tbl, width="stretch")
                    else:
                        st.info("비교할 측정 항목이 충분하지 않습니다.")
                elif len(top_df) < 2 or len(bottom_df) < 2:
                    st.info("상위·하위 25% 그룹이 각각 2개 미만이라 비교를 표시하지 못합니다.")

            with st.expander("📋 생육 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 생육 데이터 다운로드 (UTF-8-SIG)",
                    data=csv_bytes,
                    file_name="생육결과_다운로드.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

st.caption("※ 분석은 업로드된 CSV 데이터 기반으로 자동 계산되며, 학교명이나 열 이름을 하드코딩하지 않습니다.")
