import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기 및 비교 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기 및 기간별 기울기 비교")
st.write("서울 기온 데이터를 바탕으로 연도별 평균기온을 분석하고, 전체 기간과 최근 20년의 기온 상승 속도를 비교합니다.")

@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding='utf-8')
    
    # 열 이름 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 형식 변환 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 기준 기간: 2025년까지 필터링
    df = df[df['연도'] <= 2025]
    
    # 평균기온 결측치 제거
    df = df.dropna(subset=['평균기온'])
    
    # 연도별 관측일수 및 연평균기온 계산
    yearly_counts = df.groupby('연도')['평균기온'].count()
    yearly_means = df.groupby('연도')['평균기온'].mean()
    
    # 관측일이 300일 이상인 해만 선택
    valid_years = yearly_counts[yearly_counts >= 300].index
    filtered_means = yearly_means[yearly_means.index.isin(valid_years)]
    
    df_plot = pd.DataFrame({
        '연도': filtered_means.index,
        '연평균기온': filtered_means.values
    }).reset_index(drop=True)
    
    return df_plot

try:
    data = load_and_preprocess_data()
    
    # 1. 전체 기간 선형 회귀 분석
    slope_all, intercept_all = np.polyfit(data['연도'], data['연평균기온'], 1)
    correlation_all = data['연도'].corr(data['연평균기온'])
    
    data['회귀예측_전체'] = slope_all * data['연도'] + intercept_all
    
    num_years = len(data)
    start_year = int(data['연도'].min())
    end_year = int(data['연도'].max())
    
    # 100년당 상승 온도 (전체)
    rise_100_all = slope_all * 100
    
    # 2. 최근 20년 선형 회귀 분석 (예: end_year - 19 ~ end_year)
    recent_start_year = max(start_year, end_year - 19)
    df_recent = data[data['연도'] >= recent_start_year].copy()
    
    slope_recent, intercept_recent = np.polyfit(df_recent['연도'], df_recent['연평균기온'], 1)
    correlation_recent = df_recent['연도'].corr(df_recent['연평균기온'])
    df_recent['회귀예측_최근'] = slope_recent * df_recent['연도'] + intercept_recent
    
    # 100년당 상승 온도 (최근 20년)
    rise_100_recent = slope_recent * 100
    
    # 화면 정보 표시 (전체 기준)
    st.info(f"📊 **기본 분석 정보** | 분석된 유효 연도 수: **{num_years}개** ({start_year}년 ~ {end_year}년) | 전체 상관계수 ($r$): **{correlation_all:.3f}**")
    
    # --- 새로운 요구사항: 전체 기간 vs 최근 20년 기울기(100년당 상승 온도) 비교 카드 ---
    st.subheader("📈 기간별 기온 상승 속도 비교 (100년 기준)")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #31333F; text-align: center;">
                <h4 style="margin: 0; color: #31333F;">전체 기간 ({start_year}~{end_year}년)</h4>
                <p style="margin: 5px 0; font-size: 14px; color: #666;">1년당 {slope_all:+.4f} °C 변화</p>
                <h2 style="color: #2b5c8f; margin: 10px 0 0 0;">100년당 {rise_100_all:+.2f} °C</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with comp_col2:
        st.markdown(
            f"""
            <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107; text-align: center;">
                <h4 style="margin: 0; color: #856404;">최근 20년 ({recent_start_year}~{end_year}년)</h4>
                <p style="margin: 5px 0; font-size: 14px; color: #856404;">1년당 {slope_recent:+.4f} °C 변화</p>
                <h2 style="color: #d39e00; margin: 10px 0 0 0;">100년당 {rise_100_recent:+.2f} °C</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # 레이아웃 구성 (좌측: 슬라이더 및 예측 결과, 우측: 그래프)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔮 연도별 기온 예측")
        selected_year = st.slider("연도를 선택하세요", 1900, 2100, 2026)
        
        # 전체 회귀선 기준 예측값 계산
        predicted_temp = slope_all * selected_year + intercept_all
        
        st.markdown(f"### 🗓️ 선택 연도: **{selected_year}년**")
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #31333F; margin: 0;">예상 평균기온 (전체 회귀식 기준)</h3>
                <h1 style="color: #ff4b4b; margin: 10px 0 0 0;">{predicted_temp:.2f} °C</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.write(f"**전체 회귀식 정보:**")
        st.latex(f"y = {slope_all:.4f} \\times \\text{{연도}} + {intercept_all:.2f}")
        st.caption(f"이 회귀선 기준, 서울은 100년마다 약 **{rise_100_all:.2f}°C** 상승하는 추세를 보입니다.")

    with col2:
        st.subheader("📈 연도별 평균기온 산점도 및 회귀 직선 비교")
        
        fig = px.scatter(
            data, x='연도', y='연평균기온',
            labels={'연도': '연도', '연평균기온': '연평균기온 (°C)'},
            title="서울 연도별 평균기온 추이 및 회귀선"
        )
        
        # 전체 회귀 직선 추가
        fig.add_trace(
            go.Scatter(
                x=data['연도'],
                y=data['회귀예측_전체'],
                mode='lines',
                name=f'전체 회귀선 (100년당 {rise_100_all:+.2f}°C)',
                line=dict(color='blue', width=2)
            )
        )
        
        # 최근 20년 회귀 직선 추가
        fig.add_trace(
            go.Scatter(
                x=df_recent['연도'],
                y=df_recent['회귀예측_최근'],
                mode='lines',
                name=f'최근 20년 회귀선 (100년당 {rise_100_recent:+.2f}°C)',
                line=dict(color='orange', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
