import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기 (Seoul Temperature Predictor)")
st.write("서울 기온 데이터를 바탕으로 연도별 평균기온을 분석하고, 선형 회귀를 통해 미래 기온을 예측합니다.")

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
    
    # 회귀 분석 (선형 회귀)
    slope, intercept = np.polyfit(data['연도'], data['연평균기온'], 1)
    correlation = data['연도'].corr(data['연평균기온'])
    
    data['회귀예측'] = slope * data['연도'] + intercept
    
    # 메타데이터 추출
    num_years = len(data)
    start_year = int(data['연도'].min())
    end_year = int(data['연도'].max())
    
    # 화면 정보 표시
    st.info(f"📊 **분석 정보** | 직선을 만든 해의 개수: **{num_years}개** | 시작 연도: **{start_year}년** | 끝 연도: **{end_year}년** | 상관계수 ($r$): **{correlation:.3f}**")
    
    # 레이아웃 구성 (좌측: 슬라이더 및 예측 결과, 우측: 그래프)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔮 연도별 기온 예측")
        selected_year = st.slider("연도를 선택하세요", 1900, 2100, 2026)
        
        predicted_temp = slope * selected_year + intercept
        
        st.markdown(f"### 🗓️ 선택 연도: **{selected_year}년**")
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="color: #31333F; margin: 0;">예상 평균기온</h3>
                <h1 style="color: #ff4b4b; margin: 10px 0 0 0;">{predicted_temp:.2f} °C</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.write(f"**회귀식 정보:**")
        st.latex(f"y = {slope:.4f} \\times \\text{{연도}} + {intercept:.2f}")
        st.caption("회귀 직선은 300일 이상 관측된 유효 연도의 데이터를 기반으로 산출되었습니다.")

    with col2:
        st.subheader("📈 연도별 평균기온 산점도 및 회귀 직선")
        
        fig = px.scatter(
            data, x='연도', y='연평균기온',
            labels={'연도': '연도', '연평균기온': '연평균기온 (°C)'},
            title="서울 연도별 평균기온 추이"
        )
        
        # 회귀 직선 추가
        fig.add_trace(
            go.Scatter(
                x=data['연도'],
                y=data['회귀예측'],
                mode='lines',
                name=f'회귀 직선 (y = {slope:.3f}x + {intercept:.1f})',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
