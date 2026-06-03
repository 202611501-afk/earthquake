import streamlit as st
import pandas as pd
import joblib
import folium
from sklearn.preprocessing import StandardScaler

# 모델, 스케일러, 데이터 로드 함수
@st.cache_resource
def load_resources():
    model = joblib.load('earthquake_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df_new = pd.read_csv('earthquake_processed.csv')
    st.write(df_new.columns) # 데이터의 실제 열 이름을 화면에 표시해주는 코드
    return model, scaler, df_new

model, scaler, df_new = load_resources()

# 위험도 딕셔너리
risk_dict = {0: '높음', 1: '낮음', 2: '중간'}
colors = {0: 'red', 1: 'blue', 2: 'green'}

st.set_page_config(page_title="세계 지진 위험도 예측", layout="wide")
st.title("🌍 세계 지진 위험도 예측")

st.markdown("--- ")

st.sidebar.header("위도 및 경도 입력")
lat = st.sidebar.number_input("위도 (Latitude)", min_value=-90.0, max_value=90.0, value=37.0, step=0.1)
lon = st.sidebar.number_input("경도 (Longitude)", min_value=-180.0, max_value=180.0, value=127.0, step=0.1)

st.sidebar.markdown("--- ")

# 예측 로직
# 1. 페이지가 처음 켜질 때 버튼 클릭 여부를 기억할 저장소(session_state)를 만듭니다.
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# 2. 버튼을 누르면 기억 장소의 값을 True(참)로 바꿉니다.
if st.sidebar.button("위험도 예측"):
    st.session_state.clicked = True

# 3. 버튼이 한 번이라도 눌렸었다면 아래 로직을 계속 실행합니다.
if st.session_state.clicked:
    st.subheader(f"📍 입력 위치: 위도 {lat}, 경도 {lon}")
    
    mean_influence = 0.0
    mean_magnitude = 4.0
    
    try:
        mean_depth = pd.to_numeric(df_new['진원깊이'], errors='coerce').mean()
        if pd.isna(mean_depth): mean_depth = 10.0
    except:
        mean_depth = 10.0

    input_data = pd.DataFrame(
        [[mean_influence, mean_magnitude, mean_depth]], 
        columns=['영향도', '규모', '진원깊이']
    )
    
    try:
        # 스케일링 및 예측 진행
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        
        # 위험도 결과 매핑
        risk_label = risk_dict.get(prediction, f"클러스터 {prediction}")
        color = colors.get(prediction, "gray")
        
        # 결과 출력
        st.metric(label="⚠️ 예측된 지진 위험도", value=risk_label)
        
        # 4. 지도 표시 (folium 사용)
        m = folium.Map(location=[lat, lon], zoom_start=5)
        folium.Marker(
            [lat, lon],
            popup=f"위험도: {risk_label}",
            icon=folium.Icon(color=color, icon="info-sign")
        )
        
        # streamlit_folium을 이용해 지도 그리기
        from streamlit_folium import st_folium
        st_folium(m, width=700, height=500)
        
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")