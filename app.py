import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time, io, os, random
from openai import OpenAI

st.set_page_config(page_title="Mind-Link AI | 학생마음바우처", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ── OpenAI 클라이언트 초기화 ──────────────────────────────────────────────────
# .env 또는 Streamlit secrets에서 API 키를 읽습니다.
# 실행 전 환경변수 OPENAI_API_KEY 를 설정하거나 .streamlit/secrets.toml 에 입력하세요.
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        return None

    return OpenAI(api_key=api_key)

# ── LLM 호출 공통 함수 ────────────────────────────────────────────────────────
def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    """GPT-4o 호출. API 키 없으면 None 반환 → 호출부에서 fallback 처리."""
    client = get_openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT 오류: {e}")
        return None

# ── 비밀번호 설정 ─────────────────────────────────────────────────────────────
SCHOOL_PASSWORDS = {
    "가온초등학교": "1234", "나래중학교": "2345", "다솜고등학교": "3456",
    "라온초등학교": "4567", "마루중학교": "5678", "바른고등학교": "6789",
    "사임당초등학교": "7890", "아라중학교": "8901", "자람초등학교": "9012",
    "차오름중학교": "0123", "타래고등학교": "1111", "파란초등학교": "2222",
    "하늘중학교": "3333", "해랑고등학교": "4444",
}
PARENT_PASSWORD = "jiwon2026"
HOSPITAL_PASSWORDS = {
    "아이꿈상담소": "hosp01", "마음드림심리상담소": "hosp02",
    "연세정신건강의학과의원": "hosp03", "한국아동청소년발달센터": "hosp04",
    "아이사랑상담센터": "hosp05", "행복한의원": "hosp06",
    "미래심리치료소": "hosp07", "푸른숲정신과": "hosp08",
}
WEE_PASSWORD = "wee2026"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
:root{
  --primary:#1B4F9B;--primary-light:#2E6FC4;--primary-dark:#0D2E5C;
  --accent:#00A878;--accent-light:#00C896;
  --warning:#E85D04;--danger:#D62828;
  --surface:#F0F4FA;--card:#FFFFFF;
  --text:#1A2332;--text-muted:#5A6A7E;--border:#D1DCEC;
  --shadow:0 2px 12px rgba(27,79,155,0.08);--shadow-md:0 4px 24px rgba(27,79,155,0.14);
  --radius:12px;--radius-sm:8px;
}
*{font-family:'Noto Sans KR',sans-serif;}
.stApp{background:var(--surface);}
[data-testid="stSidebar"]{background:var(--primary-dark)!important;}
[data-testid="stSidebar"] *{color:#E8EFF8!important;}
[data-testid="stSidebar"] label{color:#A8BDD4!important;font-size:0.73rem;letter-spacing:0.08em;text-transform:uppercase;}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] .stSelectbox div[class*="singleValue"],
.stSelectbox div[class*="singleValue"]{color:#1A2332 !important; font-weight:600;}
div[data-baseweb="select"] div[class*="singleValue"]{color:#1A2332 !important; font-weight:600;}
div[data-baseweb="select"] input{color:#1A2332 !important;}
.ml-card{background:var(--card);border-radius:var(--radius);padding:1.2rem 1.4rem;box-shadow:var(--shadow);border:1px solid var(--border);margin-bottom:0.9rem;}
.ml-card-accent{border-left:4px solid var(--accent);}
.ml-card-danger{border-left:4px solid var(--danger);background:#FFF5F5;}
.ml-card-warn{border-left:4px solid var(--warning);background:#FFF8F4;}
.ml-card-primary{border-left:4px solid var(--primary);}
.ml-alert{background:linear-gradient(135deg,#D62828,#E85D04);color:white!important;border-radius:var(--radius);padding:1rem 1.4rem;margin-bottom:0.7rem;box-shadow:var(--shadow-md);}
.ml-alert-title{font-weight:700;font-size:0.88rem;margin-bottom:0.2rem;}
.ml-alert-body{font-size:0.78rem;opacity:0.92;}
.ml-brief{background:linear-gradient(135deg,var(--primary-dark),var(--primary));color:white;border-radius:var(--radius);padding:1.3rem 1.5rem;margin-bottom:0.9rem;box-shadow:var(--shadow-md);}
.ml-brief-label{font-size:0.67rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--accent-light);font-weight:700;margin-bottom:0.5rem;}
.ml-brief-line{font-size:0.85rem;line-height:1.9;color:rgba(255,255,255,0.92);}
.ml-brief-line::before{content:'▸ ';color:var(--accent-light);}
.ml-brief-source{font-size:0.72rem;color:rgba(255,255,255,0.5);margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid rgba(255,255,255,0.1);}
.ml-metric{background:var(--card);border-radius:var(--radius);padding:1.1rem 0.9rem;box-shadow:var(--shadow);border:1px solid var(--border);text-align:center;height:100%;min-height:110px;display:flex;flex-direction:column;justify-content:center;}
.ml-metric-icon{font-size:1.7rem;margin-bottom:0.2rem;}
.ml-metric-value{font-size:1.7rem;font-weight:900;color:var(--primary);line-height:1.1;}
.ml-metric-label{font-size:0.71rem;color:var(--text-muted);margin-top:0.2rem;font-weight:500;}
.ml-metric-sub{font-size:0.66rem;color:var(--accent);margin-top:0.12rem;font-weight:600;}
.ml-sh{font-size:0.67rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--text-muted);font-weight:700;border-bottom:1px solid var(--border);padding-bottom:0.45rem;margin-bottom:0.9rem;margin-top:1.3rem;}
.ml-logo{font-size:1.25rem;font-weight:900;color:white;letter-spacing:-0.02em;margin-bottom:0.15rem;}
.ml-logo span{color:#00C896;}
.ml-logo-sub{font-size:0.6rem;color:#7FA8CC;letter-spacing:0.1em;text-transform:uppercase;}
.ml-tag{display:inline-block;padding:0.13rem 0.5rem;border-radius:20px;font-size:0.67rem;font-weight:600;}
.tag-green{background:#E6F9F5;color:var(--accent);}
.tag-orange{background:#FFF0E5;color:var(--warning);}
.tag-red{background:#FFE8E8;color:var(--danger);}
.tag-blue{background:#E8F0FA;color:var(--primary);}
.tag-gray{background:#F0F2F5;color:var(--text-muted);}
.tag-purple{background:#F0EAFA;color:#6B4FA6;}
.tag-teal{background:#E0FAF5;color:#007A5A;}
.divider{height:1px;background:var(--border);margin:0.9rem 0;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.4rem;padding-bottom:2rem;}
.stButton>button{background:var(--primary)!important;color:white!important;border:none!important;border-radius:var(--radius-sm)!important;font-weight:600!important;font-size:0.81rem!important;padding:0.45rem 1.2rem!important;}
.stButton>button:hover{background:var(--primary-light)!important;transform:translateY(-1px)!important;box-shadow:0 4px 12px rgba(27,79,155,0.25)!important;}
.lock-box{background:white;border-radius:var(--radius);padding:2.5rem 2rem;box-shadow:var(--shadow-md);border:1px solid var(--border);max-width:420px;margin:3rem auto;text-align:center;}
.lock-icon{font-size:3rem;margin-bottom:1rem;}
.lock-title{font-size:1.1rem;font-weight:900;color:var(--primary);margin-bottom:0.3rem;}
.lock-sub{font-size:0.8rem;color:var(--text-muted);margin-bottom:1.5rem;}
.data-table{width:100%;border-collapse:collapse;font-size:0.8rem;}
.data-table th{background:#1B4F9B;color:white;padding:0.5rem 0.7rem;text-align:center;font-weight:600;font-size:0.75rem;}
.data-table td{padding:0.45rem 0.7rem;border-bottom:1px solid #E8F0FA;text-align:center;color:#1A2332;}
.data-table tr:nth-child(even){background:#F5F8FF;}
.data-table tr:hover{background:#EBF3FF;}
.ai-box{background:transparent;border:2.5px solid #1B4F9B;border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:0.9rem;}
.ai-box-green{background:transparent;border:2.5px solid #00A878;border-radius:12px;padding:1.2rem 1.4rem;}
.ai-box-red{background:transparent;border:2.5px solid #D62828;border-radius:12px;padding:1.2rem 1.4rem;}
.ai-section-title{font-size:1.25rem;font-weight:900;color:#1B4F9B;letter-spacing:-0.01em;margin-bottom:0.25rem;}
.ai-section-sub{font-size:0.8rem;color:#5A6A7E;}
.ai-gradient-bar{height:3px;background:linear-gradient(90deg,#1B4F9B,#00A878);border-radius:2px;margin-top:0.5rem;margin-bottom:1.2rem;}
.ai-label{font-size:0.67rem;letter-spacing:0.12em;text-transform:uppercase;font-weight:700;margin-bottom:0.7rem;}
.ai-line{font-size:0.85rem;color:#1A2332;line-height:1.9;padding:0.1rem 0;}
.ai-source{font-size:0.7rem;color:#5A6A7E;margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid #D1DCEC;}
.api-badge{display:inline-flex;align-items:center;gap:0.3rem;background:#E6F9F5;color:#00A878;border-radius:6px;padding:0.2rem 0.6rem;font-size:0.68rem;font-weight:700;margin-left:0.5rem;}
.fallback-badge{display:inline-flex;align-items:center;gap:0.3rem;background:#FFF0E5;color:#E85D04;border-radius:6px;padding:0.2rem 0.6rem;font-size:0.68rem;font-weight:700;margin-left:0.5rem;}
</style>
""", unsafe_allow_html=True)

# ── 상수 ──────────────────────────────────────────────────────────────────────
ACTIVE_STATUS = ['진행중', '종결', '병원종결']

# ── 위(Wee)센터 데이터 ─────────────────────────────────────────────────────────
WEE_CENTERS = [
    {"지역":"수원","유형":"일반","이름":"수원 Wee센터","주소":"수원시 장안구 연무로 8 창용중학교 별관 3층","위도":37.2910,"경도":127.0095,"전화":"031-246-0818"},
    {"지역":"성남","유형":"일반","이름":"성남 Wee센터","주소":"성남시 수정구 남문로 112 금빛초등학교 신관 1층","위도":37.4449,"경도":127.1388,"전화":"031-759-8710"},
    {"지역":"안양과천","유형":"일반","이름":"안양과천 Wee센터","주소":"안양시 동안구 관평로 210 안양과천교육지원청 5층","위도":37.3910,"경도":126.9560,"전화":"031-380-7063"},
    {"지역":"부천","유형":"일반","이름":"부천 Wee센터","주소":"경기도 부천시 원미구 계남로 219 부천교육지원청 4~5층","위도":37.5034,"경도":126.7660,"전화":"032-620-0149"},
    {"지역":"광명","유형":"일반","이름":"광명 Wee센터","주소":"광명시 광명로 777 광명교육지원청 별관 1층","위도":37.4786,"경도":126.8642,"전화":"02-2610-0550"},
    {"지역":"안산","유형":"일반","이름":"안산 Wee센터","주소":"안산시 상록구 석호공원로 5길 8 안산교육지원청 동행관 2층","위도":37.3236,"경도":126.8205,"전화":"031-8085-3960"},
    {"지역":"평택","유형":"일반","이름":"평택 Wee센터","주소":"평택시 장당길 40 장당중학교 1층","위도":36.9926,"경도":127.0950,"전화":"031-650-1680"},
    {"지역":"군포의왕","유형":"일반","이름":"군포의왕 Wee센터","주소":"군포시 청백리길 17 군포의왕교육지원청 2~3층","위도":37.3616,"경도":126.9352,"전화":"031-390-1180"},
    {"지역":"여주","유형":"일반","이름":"여주 Wee센터","주소":"여주시 여양로 103 여주교육지원청","위도":37.2982,"경도":127.6369,"전화":"031-885-0093"},
    {"지역":"화성오산","유형":"일반","이름":"화성오산 Wee센터","주소":"오산시 북삼미로 119 화성오산교육지원청 1~2층","위도":37.1507,"경도":127.0773,"전화":"031-371-0651"},
    {"지역":"광주","유형":"일반","이름":"광주하남(광주) Wee센터","주소":"광주시 송정동 240-1 광주하남교육지원청 1층","위도":37.4296,"경도":127.2550,"전화":"031-280-7271"},
    {"지역":"하남","유형":"일반","이름":"광주하남(하남) Wee센터","주소":"하남시 망월동 200, 1층","위도":37.5395,"경도":127.2147,"전화":"02-480-5131"},
    {"지역":"양평","유형":"일반","이름":"양평 Wee센터","주소":"양평군 양평읍 양근강변길 126 양평교육지원청","위도":37.4919,"경도":127.4878,"전화":"031-770-5630"},
    {"지역":"이천","유형":"일반","이름":"이천 Wee센터","주소":"이천시 이섭대천로 1311번길 18 이천교육지원청 3층","위도":37.2719,"경도":127.4347,"전화":"031-639-1654"},
    {"지역":"용인","유형":"일반","이름":"용인 Wee센터","주소":"용인시 수지구 대지로 128 현암중학교 4층","위도":37.3271,"경도":127.0962,"전화":"031-889-5890"},
    {"지역":"안성","유형":"일반","이름":"안성 Wee센터","주소":"안성시 백성2길 59, 안성이룸학교 본관 2층","위도":37.0078,"경도":127.2796,"전화":"031-670-8770"},
    {"지역":"김포","유형":"일반","이름":"김포 Wee센터","주소":"김포시 김포한강1로 98번길 35 고창초등학교 후관 4층","위도":37.6152,"경도":126.7160,"전화":"031-982-2950"},
    {"지역":"시흥","유형":"일반","이름":"시흥 Wee센터","주소":"시흥시 장현순환로 100 시흥가온중학교 4층","위도":37.3800,"경도":126.8030,"전화":"031-365-8290"},
    {"지역":"의정부","유형":"일반","이름":"의정부 Wee센터","주소":"의정부시 가능로 136번길 29 의정부교육지원청 별관 1층","위도":37.7380,"경도":127.0452,"전화":"031-820-0092"},
    {"지역":"동두천양주","유형":"일반","이름":"동두천양주 Wee센터","주소":"동두천시 중앙로 110-32 동두천양주교육지원청 2층","위도":37.9040,"경도":127.0607,"전화":"031-860-4333"},
    {"지역":"고양","유형":"일반","이름":"고양 Wee센터","주소":"경기도 고양시 일산동구 장대길 64-27","위도":37.6584,"경도":126.8310,"전화":"031-901-9170"},
    {"지역":"구리남양주","유형":"일반","이름":"구리남양주 Wee센터","주소":"남양주시 경춘로 520 구리남양주교육지원청 2층","위도":37.6560,"경도":127.2165,"전화":"031-550-6346"},
    {"지역":"파주","유형":"일반","이름":"파주 Wee센터","주소":"파주시 교하로 1290번길 38, 파주교육지원청 학교지원센터 2층","위도":37.7577,"경도":126.7796,"전화":"031-948-8834"},
    {"지역":"연천","유형":"일반","이름":"연천 Wee센터","주소":"연천군 연천읍 연천로 356-1 연천교육지원청 별관 2층","위도":38.0967,"경도":127.0750,"전화":"031-839-0165"},
    {"지역":"포천","유형":"일반","이름":"포천 Wee센터","주소":"포천시 군내면 호국로 1520 포천교육지원청 마홀나래관 2층","위도":37.8956,"경도":127.2003,"전화":"031-539-0150"},
    {"지역":"가평","유형":"일반","이름":"가평 Wee센터","주소":"가평군 가평읍 향교로 17 가평교육지원청 별관 2층","위도":37.8314,"경도":127.5096,"전화":"031-580-5174"},
    {"지역":"수원","유형":"가정형","이름":"봄날(가정형)","주소":"경기도 수원시 팔달구 인계동 246-10","위도":37.2630,"경도":127.0357,"전화":"031-548-1232"},
    {"지역":"고양","유형":"가정형","이름":"숨겨진 보물(가정형)","주소":"경기도 일산동구 설문동 159-8 가동","위도":37.6712,"경도":126.8021,"전화":"031-976-0179"},
    {"지역":"성남","유형":"병원형","이름":"도담도담 Wee센터","주소":"성남 수정구 성남대로1262","위도":37.4330,"경도":127.1481,"전화":"031-698-3320"},
    {"지역":"의정부","유형":"병원형","이름":"룰루랄라 Wee센터","주소":"의정부시 평화로447","위도":37.7385,"경도":127.0430,"전화":"031-928-0911"},
    {"지역":"용인","유형":"병원형","이름":"이음 Wee센터","주소":"용인시 기흥구 흥덕1로 97","위도":37.2899,"경도":127.1162,"전화":"031-812-1500"},
    {"지역":"부천","유형":"병원형","이름":"피노키오 Wee센터","주소":"부천시 소사구 경인로 88","위도":37.4785,"경도":126.7902,"전화":"032-310-0171"},
]

# ── 연계기관 샘플 데이터 ──────────────────────────────────────────────────────
SAMPLE_AGENCIES = [
    {"연번":1,"기관명":"시은심리상담센터","지역":"안양","유형":"상담지원형","주소":"안양시 만안구 안양로","위도":37.3974,"경도":126.9560,"전화":"031-000-0001","상태":"승인","승인일":"2026-03-01"},
    {"연번":2,"기관명":"뉴로마인드 심리상담","지역":"안양","유형":"상담지원형","주소":"안양시 동안구 평촌로","위도":37.3929,"경도":126.9670,"전화":"031-000-0002","상태":"승인","승인일":"2026-03-01"},
    {"연번":3,"기관명":"다솜치료교육센터","지역":"안양","유형":"종합지원형","주소":"안양시 만안구 박달로","위도":37.4020,"경도":126.9440,"전화":"031-000-0003","상태":"승인","승인일":"2026-03-01"},
    {"연번":4,"기관명":"별심은나무 심리상담센터","지역":"과천","유형":"상담지원형","주소":"과천시 관문로","위도":37.4299,"경도":126.9877,"전화":"031-000-0004","상태":"승인","승인일":"2026-03-05"},
    {"연번":5,"기관명":"온누리심리상담센터","지역":"안양","유형":"상담지원형","주소":"안양시 동안구 흥안대로","위도":37.3882,"경도":126.9593,"전화":"031-000-0005","상태":"승인","승인일":"2026-03-05"},
    {"연번":6,"기관명":"위드유교육치료연구소","지역":"안양","유형":"종합지원형","주소":"안양시 만안구 냉천로","위도":37.4065,"경도":126.9501,"전화":"031-000-0006","상태":"검토중","승인일":""},
    {"연번":7,"기관명":"유해피아동청소년심리상담센터","지역":"과천","유형":"상담지원형","주소":"과천시 별양로","위도":37.4329,"경도":126.9870,"전화":"031-000-0007","상태":"승인","승인일":"2026-03-10"},
    {"연번":8,"기관명":"윤심리상담센터","지역":"안양","유형":"상담지원형","주소":"안양시 동안구 시민대로","위도":37.3945,"경도":126.9627,"전화":"031-000-0008","상태":"승인","승인일":"2026-03-10"},
    {"연번":9,"기관명":"이너스심리상담센터","지역":"안양","유형":"정신의료형","주소":"안양시 만안구 삼덕로","위도":37.4005,"경도":126.9480,"전화":"031-000-0009","상태":"승인","승인일":"2026-03-15"},
    {"연번":10,"기관명":"이든샘가족아동청소년상담소","지역":"안양","유형":"상담지원형","주소":"안양시 동안구 귀인로","위도":37.3905,"경도":126.9705,"전화":"031-000-0010","상태":"대기","승인일":""},
]

# ── 샘플 데이터 생성 ──────────────────────────────────────────────────────────
@st.cache_data
def make_sample():
    np.random.seed(42)
    schools = ['가온초등학교','나래중학교','다솜고등학교','라온초등학교','마루중학교']
    levels  = ['초등','중등','고등']
    types_  = ['상담지원형','정신의료형','종합지원형']
    orgs    = ['마음드림심리상담소','연세정신건강의학과의원','한국아동청소년발달센터','아이사랑상담센터','아이꿈상담소']
    stati   = ['진행중','진행중','진행중','종결','병원종결','자퇴','취소']
    names   = ['김하늘','이바다','박구름','최태양','정별이','송우주','윤사랑','한미소','오슬기','조진우',
               '양현석','민경수','손예진','권지용','유재석','이효리','강다니엘','박지훈','최유나','나은별']
    rows=[]
    for i in range(80):
        total=3_000_000; jaboo=random.choice([0,50000,100000,150000,200000,300000])
        used=int(total*random.uniform(0.30,0.95))
        lv=random.choice(levels)
        gr=random.randint(1,6) if lv=='초등' else random.randint(1,3)
        rows.append({'번호':i+1,'접수일':'2026-03-01','학교':random.choice(schools),'학교급':lv,'학년':gr,'반':random.randint(1,10),'학생번호':random.randint(1,30),'성명':names[i%len(names)]+str(i),'유형':random.choice(types_),'연계기관':random.choice(orgs),'자부담금액':jaboo,'사용금액':used,'총 지원금액':total,'종결서류':random.choice(stati),'Source':1})
    rows.append({'번호':99,'접수일':'2026-06-12','학교':'가온초등학교','학교급':'초등','학년':3,'반':6,'학생번호':29,'성명':'김지원','유형':'상담지원형','연계기관':'아이꿈상담소','자부담금액':0,'사용금액':1150000,'총 지원금액':3000000,'종결서류':'진행중','Source':1})
    df=pd.DataFrame(rows)
    df['잔액']=df['총 지원금액']-df['사용금액']
    df['집행률']=(df['사용금액']/df['총 지원금액']*100).round(1)
    def risk(r):
        s=str(r['종결서류']).strip()
        if s=='진행중':
            if r['집행률']>=85: return '높음'
            if r['집행률']>=50: return '보통'
            return '낮음'
        if s=='병원종결': return '종결(병원)'
        return '종결'
    df['위험도']=df.apply(risk,axis=1)
    return df

def load_csv(file_obj):
    try:
        df = pd.read_csv(file_obj, encoding='cp949')
    except Exception:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding='utf-8-sig')
    df.rename(columns={'번호.1': '학생번호'}, inplace=True)
    for col in ['총 지원금액', '자부담금액', '사용금액']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',','',regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['잔액'] = df['총 지원금액'] - df['사용금액']
    df['집행률'] = (df['사용금액'] / df['총 지원금액'].replace(0, np.nan) * 100).fillna(0).round(1)
    def risk(r):
        s = str(r['종결서류']).strip()
        if s == '진행중':
            if r['집행률'] >= 85: return '높음'
            if r['집행률'] >= 50: return '보통'
            return '낮음'
        if s == '병원종결': return '종결(병원)'
        return '종결'
    df['위험도'] = df.apply(risk, axis=1)
    return df

# ── LLM 기반 AI 함수들 ─────────────────────────────────────────────────────────

# Rule-based fallback (API 없을 때)
_TEACHER_FALLBACK = {
    '상담지원형': {
        'lines': [
            "수업 중 소그룹 활동을 우선 배정하고 발표보다 참여에 초점을 맞춰주세요.",
            "긍정적 피드백을 자주 제공하고, 감정 표현을 자연스럽게 격려해주세요.",
            "감정 변화 관찰 시 즉시 기록하고 Wee센터로 공유해주세요.",
        ],
        'basis': "상담지원형 · 정서·사회성 지원 중심"
    },
    '정신의료형': {
        'lines': [
            "과도한 자극(큰 소리, 급작스러운 변경)을 최소화하고 예측 가능한 루틴을 유지해주세요.",
            "복약 일정을 존중하고 조용히 여부를 확인해주세요.",
            "'힘들다' 표현 시 즉각 경청하고, 필요 시 보건실 동행을 허용해주세요.",
        ],
        'basis': "정신의료형 · 의료기관 연계 치료 중"
    },
    '종합지원형': {
        'lines': [
            "IEP 내용과 연계하여 과제량과 평가 방식을 조정해주세요.",
            "학교·가정·기관 간 일관된 메시지가 중요합니다.",
            "주 1회 이상 짧은 면담으로 적응도를 체크하고 긍정 변화를 기록해두세요.",
        ],
        'basis': "종합지원형 · 의료+상담 복합 지원 중"
    }
}
_PARENT_FALLBACK = {
    '상담지원형': [
        "🏠 가정에서 규칙적인 일과(식사·수면·놀이)를 유지해주세요. 안정된 루틴이 치료 효과를 높입니다.",
        "💬 아이가 학교 이야기를 꺼낼 때 판단 없이 들어주고, '오늘 어땠어?'로 대화를 시작해보세요.",
        "📖 전문 상담사가 권장하는 감정 카드 활동을 주 2~3회 함께 해보세요.",
    ],
    '정신의료형': [
        "💊 처방받은 약은 정해진 시간에 빠짐없이 복용시켜주세요. 임의 중단은 반드시 의사와 상의하세요.",
        "😴 충분한 수면(10시 이전 취침)을 지켜주세요. 수면 부족은 증상을 악화시킵니다.",
        "📅 다음 진료 전 아이의 변화(식욕·수면·감정)를 메모해 의사에게 전달해주세요.",
    ],
    '종합지원형': [
        "🤝 학교 담임 선생님과 월 1회 이상 소통하여 학교·가정 환경을 일관되게 유지해주세요.",
        "🎮 치료 외 시간에 아이가 좋아하는 활동을 함께 즐기며 긍정적 경험을 쌓아주세요.",
        "📋 기관 권장 가정 활동지를 꾸준히 실천하고, 작은 성공도 크게 칭찬해주세요.",
    ]
}

def ai_teacher_guide(stu: dict) -> tuple[list, str, bool]:
    """
    담임교사용 AI 교실 지원 가이드.
    Returns: (lines, basis, is_llm)
      is_llm=True  → GPT-4o 실제 생성
      is_llm=False → rule-based fallback
    """
    유형 = stu.get('유형', '상담지원형')
    성명 = stu.get('성명', '학생')
    종결 = stu.get('종결서류', '진행중')
    집행률 = float(stu.get('집행률', 0))
    연계기관 = stu.get('연계기관', '연계기관')
    위험도 = stu.get('위험도', '보통')

    sys_prompt = (
        "너는 10년 차 전문 아동 심리 상담사이자 특수교육 전문가야. "
        "아래 학생 데이터를 바탕으로 담임교사가 교실에서 취해야 할 행동 지침 3가지를 "
        "친절하고 전문적인 어투로 작성해 줘. "
        "각 지침은 '▸ ' 없이 한 문장으로, 번호 없이 줄바꿈으로만 구분해줘. "
        "전문 용어는 간단히 풀어서 설명해주고, 구체적이고 실행 가능한 내용이어야 해."
    )
    user_prompt = (
        f"학생명: {성명}\n"
        f"지원유형: {유형}\n"
        f"연계기관: {연계기관}\n"
        f"바우처 집행률: {집행률:.0f}%\n"
        f"위험도: {위험도}\n"
        f"현재상태: {종결}\n"
        "위 정보를 바탕으로 담임교사 행동 지침 3가지를 작성해줘."
    )

    result = call_llm(sys_prompt, user_prompt, max_tokens=400)
    if result:
        lines = [l.strip() for l in result.split('\n') if l.strip()][:4]
        if 종결 in ['종결', '병원종결']:
            lines.append("⚠️ 치료 종결 이후에도 적응 기간 세심한 관찰이 필요합니다. 재개입 신호에 주의하세요.")
        basis = f"GPT-4o 생성 · {연계기관} 연계 데이터 기반 · 집행률 {집행률:.0f}%"
        return lines, basis, True

    # fallback
    fb = _TEACHER_FALLBACK.get(유형, _TEACHER_FALLBACK['상담지원형'])
    lines = list(fb['lines'])
    if 종결 in ['종결', '병원종결']:
        lines.append("⚠️ 치료 종결 이후에도 적응 기간 세심한 관찰이 필요합니다. 재개입 신호에 주의하세요.")
    basis = f"규칙 기반 가이드 · {fb['basis']} · 집행률 {집행률:.0f}%"
    return lines, basis, False


def ai_parent_message(stu: dict) -> tuple[list, bool]:
    """
    학부모용 AI 행동 추천.
    Returns: (messages, is_llm)
    """
    유형 = stu.get('유형', '상담지원형')
    성명 = stu.get('성명', '학생')
    연계기관 = stu.get('연계기관', '연계기관')
    집행률 = float(stu.get('집행률', 0))
    위험도 = stu.get('위험도', '보통')

    sys_prompt = (
        "너는 10년 차 전문 아동 심리 상담사야. "
        "아래 학생 데이터를 바탕으로 학부모가 가정에서 실천할 수 있는 구체적인 행동 추천 3가지를 "
        "따뜻하고 공감적인 어투로 작성해줘. "
        "이모지를 앞에 붙여서 읽기 쉽게 만들어줘. "
        "번호나 기호 없이 줄바꿈으로만 구분해줘."
    )
    user_prompt = (
        f"학생명: {성명}\n"
        f"지원유형: {유형}\n"
        f"연계기관: {연계기관}\n"
        f"바우처 집행률: {집행률:.0f}%\n"
        f"위험도: {위험도}\n"
        "위 정보를 바탕으로 학부모 행동 추천 3가지를 작성해줘."
    )

    result = call_llm(sys_prompt, user_prompt, max_tokens=400)
    if result:
        msgs = [l.strip() for l in result.split('\n') if l.strip()][:3]
        return msgs, True

    # fallback
    return _PARENT_FALLBACK.get(유형, _PARENT_FALLBACK['상담지원형']), False


def ai_hospital_reasoning(symptoms: list, matched: list) -> tuple[str, bool]:
    """
    매칭된 병원 TOP3에 대해 AI가 추천 근거를 설명.
    Returns: (reasoning_text, is_llm)
    """
    if not matched:
        return "", False

    sys_prompt = (
        "너는 아동·청소년 정신건강 전문가야. "
        "아래 증상과 추천된 기관 정보를 보고, "
        "왜 이 기관들이 해당 학생에게 최적의 선택인지 2~3문장으로 명확하게 설명해줘. "
        "각 기관별로 한 줄씩 설명해줘. 전문용어는 쉽게 풀어서 작성해줘."
    )
    hosp_info = "\n".join([
        f"{i+1}. {h['이름']} (전문: {', '.join(h['전문'])}, 평점: {h['평점']}, 대기: {h['대기']})"
        for i, h in enumerate(matched)
    ])
    user_prompt = (
        f"학생 증상: {', '.join(symptoms)}\n"
        f"추천 기관:\n{hosp_info}\n\n"
        "각 기관이 이 학생에게 적합한 이유를 간략히 설명해줘."
    )

    result = call_llm(sys_prompt, user_prompt, max_tokens=300)
    if result:
        return result, True
    # fallback
    fallback = f"입력하신 증상({', '.join(symptoms)})과 가장 높은 전문성 매칭 점수 및 거리를 기반으로 자동 선별된 기관입니다."
    return fallback, False


# ── 병원 데이터 ───────────────────────────────────────────────────────────────
HOSPITAL_DB = [
    {'이름':'아이꿈상담소','주소':'성남시 분당구 정자동','위도':37.362,'경도':127.108,'전문':['상담지원형','정서불안','또래관계'],'평점':4.8,'대기':'2일','거리_km':1.2},
    {'이름':'마음드림심리상담소','주소':'성남시 수정구 신흥동','위도':37.443,'경도':127.137,'전문':['상담지원형','ADHD','우울'],'평점':4.7,'대기':'5일','거리_km':3.8},
    {'이름':'연세정신건강의학과의원','주소':'성남시 중원구 금광동','위도':37.435,'경도':127.145,'전문':['정신의료형','ADHD','경계선지능'],'평점':4.9,'대기':'1일','거리_km':4.1},
    {'이름':'한국아동청소년발달센터','주소':'용인시 수지구 풍덕천동','위도':37.323,'경도':127.095,'전문':['종합지원형','경계선지능','발달지연'],'평점':4.6,'대기':'7일','거리_km':6.3},
    {'이름':'아이사랑상담센터','주소':'성남시 분당구 야탑동','위도':37.413,'경도':127.127,'전문':['상담지원형','학교폭력','가정불화'],'평점':4.5,'대기':'3일','거리_km':2.7},
    {'이름':'행복한의원','주소':'광주시 태전동','위도':37.414,'경도':127.265,'전문':['정신의료형','우울','불안'],'평점':4.4,'대기':'4일','거리_km':9.1},
    {'이름':'미래심리치료소','주소':'성남시 분당구 수내동','위도':37.383,'경도':127.121,'전문':['종합지원형','ADHD','정서불안'],'평점':4.7,'대기':'3일','거리_km':2.1},
    {'이름':'푸른숲정신과','주소':'성남시 수정구 태평동','위도':37.457,'경도':127.135,'전문':['정신의료형','경계선지능','우울'],'평점':4.6,'대기':'6일','거리_km':5.4},
]
SYMPTOM_MAP = {
    'ADHD':['정신의료형','ADHD'],'우울':['정신의료형','우울','상담지원형'],'불안장애':['정신의료형','불안','상담지원형'],
    '경계선 지능':['종합지원형','경계선지능'],'정서 불안':['상담지원형','정서불안','종합지원형'],
    '학교폭력 피해':['상담지원형','학교폭력'],'가정불화':['상담지원형','가정불화'],'또래관계 어려움':['상담지원형','또래관계'],
    '발달지연':['종합지원형','발달지연'],
}

def match_hospitals(symptoms):
    if not symptoms: return []
    kws = []
    for s in symptoms:
        kws.extend(SYMPTOM_MAP.get(s, [s]))
    scored = []
    for h in HOSPITAL_DB:
        score = sum(1 for k in kws if any(k in f for f in h['전문']))
        if score > 0:
            scored.append((score, h['거리_km'], h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [h for _,_,h in scored[:3]]

@st.cache_data
def get_checkin_data(student_name):
    np.random.seed(abs(hash(student_name)) % 300)
    weeks = []
    base = datetime(2026, 3, 1)
    for i in range(8):
        dt = base + timedelta(weeks=i)
        mood = max(1, min(5, round(2.5 + np.random.normal(0.3*i/7, 0.8))))
        attend = random.choice([True, True, True, False]) if i < 6 else random.choice([True, True, False])
        weeks.append({'주차': f"{i+1}주차", '날짜': dt.strftime('%m/%d'), '기분점수': mood, '출석': attend,
                      '메모': random.choice(['평범한 하루였어요','친구랑 놀았어요','조금 힘들었어요','선생님이 칭찬해줬어요','배가 아팠어요'])})
    return weeks

@st.cache_data
def get_visit_frequency(student_name):
    np.random.seed(abs(hash(student_name)) % 150)
    return [random.randint(1, 5) for _ in ['1월', '2월', '3월', '4월']]

# ── 인증 화면 ─────────────────────────────────────────────────────────────────
def auth_screen(title: str, subtitle: str, pw_key: str, correct_pw: str):
    """비밀번호 인증 화면. 버튼이 사라지지 않도록 form을 사용."""
    st.markdown(f"""
    <div class="lock-box">
      <div class="lock-icon">🔐</div>
      <div class="lock-title">{title}</div>
      <div class="lock-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
    with col:
        with st.form(key=f"form_{pw_key}", clear_on_submit=False):
            pw = st.text_input(
                "비밀번호",
                type="password",
                label_visibility='collapsed',
                placeholder="비밀번호를 입력하세요"
            )
            submitted = st.form_submit_button("🔓 접속", use_container_width=True)
            if submitted:
                if pw == correct_pw:
                    st.session_state[f'auth_{pw_key}'] = True
                    st.rerun()
                else:
                    st.error("❌ 비밀번호가 올바르지 않습니다.")

# ── session_state 초기화 ──────────────────────────────────────────────────────
if 'agency_list' not in st.session_state:
    st.session_state['agency_list'] = list(SAMPLE_AGENCIES)
if 'closure_reports' not in st.session_state:
    st.session_state['closure_reports'] = [
        {'기관':'아이꿈상담소','학생':'김지원','종결일':'2026-04-15','사유':'목표 달성 종결','소견':'정서 안정 및 또래 관계 호전','상태':'검토중'},
        {'기관':'연세정신건강의학과의원','학생':'이바다0','종결일':'2026-04-08','사유':'병원종결','소견':'약물 조절 완료, 추적 관찰 권고','상태':'완료'},
    ]
if 'hosp_applications' not in st.session_state:
    st.session_state['hosp_applications'] = []

# ── AI 결과 캐시 (session_state) ──────────────────────────────────────────────
def get_ai_cache_key(prefix, stu_name, 유형):
    return f"{prefix}_{stu_name}_{유형}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 1rem 0;">'
        '<div class="ml-logo">Mind<span>-Link</span> AI</div>'
        '<div class="ml-logo-sub">학생마음바우처 통합관리</div></div>'
        '<hr style="border-color:rgba(255,255,255,0.1);margin-bottom:1rem;">',
        unsafe_allow_html=True
    )
    role = st.selectbox("🔐 사용자 권한", ["담임교사","위센터 주무관","학부모","병원 관계자"])
    st.markdown("---")

    school_sel = grade_sel = class_sel = None
    df = make_sample()
    data_source = "샘플 데이터"

    if role == "담임교사":
        school_sel = st.selectbox("담당 학교", sorted(df['학교'].unique()))
        grades_avail = sorted(df[df['학교']==school_sel]['학년'].unique())
        grade_sel = st.selectbox("학년", grades_avail)
        classes_avail = sorted(df[(df['학교']==school_sel)&(df['학년']==grade_sel)]['반'].unique())
        class_sel = st.selectbox("반", classes_avail)

    st.markdown("---")

    # API 키 입력 (사이드바 하단)
    with st.expander("⚙️ AI API 설정", expanded=False):
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="openai_api_key_input",
            label_visibility='visible'
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.success("✅ API 키 적용됨")
        else:
            st.caption("API 키 없으면 규칙 기반(fallback) 작동")

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""

    has_api = bool(os.environ.get("OPENAI_API_KEY", "") or secret_key)
    
    if has_api:
        st.markdown('<div style="background:#E6F9F5;border-radius:6px;padding:0.3rem 0.6rem;font-size:0.68rem;color:#00A878;font-weight:700;margin-top:0.3rem;">🤖 GPT-4o 연결됨</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#FFF0E5;border-radius:6px;padding:0.3rem 0.6rem;font-size:0.68rem;color:#E85D04;font-weight:700;margin-top:0.3rem;">📋 규칙 기반 모드</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:0.67rem;color:#7FA8CC;line-height:1.9;margin-top:0.5rem;">📅 {datetime.now().strftime("%Y년 %m월 %d일")}<br>🗂 {data_source}<br>🏫 학교: {df["학교"].nunique()}개교<br>👤 학생: {len(df)}명</div>', unsafe_allow_html=True)

# ── AI 카드 렌더 헬퍼 ─────────────────────────────────────────────────────────
def render_ai_card(lines, basis, is_llm, label, border_color="#1B4F9B", label_color="#1B4F9B"):
    badge = (
        '<span class="api-badge">✨ GPT-4o</span>' if is_llm
        else '<span class="fallback-badge">📋 규칙기반</span>'
    )
    lines_html = "".join(f'<div class="ai-line">▸ {l}</div>' for l in lines)
    st.markdown(f"""
    <div style="background:transparent;border:2.5px solid {border_color};border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:0.9rem;">
      <div class="ai-label" style="color:{label_color};">{label}{badge}</div>
      {lines_html}
      <div class="ai-source">📎 {basis}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: 담임교사
# ═══════════════════════════════════════════════════════════════════════════════
def page_teacher():
    auth_key = f"teacher_{school_sel}"
    correct_pw = SCHOOL_PASSWORDS.get(school_sel, "0000")
    if not st.session_state.get(f'auth_{auth_key}'):
        auth_screen(f"🏫 {school_sel} 교사 전용 접속",
                    f"담임교사 본인 인증이 필요합니다 · 비밀번호: {correct_pw} (데모)",
                    auth_key, correct_pw)
        return

    base = df[(df['학교']==school_sel)&(df['학년']==grade_sel)&(df['반']==class_sel)]
    class_df = base[base['종결서류'].isin(ACTIVE_STATUS)].copy()
    high_risk = class_df[class_df['위험도']=='높음']

    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">'
        f'<div><h2 style="margin:0;color:#1B4F9B;font-size:1.4rem;font-weight:900;">🏫 {school_sel} {grade_sel}학년 {class_sel}반</h2>'
        f'<p style="margin:0;color:#5A6A7E;font-size:0.79rem;">담임교사 전용 · 학생 케어 지원 시스템</p></div>'
        f'<div style="background:#E6F9F5;padding:0.4rem 0.9rem;border-radius:8px;text-align:right;">'
        f'<div style="font-size:0.67rem;color:#00A878;font-weight:600;">TODAY</div>'
        f'<div style="font-size:0.8rem;color:#1A2332;font-weight:700;">{datetime.now().strftime("%m/%d %H:%M")}</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    if len(class_df) == 0:
        st.info("이 학급에 관리 대상 학생(진행중/종결/병원종결)이 없습니다.")
        return

    for _, r in high_risk.iterrows():
        st.markdown(
            f'<div class="ml-alert">'
            f'<div class="ml-alert-title">⚠️ 즉시 관리 필요 — {r["성명"]} ({r["학생번호"]}번) · {r["유형"]}</div>'
            f'<div class="ml-alert-body">연계기관: {r["연계기관"]} · 바우처 집행률 {r["집행률"]:.0f}% 높음 · 이번 주 개별 면담 권고</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">👥</div><div class="ml-metric-value">{len(class_df)}</div><div class="ml-metric-label">관리 대상 학생</div><div class="ml-metric-sub">진행중/종결/병원종결</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">🚨</div><div class="ml-metric-value">{len(high_risk)}</div><div class="ml-metric-label">고위험 학생</div><div class="ml-metric-sub">{"즉시 면담 필요" if len(high_risk)>0 else "이상 없음"}</div></div>', unsafe_allow_html=True)
    with c3:
        end_cnt = len(class_df[class_df['종결서류'].isin(['종결','병원종결'])])
        st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">✅</div><div class="ml-metric-value">{end_cnt}</div><div class="ml-metric-label">종결 학생</div><div class="ml-metric-sub">사후 모니터링 대상</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="ml-sh">📋 학급 학생 현황 목록</div>', unsafe_allow_html=True)
    table_rows = ""
    for _, r in class_df.iterrows():
        s_color = {'진행중':'#1B4F9B','종결':'#94A3B8','병원종결':'#6B4FA6'}.get(str(r['종결서류']),'#94A3B8')
        ri_color = {'높음':'#D62828','보통':'#E85D04','낮음':'#00A878','종결':'#94A3B8','종결(병원)':'#6B4FA6'}.get(r['위험도'],'#94A3B8')
        table_rows += f"<tr><td>{r['학생번호']}번</td><td><b>{r['성명']}</b></td><td>{r['유형']}</td><td><span style='color:{s_color};font-weight:600;'>{r['종결서류']}</span></td><td><span style='color:{ri_color};font-weight:600;'>{r['위험도']}</span></td><td>{r['집행률']:.0f}%</td><td style='font-size:0.72rem;color:#5A6A7E;'>{str(r['연계기관'])[:12]}</td></tr>"
    st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><table class="data-table"><thead><tr><th>번호</th><th>성명</th><th>유형</th><th>상태</th><th>위험도</th><th>집행률</th><th>연계기관</th></tr></thead><tbody>{table_rows}</tbody></table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="ml-sh">🎯 학생 선택 → AI 교실 지원 가이드</div>', unsafe_allow_html=True)
    stu_options = class_df.apply(lambda r: f"{r['학생번호']}번 {r['성명']} ({r['종결서류']})", axis=1).tolist()
    selected_label = st.selectbox("학생 선택 (개인정보 보호 — 본인 학급만 조회)", stu_options)
    stu = class_df.iloc[stu_options.index(selected_label)]
    stu_dict = stu.to_dict()

    s_tag = {'진행중':'tag-blue','종결':'tag-gray','병원종결':'tag-purple'}.get(stu['종결서류'],'tag-gray')
    r_tag = {'높음':'tag-red','보통':'tag-orange','낮음':'tag-green','종결':'tag-gray','종결(병원)':'tag-purple'}.get(stu['위험도'],'tag-gray')

    st.markdown(f"""<div class="ml-card ml-card-primary">
      <div style="display:flex;align-items:center;gap:0.8rem;">
        <div style="width:40px;height:40px;background:#E8F0FA;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.2rem;">
          {'🔴' if stu['위험도']=='높음' else '🟡' if stu['위험도']=='보통' else '⚫' if '종결' in str(stu['위험도']) else '🟢'}
        </div>
        <div>
          <div style="font-weight:900;font-size:1.05rem;color:#1A2332;">{stu['성명']} &nbsp;<span style="font-size:0.75rem;color:#5A6A7E;">{stu['학생번호']}번 · {stu['유형']}</span></div>
          <div style="font-size:0.73rem;color:#5A6A7E;margin-top:0.1rem;">연계기관: {stu['연계기관']}</div>
        </div>
        <div style="margin-left:auto;display:flex;gap:0.4rem;">
          <span class="ml-tag {s_tag}">{stu['종결서류']}</span>
          <span class="ml-tag {r_tag}">{stu['위험도']}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 핵심 섹션 제목 ──
    st.markdown("""
    <div style="margin:1.5rem 0 1rem 0;">
      <div class="ai-section-title">🤖 AI 교실 지원 가이드 &nbsp;|&nbsp; ✅ Do's &amp; 🚫 Don'ts</div>
      <div class="ai-section-sub">담임교사를 위한 핵심 행동 가이드 — 아래 내용을 꼭 확인해주세요</div>
      <div class="ai-gradient-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    # AI 가이드 생성 (캐시 적용)
    cache_key = get_ai_cache_key("teacher", stu['성명'], stu['유형'])
    if cache_key not in st.session_state:
        with st.spinner("🤖 AI가 맞춤 가이드를 생성하고 있습니다..."):
            guide_lines, basis, is_llm = ai_teacher_guide(stu_dict)
        st.session_state[cache_key] = (guide_lines, basis, is_llm)
    else:
        guide_lines, basis, is_llm = st.session_state[cache_key]

    do_tips = {'상담지원형':["소그룹 활동 우선 배정","긍정적 피드백 자주 제공","감정 표현 격려"],
               '정신의료형':["예측 가능한 루틴 유지","조용히 복약 여부 확인","힘들다 표현 시 즉시 경청"],
               '종합지원형':["IEP 연계 과제량 조정","주 1회 짧은 면담 실시","긍정적 변화 기록"]}.get(stu['유형'],["긍정적 피드백","규칙적 루틴","변화 관찰"])
    dont_tips = {'상담지원형':["전체 앞 지적·비교 금지","감정 무시·억압 금지","급격한 환경 변화 자제"],
                 '정신의료형':["큰 소리·급작스러운 변경 금지","임의 복약 독촉 금지","피로·과제 과부하 자제"],
                 '종합지원형':["과도한 경쟁 노출 금지","갑작스러운 역할 변경 금지","장시간 단독 집중 금지"]}.get(stu['유형'],["전체 앞 지적 금지","감정 무시 금지","급격한 변화 자제"])

    g1, g2, g3 = st.columns([2, 1, 1])
    with g1:
        render_ai_card(guide_lines, basis, is_llm,
                       f"🤖 AI 교실 지원 가이드 — {stu['성명']} ({stu['유형']})",
                       border_color="#1B4F9B", label_color="#1B4F9B")
    with g2:
        st.markdown(f"""<div class="ai-box-green">
          <div class="ai-label" style="color:#00A878;">✅ Do's</div>
          {''.join(f'<div class="ai-line">▸ {t}</div>' for t in do_tips)}
        </div>""", unsafe_allow_html=True)
    with g3:
        st.markdown(f"""<div class="ai-box-red">
          <div class="ai-label" style="color:#D62828;">🚫 Don\'ts</div>
          {''.join(f'<div class="ai-line">▸ {t}</div>' for t in dont_tips)}
        </div>""", unsafe_allow_html=True)

    if st.button("🔄 AI 가이드 재생성", key="regen_teacher"):
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        st.rerun()

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    ta, tb = st.columns(2)
    with ta:
        st.markdown('<div class="ml-sh">📅 상담 방문 빈도 & 치료 참여 현황</div>', unsafe_allow_html=True)
        visit_freq = get_visit_frequency(stu['성명'])
        months = ['1월','2월','3월','4월']
        fig_v = go.Figure(go.Bar(x=months, y=visit_freq,
                                  marker_color=['#D62828' if v<2 else '#1B4F9B' for v in visit_freq],
                                  text=visit_freq, textposition='outside'))
        fig_v.update_layout(height=180, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='white', plot_bgcolor='white',
                             yaxis=dict(range=[0,7], gridcolor='#F0F4FA', title='방문 횟수'),
                             xaxis=dict(gridcolor='#F0F4FA'), font=dict(family='Noto Sans KR', size=11))
        st.plotly_chart(fig_v, width='stretch', config={'displayModeBar':False})
        avg_v = sum(visit_freq)/len(visit_freq)
        if avg_v < 2:
            st.markdown(f'<div class="ml-card ml-card-warn">⚠️ <b>방문 빈도 낮음</b> — 월평균 {avg_v:.1f}회. 기관 연계 상태를 확인해주세요.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ml-card ml-card-accent">✅ <b>방문 양호</b> — 월평균 {avg_v:.1f}회 정기 방문 중입니다.</div>', unsafe_allow_html=True)

    with tb:
        st.markdown('<div class="ml-sh">💬 AI 주간 체크인 — 학생 일상 회복도</div>', unsafe_allow_html=True)
        checkins = get_checkin_data(stu['성명'])
        ci_weeks = [c['주차'] for c in checkins]
        ci_mood  = [c['기분점수'] for c in checkins]
        ci_attend= [1 if c['출석'] else 0 for c in checkins]
        fig_ci = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ci.add_trace(go.Scatter(x=ci_weeks, y=ci_mood, mode='lines+markers', name='기분점수(1-5)',
                                     line=dict(color='#1B4F9B', width=2.5), marker=dict(size=8)), secondary_y=False)
        fig_ci.add_trace(go.Bar(x=ci_weeks, y=ci_attend, name='출석여부', marker_color='rgba(0,168,120,0.25)',
                                 text=['출석' if a==1 else '결석' for a in ci_attend]), secondary_y=True)
        fig_ci.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='white', plot_bgcolor='white',
                              font=dict(family='Noto Sans KR', size=10), legend=dict(orientation='h',y=-0.25))
        fig_ci.update_yaxes(range=[0,5.5], gridcolor='#F0F4FA', title_text='기분점수', secondary_y=False)
        fig_ci.update_yaxes(range=[0,1.8], showgrid=False, title_text='출석', secondary_y=True)
        st.plotly_chart(fig_ci, width='stretch', config={'displayModeBar':False})
        st.markdown(f'<div style="font-size:0.72rem;color:#5A6A7E;margin-bottom:0.7rem;">💬 최근 체크인: "{checkins[-1]["메모"]}" ({checkins[-1]["날짜"]})</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: 위센터 주무관
# ═══════════════════════════════════════════════════════════════════════════════
def page_wee():
    if not st.session_state.get('auth_wee_officer'):
        auth_screen("🏢 위(Wee)센터 주무관 접속",
                    f"주무관 전용 시스템입니다 · 데모 비밀번호: {WEE_PASSWORD}",
                    "wee_officer", WEE_PASSWORD)
        return

    st.markdown('<h2 style="color:#1B4F9B;font-size:1.4rem;font-weight:900;margin-bottom:0.2rem;">🏢 위(Wee)센터 주무관 대시보드</h2><p style="color:#5A6A7E;font-size:0.79rem;margin-bottom:1rem;">전체 학교 바우처 집행 현황 · 연계기관 관리 · 종결 보고서 수신</p>', unsafe_allow_html=True)

    with st.expander("📁 바우처 현황 CSV 업로드", expanded=False):
        uploaded = st.file_uploader("CSV/Excel 파일", type=['csv','xlsx'], label_visibility='collapsed', key="wee_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'):
                    df_up = load_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded, sheet_name=0)
                    tmp = io.BytesIO(); df_raw.to_csv(tmp, index=False, encoding='utf-8-sig'); tmp.seek(0)
                    df_up = load_csv(tmp)
                st.session_state['uploaded_df'] = df_up
                st.success(f"✅ {len(df_up)}건 로드 완료")
            except Exception as e:
                st.error(f"파일 오류: {e}")

    use_df = st.session_state.get('uploaded_df', df)
    total_b=use_df['총 지원금액'].sum(); used_b=use_df['사용금액'].sum()
    rate=used_b/total_b*100 if total_b>0 else 0

    c1,c2,c3,c4=st.columns(4)
    ms = 'style="height:110px;"'
    with c1: st.markdown(f'<div class="ml-metric" {ms}><div class="ml-metric-icon">🏫</div><div class="ml-metric-value">{use_df["학교"].nunique()}</div><div class="ml-metric-label">연계 학교 수</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="ml-metric" {ms}><div class="ml-metric-icon">👥</div><div class="ml-metric-value">{len(use_df)}</div><div class="ml-metric-label">전체 지원 학생</div><div class="ml-metric-sub">진행중 {len(use_df[use_df["종결서류"]=="진행중"])}명</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="ml-metric" {ms}><div class="ml-metric-icon">💳</div><div class="ml-metric-value">{int(total_b/10000)}만</div><div class="ml-metric-label">총 예산 (원)</div><div class="ml-metric-sub">사용 {int(used_b/10000)}만원</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="ml-metric" {ms}><div class="ml-metric-icon">📈</div><div class="ml-metric-value">{rate:.1f}%</div><div class="ml-metric-label">전체 예산 소진율</div><div class="ml-metric-sub">{"⚠️ 높음" if rate>75 else "✅ 정상"}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4,tab5=st.tabs(["📊 집행 현황","🏥 기관별 학생 조회","📋 연계기관 신청 리스트","📂 종결 보고서","🗺️ Wee센터 지도"])

    with tab1:
        top_row = st.columns(3)
        with top_row[0]:
            risk_all=use_df[use_df['종결서류']=='진행중']['위험도'].value_counts().reset_index(); risk_all.columns=['위험도','수']
            fig_r=px.pie(risk_all,names='위험도',values='수',hole=0.5,height=200,color='위험도',color_discrete_map={'높음':'#D62828','보통':'#E85D04','낮음':'#00A878'})
            fig_r.update_layout(margin=dict(l=0,r=0,t=30,b=0),font=dict(family='Noto Sans KR',size=10),title=dict(text='진행중 위험도 분포',font=dict(size=11,color='#5A6A7E'),x=0.5),paper_bgcolor='white',legend=dict(font=dict(size=9)))
            fig_r.update_traces(textinfo='percent+label',textfont_size=9)
            st.plotly_chart(fig_r, width='stretch', config={'displayModeBar':False})
        with top_row[1]:
            typ=use_df['유형'].value_counts().reset_index(); typ.columns=['유형','수']
            fig_t=px.pie(typ,names='유형',values='수',hole=0.5,height=200,color_discrete_sequence=['#1B4F9B','#00A878','#E85D04'])
            fig_t.update_layout(margin=dict(l=0,r=0,t=30,b=0),font=dict(family='Noto Sans KR',size=10),title=dict(text='지원 유형 분포',font=dict(size=11,color='#5A6A7E'),x=0.5),paper_bgcolor='white',legend=dict(font=dict(size=9)))
            fig_t.update_traces(textinfo='percent+label',textfont_size=9)
            st.plotly_chart(fig_t, width='stretch', config={'displayModeBar':False})
        with top_row[2]:
            status_all=use_df['종결서류'].value_counts().head(6).reset_index(); status_all.columns=['상태','수']
            fig_st=px.bar(status_all,x='수',y='상태',orientation='h',text='수',height=200,color_discrete_sequence=['#2E6FC4'])
            fig_st.update_layout(margin=dict(l=0,r=30,t=30,b=0),font=dict(family='Noto Sans KR',size=10),title=dict(text='종결 사유 현황',font=dict(size=11,color='#5A6A7E'),x=0.5),paper_bgcolor='white',plot_bgcolor='white',xaxis=dict(gridcolor='#F0F4FA'),yaxis=dict(gridcolor='#F0F4FA'))
            fig_st.update_traces(textposition='outside')
            st.plotly_chart(fig_st, width='stretch', config={'displayModeBar':False})

        cl,cr=st.columns([3,2])
        with cl:
            st.markdown('<div class="ml-sh">🏫 학교별 예산 집행률</div>', unsafe_allow_html=True)
            sch=use_df.groupby('학교').agg(총예산=('총 지원금액','sum'),사용=('사용금액','sum')).reset_index()
            sch['집행률']=sch['사용']/sch['총예산']*100
            sch=sch.sort_values('집행률',ascending=True)
            fig1=go.Figure(go.Bar(x=sch['집행률'],y=sch['학교'],orientation='h',marker_color=['#D62828' if r>85 else '#1B4F9B' for r in sch['집행률']],text=[f"{r:.1f}%" for r in sch['집행률']],textposition='outside'))
            fig1.update_layout(height=max(250,len(sch)*28),margin=dict(l=0,r=50,t=10,b=0),paper_bgcolor='white',plot_bgcolor='white',xaxis=dict(range=[0,115],gridcolor='#F0F4FA',title='집행률(%)'),yaxis=dict(gridcolor='#F0F4FA'),font=dict(family='Noto Sans KR',size=11))
            st.plotly_chart(fig1, width='stretch', config={'displayModeBar':False})
        with cr:
            st.markdown('<div class="ml-sh">🏥 기관별 정산 현황</div>', unsafe_allow_html=True)
            orgs_s=use_df.groupby('연계기관').agg(건수=('성명','count'),사용=('사용금액','sum')).reset_index().sort_values('건수',ascending=False).head(8)
            stati_list=['정산완료','검토중','미제출','정산완료','검토중','정산완료','미제출','검토중']
            tbl_rows=""
            for i,(_,row) in enumerate(orgs_s.iterrows()):
                s=stati_list[i%len(stati_list)]
                s_color={'정산완료':'#00A878','검토중':'#E85D04','미제출':'#D62828'}[s]
                tbl_rows+=f"<tr><td>{str(row['연계기관'])[:14]}</td><td>{row['건수']}</td><td>{int(row['사용'])//10000}만원</td><td style='color:{s_color};font-weight:600;'>{s}</td></tr>"
            st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><table class="data-table"><thead><tr><th>기관명</th><th>건수</th><th>사용금액</th><th>정산상태</th></tr></thead><tbody>{tbl_rows}</tbody></table></div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="ml-sh">🏥 기관별 학생 현황 조회</div>', unsafe_allow_html=True)
        sel_org=st.selectbox("기관 선택", sorted(use_df['연계기관'].unique()), key="wee_sel_org")
        org_df=use_df[use_df['연계기관']==sel_org].copy()
        oc1,oc2,oc3=st.columns(3)
        with oc1: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">👥</div><div class="ml-metric-value">{len(org_df)}</div><div class="ml-metric-label">연계 학생 수</div></div>', unsafe_allow_html=True)
        with oc2: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">💰</div><div class="ml-metric-value">{int(org_df["사용금액"].sum()/10000)}만</div><div class="ml-metric-label">총 사용금액</div></div>', unsafe_allow_html=True)
        with oc3: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">🏫</div><div class="ml-metric-value">{org_df["학교"].nunique()}</div><div class="ml-metric-label">연계 학교 수</div></div>', unsafe_allow_html=True)
        if len(org_df) > 0:
            tbl_stu=""
            for _,r in org_df.iterrows():
                s_color={'진행중':'#1B4F9B','종결':'#94A3B8','병원종결':'#6B4FA6'}.get(str(r['종결서류']),'#94A3B8')
                ri_color={'높음':'#D62828','보통':'#E85D04','낮음':'#00A878','종결':'#94A3B8','종결(병원)':'#6B4FA6'}.get(r['위험도'],'#94A3B8')
                tbl_stu+=f"<tr><td>{r['성명']}</td><td>{r['학교']}</td><td>{r['학년']}학년 {r['반']}반</td><td>{r['유형']}</td><td style='color:{s_color};font-weight:600;'>{r['종결서류']}</td><td style='color:{ri_color};font-weight:600;'>{r['위험도']}</td><td>{r['집행률']:.0f}%</td><td>{int(r['잔액'])//10000}만원</td></tr>"
            st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><div style="font-weight:700;font-size:0.88rem;color:#1B4F9B;margin-bottom:0.7rem;">📋 {sel_org} 연계 학생 목록 ({len(org_df)}명)</div><table class="data-table"><thead><tr><th>성명</th><th>학교</th><th>학년/반</th><th>유형</th><th>상태</th><th>위험도</th><th>집행률</th><th>잔액</th></tr></thead><tbody>{tbl_stu}</tbody></table></div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="ml-sh">📋 연계기관 지정 신청 현황</div>', unsafe_allow_html=True)
        agencies = st.session_state['agency_list']
        hosp_apps = st.session_state.get('hosp_applications', [])
        all_apps = agencies + [dict(연번=len(agencies)+i+1, **a, 상태=a.get('상태','대기'), 승인일=a.get('승인일','')) for i,a in enumerate(hosp_apps)]
        f1,f2 = st.columns(2)
        with f1: filter_status = st.selectbox("상태 필터", ["전체","승인","검토중","대기"], key="ag_filter")
        with f2: filter_type = st.selectbox("유형 필터", ["전체","상담지원형","정신의료형","종합지원형"], key="ag_type_filter")
        filtered = [a for a in all_apps if (filter_status=="전체" or a.get('상태')==filter_status) and (filter_type=="전체" or a.get('유형')==filter_type)]
        tbl=""
        for a in filtered:
            s=a.get('상태','대기'); s_color={'승인':'#00A878','검토중':'#E85D04','대기':'#94A3B8'}.get(s,'#94A3B8')
            tbl+=f"<tr><td>{a.get('연번','')}</td><td><b>{a.get('기관명','')}</b></td><td>{a.get('지역','')}</td><td>{a.get('유형','')}</td><td>{a.get('전화','')}</td><td style='color:{s_color};font-weight:600;'>{s}</td><td style='font-size:0.72rem;'>{a.get('승인일','')}</td></tr>"
        st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><table class="data-table"><thead><tr><th>#</th><th>기관명</th><th>지역</th><th>유형</th><th>전화</th><th>상태</th><th>승인일</th></tr></thead><tbody>{tbl}</tbody></table></div>""", unsafe_allow_html=True)
        st.markdown('<div class="ml-sh">✅ 신청 기관 승인 처리</div>', unsafe_allow_html=True)
        pending = [a['기관명'] for a in all_apps if a.get('상태') in ['대기','검토중']]
        if pending:
            with st.form("approve_form"):
                sel_agency = st.selectbox("승인할 기관 선택", pending)
                new_status = st.selectbox("처리 상태", ["승인","검토중","반려"])
                approval_date = st.date_input("처리일", value=datetime.now())
                if st.form_submit_button("✅ 처리 저장"):
                    for a in st.session_state['agency_list']:
                        if a['기관명'] == sel_agency:
                            a['상태'] = new_status; a['승인일'] = str(approval_date)
                    for a in st.session_state['hosp_applications']:
                        if a.get('기관명') == sel_agency:
                            a['상태'] = new_status; a['승인일'] = str(approval_date)
                    st.success(f"✅ {sel_agency} → {new_status} 처리 완료. 지도에 자동 반영됩니다.")
                    st.rerun()
        else:
            st.info("대기 중인 신청 기관이 없습니다.")

    with tab4:
        st.markdown('<div class="ml-sh">📂 병원 종결 보고서 수신함</div>', unsafe_allow_html=True)
        reports = st.session_state['closure_reports']
        tbl_rep=""
        for r in reports:
            s=r.get('상태','검토중'); s_color={'완료':'#00A878','검토중':'#E85D04','반려':'#D62828'}.get(s,'#E85D04')
            tbl_rep+=f"<tr><td>{r.get('기관','')}</td><td><b>{r.get('학생','')}</b></td><td>{r.get('종결일','')}</td><td>{r.get('사유','')}</td><td style='font-size:0.72rem;'>{r.get('소견','')[:30]}...</td><td style='color:{s_color};font-weight:600;'>{s}</td></tr>"
        st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><table class="data-table"><thead><tr><th>기관</th><th>학생</th><th>종결일</th><th>사유</th><th>소견요약</th><th>상태</th></tr></thead><tbody>{tbl_rep}</tbody></table></div>""", unsafe_allow_html=True)
        rep_names = [f"{r['기관']} - {r['학생']} ({r['종결일']})" for r in reports]
        if rep_names:
            with st.form("report_status_form"):
                sel_rep = st.selectbox("보고서 선택", rep_names)
                new_rep_status = st.selectbox("처리 상태", ["검토중","완료","반려"])
                if st.form_submit_button("💾 상태 저장"):
                    idx = rep_names.index(sel_rep)
                    st.session_state['closure_reports'][idx]['상태'] = new_rep_status
                    st.success("✅ 상태가 업데이트되었습니다."); st.rerun()

    with tab5:
        st.markdown('<div class="ml-sh">🗺️ 경기도 위(Wee)센터 · 연계기관 지도</div>', unsafe_allow_html=True)
        map_tab1, map_tab2 = st.tabs(["📍 Wee센터 위치", "🏥 연계기관 위치"])
        with map_tab1:
            wee_df = pd.DataFrame(WEE_CENTERS)
            color_map = {'일반':'#1B4F9B','가정형':'#00A878','병원형':'#E85D04'}
            fig_wee = px.scatter_mapbox(wee_df,lat='위도',lon='경도',hover_name='이름',hover_data={'위도':False,'경도':False,'주소':True,'전화':True,'유형':True},color='유형',color_discrete_map=color_map,zoom=8,height=500)
            fig_wee.update_layout(mapbox_style="open-street-map",margin=dict(l=0,r=0,t=0,b=0),font=dict(family='Noto Sans KR'),legend=dict(title="유형",orientation="v",x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.85)",bordercolor="#D1DCEC",borderwidth=1))
            fig_wee.update_traces(marker=dict(size=12))
            st.plotly_chart(fig_wee, use_container_width=True, config={'displayModeBar':False})
            c_a,c_b,c_c=st.columns(3)
            with c_a: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">🔵</div><div class="ml-metric-value">{len(wee_df[wee_df["유형"]=="일반"])}</div><div class="ml-metric-label">일반 Wee센터</div></div>', unsafe_allow_html=True)
            with c_b: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">🟢</div><div class="ml-metric-value">{len(wee_df[wee_df["유형"]=="가정형"])}</div><div class="ml-metric-label">가정형 Wee센터</div></div>', unsafe_allow_html=True)
            with c_c: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">🟠</div><div class="ml-metric-value">{len(wee_df[wee_df["유형"]=="병원형"])}</div><div class="ml-metric-label">병원형 Wee센터</div></div>', unsafe_allow_html=True)
        with map_tab2:
            agencies_map=[a for a in st.session_state['agency_list'] if a.get('상태')=='승인']
            hosp_approved=[a for a in st.session_state.get('hosp_applications',[]) if a.get('상태')=='승인']
            all_approved=agencies_map+hosp_approved
            if all_approved:
                ag_df=pd.DataFrame(all_approved)
                color_map2={'상담지원형':'#1B4F9B','정신의료형':'#E85D04','종합지원형':'#00A878'}
                fig_ag=px.scatter_mapbox(ag_df,lat='위도',lon='경도',hover_name='기관명',hover_data={'위도':False,'경도':False,'주소':True,'전화':True,'유형':True},color='유형',color_discrete_map=color_map2,zoom=11,height=450)
                fig_ag.update_layout(mapbox_style="open-street-map",margin=dict(l=0,r=0,t=0,b=0),font=dict(family='Noto Sans KR'),legend=dict(title="유형",orientation="v",x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.85)",bordercolor="#D1DCEC",borderwidth=1))
                fig_ag.update_traces(marker=dict(size=13))
                st.plotly_chart(fig_ag, use_container_width=True, config={'displayModeBar':False})
                st.caption(f"📍 승인된 연계기관 {len(all_approved)}개소 표시 중 · 출처: 안양과천교육지원청 Wee센터 (2026)")
            else:
                st.info("승인된 연계기관이 없습니다. 연계기관 신청 리스트 탭에서 승인 처리하면 지도에 표시됩니다.")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: 학부모
# ═══════════════════════════════════════════════════════════════════════════════
def page_parent():
    if not st.session_state.get('auth_parent'):
        auth_screen("👨‍👩‍👧 학부모 전용 접속",
                    f"자녀 개인정보 보호를 위해 인증이 필요합니다 (데모 비밀번호: {PARENT_PASSWORD})",
                    "parent", PARENT_PASSWORD)
        return

    # ── 학생 선택 ──────────────────────────────────────────────────────────────
    st.markdown('<h2 style="color:#1B4F9B;font-size:1.4rem;font-weight:900;margin-bottom:0.2rem;">👨‍👩‍👧 학부모 포털</h2><p style="color:#5A6A7E;font-size:0.79rem;margin-bottom:1rem;">자녀의 마음바우처 현황과 AI 맞춤 가이드를 확인하세요</p>', unsafe_allow_html=True)

    active_students = df[df['종결서류'].isin(ACTIVE_STATUS)]['성명'].unique().tolist()
    # 데모용: 기본값 김지원
    default_idx = active_students.index('김지원') if '김지원' in active_students else 0
    sel_student = st.selectbox(
        "👶 자녀 이름 선택 (데모: 김지원 선택 권장)",
        active_students,
        index=default_idx,
        key="parent_stu_sel"
    )

    rows = df[df['성명'] == sel_student]
    if len(rows) == 0:
        st.warning("해당 학생 데이터를 찾을 수 없습니다.")
        return
    stu = rows.iloc[0]
    stu_dict = stu.to_dict()
    rem_pct = (stu['잔액'] / stu['총 지원금액'] * 100) if stu['총 지원금액'] > 0 else 0

    st.markdown(f'<div style="background:#E8F0FA;border-radius:10px;padding:0.7rem 1rem;margin-bottom:1rem;font-size:0.82rem;color:#1A2332;"><b>📌 선택된 자녀:</b> {stu["성명"]} · {stu["학교"]} {stu["학년"]}학년 {stu["반"]}반 · {stu["연계기관"]}</div>', unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    pm = 'style="height:110px;"'
    with c1: st.markdown(f'<div class="ml-metric" {pm}><div class="ml-metric-icon">💳</div><div class="ml-metric-value">{int(stu["잔액"])//10000}만</div><div class="ml-metric-label">남은 바우처 잔액</div><div class="ml-metric-sub">총 {int(stu["총 지원금액"])//10000}만원 중</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="ml-metric" {pm}><div class="ml-metric-icon">🏥</div><div class="ml-metric-value" style="font-size:0.95rem;">{str(stu["연계기관"])[:8]}</div><div class="ml-metric-label">연계 기관</div></div>', unsafe_allow_html=True)
    with c3:
        emoji='🟢' if stu['종결서류']=='진행중' else '⚫'
        st.markdown(f'<div class="ml-metric" {pm}><div class="ml-metric-icon">{emoji}</div><div class="ml-metric-value" style="font-size:1.1rem;">{stu["종결서류"]}</div><div class="ml-metric-label">현재 지원 상태</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="ml-card ml-card-accent"><div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;"><span style="font-weight:600;font-size:0.86rem;">바우처 잔액 현황</span><span style="font-size:0.78rem;color:#5A6A7E;">{rem_pct:.1f}% 잔여</span></div><div style="background:#D1DCEC;border-radius:20px;height:13px;"><div style="width:{min(rem_pct,100):.0f}%;height:13px;border-radius:20px;background:linear-gradient(90deg,#00A878,#00C896);"></div></div><div style="display:flex;justify-content:space-between;margin-top:0.35rem;font-size:0.7rem;color:#5A6A7E;"><span>사용: {int(stu["사용금액"]):,}원</span><span>잔액: {int(stu["잔액"]):,}원</span></div></div>', unsafe_allow_html=True)

    cl, cr = st.columns(2)
    with cl:
        # 치료 일정
        st.markdown('<div class="ml-sh">📅 치료 일정 & 자동 차감</div>', unsafe_allow_html=True)
        visits=[{'날짜':'2026-04-10','상태':'완료','금액':80000},{'날짜':'2026-03-27','상태':'완료','금액':80000},{'날짜':'2026-04-24','상태':'예정','금액':80000}]
        tbl_vis=""
        for v in visits:
            s_color='#00A878' if v['상태']=='완료' else '#1B4F9B'
            tbl_vis+=f"<tr><td>{v['날짜']}</td><td>{stu['연계기관'][:10]}</td><td style='color:#D62828;'>-{v['금액']:,}원</td><td style='color:{s_color};font-weight:600;'>{v['상태']}</td></tr>"
        st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><table class="data-table"><thead><tr><th>날짜</th><th>기관</th><th>차감금액</th><th>상태</th></tr></thead><tbody>{tbl_vis}</tbody></table></div>""", unsafe_allow_html=True)

        # 스마트 병원 매칭
        st.markdown('<div class="ml-sh">🔍 스마트 병원 매칭 TOP 3</div>', unsafe_allow_html=True)
        sym = st.multiselect("증상 키워드 선택", list(SYMPTOM_MAP.keys()), key="parent_sym")
        if sym:
            matched = match_hospitals(sym)
            if matched:
                map_df = pd.DataFrame(matched)
                fig_map = px.scatter_mapbox(map_df,lat='위도',lon='경도',hover_name='이름',hover_data={'위도':False,'경도':False,'주소':True,'평점':True,'대기':True},color_discrete_sequence=['#1B4F9B'],zoom=11,height=220)
                fig_map.update_traces(marker=dict(size=14))
                fig_map.update_layout(mapbox_style="open-street-map",margin=dict(l=0,r=0,t=0,b=0),font=dict(family='Noto Sans KR'))
                fig_map.add_trace(go.Scattermapbox(lat=[37.380],lon=[127.112],mode='markers',marker=dict(size=16,color='#D62828'),name='현재 위치',hovertext='현재 위치'))
                st.plotly_chart(fig_map, width='stretch', config={'displayModeBar':False})

                tbl_h=""
                for i,h in enumerate(matched):
                    tbl_h+=f"<tr><td><b>{i+1}</b></td><td><b>{h['이름']}</b></td><td>⭐{h['평점']}</td><td>🚗{h['거리_km']}km</td><td>⏱{h['대기']}</td></tr>"
                st.markdown(f"""<div class="ml-card" style="padding:0.7rem;"><table class="data-table"><thead><tr><th>#</th><th>기관명</th><th>평점</th><th>거리</th><th>대기</th></tr></thead><tbody>{tbl_h}</tbody></table></div>""", unsafe_allow_html=True)

                # AI 매칭 사유 (XAI)
                reason_key = f"hosp_reason_{'-'.join(sym)}"
                if reason_key not in st.session_state:
                    with st.spinner("🤖 AI가 추천 근거를 분석 중..."):
                        reason_txt, is_llm_r = ai_hospital_reasoning(sym, matched)
                    st.session_state[reason_key] = (reason_txt, is_llm_r)
                else:
                    reason_txt, is_llm_r = st.session_state[reason_key]

                badge_r = '<span class="api-badge">✨ GPT-4o</span>' if is_llm_r else '<span class="fallback-badge">📋 규칙기반</span>'
                st.markdown(f"""<div style="background:transparent;border:2px solid #00A878;border-radius:10px;padding:1rem 1.2rem;margin-top:0.5rem;">
                  <div style="font-size:0.7rem;font-weight:700;color:#00A878;margin-bottom:0.4rem;">🧠 AI 추천 근거 (XAI){badge_r}</div>
                  <div style="font-size:0.81rem;color:#1A2332;line-height:1.8;">{reason_txt}</div>
                </div>""", unsafe_allow_html=True)

        # 승인된 연계기관 지도
        st.markdown('<div class="ml-sh">🗺️ 승인된 연계기관 찾기</div>', unsafe_allow_html=True)
        approved_agencies = [a for a in st.session_state['agency_list'] if a.get('상태')=='승인']
        if approved_agencies:
            ag_df2 = pd.DataFrame(approved_agencies)
            color_map3 = {'상담지원형':'#1B4F9B','정신의료형':'#E85D04','종합지원형':'#00A878'}
            fig_ag2 = px.scatter_mapbox(ag_df2,lat='위도',lon='경도',hover_name='기관명',hover_data={'위도':False,'경도':False,'주소':True,'전화':True,'유형':True},color='유형',color_discrete_map=color_map3,zoom=11,height=280)
            fig_ag2.update_traces(marker=dict(size=12))
            fig_ag2.update_layout(mapbox_style="open-street-map",margin=dict(l=0,r=0,t=0,b=0),font=dict(family='Noto Sans KR'),legend=dict(orientation="h",y=-0.12,bgcolor="rgba(255,255,255,0.9)"))
            st.plotly_chart(fig_ag2, use_container_width=True, config={'displayModeBar':False})
            st.caption(f"📍 현재 승인된 연계기관 {len(approved_agencies)}개소 · 출처: 안양과천교육지원청 Wee센터")

    with cr:
        st.markdown('<div class="ml-sh">📈 치료 진행 상태</div>', unsafe_allow_html=True)
        months=['1월','2월','3월','4월']
        np.random.seed(abs(hash(sel_student)) % 99)
        mood=[52]+[min(90,max(30,v+np.random.randint(-8,18))) for v in [60,68,74]]
        attn=[45]+[min(90,max(25,v+np.random.randint(-6,20))) for v in [55,63,70]]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=months,y=mood,mode='lines+markers',name='정서 안정도',line=dict(color='#1B4F9B',width=2.5),marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=months,y=attn,mode='lines+markers',name='집중력',line=dict(color='#00A878',width=2.5),marker=dict(size=9)))
        fig.update_layout(height=220,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor='white',plot_bgcolor='white',yaxis=dict(range=[0,100],gridcolor='#F0F4FA'),xaxis=dict(gridcolor='#F0F4FA'),font=dict(family='Noto Sans KR',size=11),legend=dict(orientation='h',y=-0.22))
        st.plotly_chart(fig, width='stretch', config={'displayModeBar':False})

        # ── AI 부모 행동 추천 ──
        st.markdown("""
        <div style="margin:1.5rem 0 1rem 0;">
          <div class="ai-section-title">🤖 AI 부모 행동 추천</div>
          <div class="ai-section-sub">자녀를 위해 지금 바로 실천할 수 있는 핵심 행동 가이드입니다</div>
          <div class="ai-gradient-bar"></div>
        </div>
        """, unsafe_allow_html=True)

        p_cache_key = get_ai_cache_key("parent", sel_student, stu['유형'])
        if p_cache_key not in st.session_state:
            with st.spinner("🤖 AI가 맞춤 행동 가이드를 생성 중..."):
                parent_msgs, is_llm_p = ai_parent_message(stu_dict)
            st.session_state[p_cache_key] = (parent_msgs, is_llm_p)
        else:
            parent_msgs, is_llm_p = st.session_state[p_cache_key]

        render_ai_card(
            parent_msgs,
            f"{stu['연계기관']} 진료 기록 기반 · {datetime.now().strftime('%Y.%m.%d')} AI 분석",
            is_llm_p,
            "💌 부모님께 드리는 추천 행동 가이드",
            border_color="#1B4F9B", label_color="#1B4F9B"
        )

        if st.button("🔄 AI 추천 재생성", key="regen_parent"):
            if p_cache_key in st.session_state:
                del st.session_state[p_cache_key]
            st.rerun()

        st.markdown('<div class="ml-sh">💬 아이의 주간 체크인 현황</div>', unsafe_allow_html=True)
        checkins=get_checkin_data(sel_student)
        tbl_ci=""
        for c in checkins[-5:]:
            mood_emoji=['','😞','😐','🙂','😊','😄'][c['기분점수']]
            att_color='#00A878' if c['출석'] else '#D62828'
            tbl_ci+=f"<tr><td>{c['주차']}</td><td>{c['날짜']}</td><td>{mood_emoji} {c['기분점수']}점</td><td style='color:{att_color};font-weight:600;'>{'출석' if c['출석'] else '결석'}</td><td style='font-size:0.7rem;color:#5A6A7E;'>{c['메모']}</td></tr>"
        st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><table class="data-table"><thead><tr><th>주차</th><th>날짜</th><th>기분</th><th>출석</th><th>메모</th></tr></thead><tbody>{tbl_ci}</tbody></table></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: 병원 관계자
# ═══════════════════════════════════════════════════════════════════════════════
def page_hospital():
    org_list = sorted(HOSPITAL_PASSWORDS.keys())
    my_org = st.selectbox("🏥 소속 기관 선택", org_list, key="hosp_org_sel")
    auth_key = f"hosp_{my_org}"
    correct_hosp_pw = HOSPITAL_PASSWORDS.get(my_org, "0000")
    if not st.session_state.get(f'auth_{auth_key}'):
        auth_screen(f"🏥 {my_org} 관계자 전용 접속",
                    f"소속 기관 인증이 필요합니다 · 데모 비밀번호: {correct_hosp_pw}",
                    auth_key, correct_hosp_pw)
        return

    st.markdown(f'<h2 style="color:#1B4F9B;font-size:1.4rem;font-weight:900;margin-bottom:0.2rem;">🏥 병원 관계자 포털</h2><p style="color:#5A6A7E;font-size:0.79rem;margin-bottom:1rem;">{my_org} · 환자 방문 기록 · 소견서 입력 · 치료 종결 · 연계기관 신청</p>', unsafe_allow_html=True)

    my_df=df[df['연계기관']==my_org]
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">🧑‍⚕️</div><div class="ml-metric-value">{len(my_df)}</div><div class="ml-metric-label">연계 학생 수</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">✅</div><div class="ml-metric-value">{len(my_df[my_df["종결서류"].isin(["종결","병원종결"])])}</div><div class="ml-metric-label">종결 학생</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="ml-metric"><div class="ml-metric-icon">💰</div><div class="ml-metric-value">{int(my_df["사용금액"].sum())//10000}만</div><div class="ml-metric-label">총 사용금액</div></div>', unsafe_allow_html=True)

    if len(my_df)>0:
        vr1,vr2=st.columns(2)
        with vr1:
            rc=my_df[my_df['종결서류']=='진행중']['위험도'].value_counts().reset_index(); rc.columns=['위험도','수']
            if len(rc)>0:
                fig_rc=px.pie(rc,names='위험도',values='수',hole=0.5,height=170,color='위험도',color_discrete_map={'높음':'#D62828','보통':'#E85D04','낮음':'#00A878'})
                fig_rc.update_layout(margin=dict(l=0,r=0,t=25,b=0),paper_bgcolor='white',font=dict(family='Noto Sans KR',size=9),title=dict(text='진행중 학생 위험도',font=dict(size=10),x=0.5),legend=dict(font=dict(size=8)))
                fig_rc.update_traces(textinfo='percent+label',textfont_size=8)
                st.plotly_chart(fig_rc, width='stretch', config={'displayModeBar':False})
        with vr2:
            months=['1월','2월','3월','4월']
            visit_counts=[random.randint(5,20) for _ in months]
            fig_vc=go.Figure(go.Bar(x=months,y=visit_counts,marker_color='#1B4F9B',text=visit_counts,textposition='outside'))
            fig_vc.update_layout(height=170,margin=dict(l=0,r=0,t=25,b=0),paper_bgcolor='white',plot_bgcolor='white',font=dict(family='Noto Sans KR',size=10),title=dict(text='월별 방문 횟수',font=dict(size=10),x=0.5),yaxis=dict(gridcolor='#F0F4FA'),xaxis=dict(gridcolor='#F0F4FA'))
            st.plotly_chart(fig_vc, width='stretch', config={'displayModeBar':False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4=st.tabs(["📝 방문 기록 입력","📋 소견서 & AI 발송","📤 종결 보고서","📄 연계기관 신청"])

    with tab1:
        stu_list=my_df['성명'].tolist() if len(my_df)>0 else ['없음']
        with st.form("visit_form"):
            stu_sel2=st.selectbox("학생 선택",stu_list)
            col_a,col_b=st.columns(2)
            with col_a: v_date=st.date_input("방문일",value=datetime.now())
            with col_b: v_type=st.selectbox("진료 유형",["초진","재진","심리검사","집단치료","상담"])
            v_amount=st.number_input("청구 금액 (원)",min_value=0,max_value=500000,value=80000,step=10000)
            v_note=st.text_area("진료 기록 메모",height=90,placeholder="증상 변화, 특이사항, 보호자 상담 내용...")
            share_both=st.checkbox("✅ 교사 및 학부모 행동 추천 AI 생성 & 발송")
            if st.form_submit_button("💾 방문 기록 저장"):
                with st.spinner("저장 중..."): time.sleep(1)
                st.success(f"✅ {stu_sel2} 방문 기록 저장 완료 ({v_type} · {v_amount:,}원)")
                if share_both:
                    st.info("🤖 AI가 교사용 교실 지원 가이드 & 학부모 행동 추천 메시지를 생성하여 발송 완료했습니다.")
        st.markdown('<div class="ml-sh">📅 최근 방문 이력</div>', unsafe_allow_html=True)
        recent_visits=[{'날짜':'2026-04-15','학생':'김지원','유형':'재진','금액':80000,'메모':'정서 안정도 호전, 집중력 향상'},{'날짜':'2026-04-08','학생':'이바다','유형':'심리검사','금액':150000,'메모':'ADHD 척도 재검사 실시'},{'날짜':'2026-04-01','학생':'박구름','유형':'상담','금액':80000,'메모':'가정-학교 연계 상담 진행'}]
        tbl_v=""
        for v in recent_visits:
            tbl_v+=f"<tr><td>{v['날짜']}</td><td><b>{v['학생']}</b></td><td>{v['유형']}</td><td>{v['금액']:,}원</td><td style='font-size:0.72rem;color:#5A6A7E;'>{v['메모']}</td></tr>"
        st.markdown(f"""<div class="ml-card" style="padding:0.8rem;"><table class="data-table"><thead><tr><th>날짜</th><th>학생</th><th>유형</th><th>금액</th><th>메모</th></tr></thead><tbody>{tbl_v}</tbody></table></div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="ml-sh">🔬 소견서 작성 → AI 교사·학부모 가이드 생성</div>', unsafe_allow_html=True)
        cl2,cr2=st.columns(2)
        with cl2:
            st.markdown('<div class="ml-card ml-card-primary"><b>병원 전문 소견서 입력</b><br><br>', unsafe_allow_html=True)
            opinion_stu=st.selectbox("대상 학생",stu_list,key="op_stu")
            opinion_txt=st.text_area("소견 내용",placeholder="예: DSM-5 기준 ADHD 혼합형 진단...",height=140,label_visibility='collapsed')
            severity=st.select_slider("증상 심각도",options=["경미","보통","심각","위기"])
            send_target=st.multiselect("발송 대상",["담임교사","학부모","위센터"],default=["담임교사","학부모"])
            if st.button("🤖 AI 가이드 생성 & 발송",key="op_send"):
                if opinion_txt.strip():
                    with st.spinner("AI가 교사·학부모 맞춤 가이드 생성 중..."): time.sleep(1.5)
                    st.session_state['opinion_done']=True; st.session_state['opinion_target']=send_target
                else: st.warning("소견 내용을 입력해주세요.")
            st.markdown('</div>', unsafe_allow_html=True)
        with cr2:
            if st.session_state.get('opinion_done'):
                targets=st.session_state.get('opinion_target',[])
                if "담임교사" in targets:
                    st.markdown("""<div class="ai-box" style="margin-bottom:0.7rem;">
                      <div class="ai-label" style="color:#1B4F9B;">📋 담임교사용 교실 지원 가이드</div>
                      <div class="ai-line">▸ 좌석: 교사 시야 정면, 창가 1칸 안쪽 / 이동 동선 단순화</div>
                      <div class="ai-line">▸ Do: 짧은 지시 → 즉각 칭찬</div>
                      <div class="ai-line">▸ Don't: 장시간 단독 집중 과제 / 전체 앞 지적 자제</div>
                    </div>""", unsafe_allow_html=True)
                if "학부모" in targets:
                    st.markdown("""<div class="ai-box-green">
                      <div class="ai-label" style="color:#00A878;">💌 학부모 행동 추천</div>
                      <div class="ai-line">▸ 규칙적인 수면·식사 루틴 유지 (10시 취침 권장)</div>
                      <div class="ai-line">▸ 처방 약물 정시 복용 / 임의 중단 금지</div>
                      <div class="ai-line">▸ 다음 진료 전 변화(식욕·수면·기분) 메모</div>
                    </div>""", unsafe_allow_html=True)
                for t in targets: st.success(f"✅ {t}에게 발송 완료")
            else:
                st.markdown('<div class="ml-card" style="text-align:center;color:#5A6A7E;padding:2rem;"><div style="font-size:1.8rem;margin-bottom:0.5rem;">🔄</div><div style="font-size:0.8rem;">소견서를 입력하고 AI 생성을 실행하면<br>교사·학부모 맞춤 가이드가 생성됩니다</div></div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="ml-sh">📋 치료 종결 보고서 작성 & 제출</div>', unsafe_allow_html=True)
        active_stus=my_df[my_df['종결서류']=='진행중']['성명'].tolist()
        with st.form("end_report_form"):
            end_stu=st.selectbox("종결 처리 학생",active_stus if active_stus else ['없음'])
            col_e1,col_e2=st.columns(2)
            with col_e1: end_why=st.selectbox("종결 사유",["목표 달성 종결","보호자 요청","전학/이사","병원종결","기타"])
            with col_e2: end_date=st.date_input("종결일",value=datetime.now())
            end_note=st.text_area("종결 소견 요약",height=90,placeholder="치료 기간 중 변화 및 향후 권고사항...")
            st.file_uploader("종결 보고서 파일 첨부",type=['pdf','docx'])
            if st.form_submit_button("📤 종결 보고서 제출 & 정산 신청"):
                if end_stu != '없음':
                    with st.spinner("병원-학교-교육청 서류 자동 생성 중..."): time.sleep(2)
                    new_report={'기관':my_org,'학생':end_stu,'종결일':str(end_date),'사유':end_why,'소견':end_note if end_note else "(소견 없음)",'상태':'검토중'}
                    st.session_state['closure_reports'].append(new_report)
                    st.success("✅ 종결 보고서 제출 완료 — 위센터 주무관에게 자동 전달되었습니다.")
                    st.markdown('<div class="ml-card ml-card-accent" style="font-size:0.79rem;">✅ 담임교사 종결 알림 전송<br>✅ 위센터 정산 서류 자동 생성<br>✅ 전자결재 연동 완료</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="ml-sh">📄 연계기관 지정 신청서 작성</div>', unsafe_allow_html=True)
        st.markdown("""<div class="ml-card ml-card-primary" style="font-size:0.78rem;margin-bottom:1rem;"><b>📌 안내</b><br>2026 학생 마음바우처 지원사업 연계기관 지정 신청서를 작성하세요.<br>작성 후 제출하면 <b>위(Wee)센터 주무관</b>의 검토 후 승인 처리됩니다.<br><span style="color:#5A6A7E;font-size:0.72rem;">출처: 경기도교육지원청 Wee센터 (인천교육지원청 양식 참고)</span></div>""", unsafe_allow_html=True)
        already_applied = any(a.get('기관명')==my_org for a in st.session_state['hosp_applications'])
        already_in_list = any(a.get('기관명')==my_org for a in st.session_state['agency_list'])
        if already_in_list:
            existing = next(a for a in st.session_state['agency_list'] if a.get('기관명')==my_org)
            s_color={'승인':'#00A878','검토중':'#E85D04','대기':'#94A3B8','반려':'#D62828'}.get(existing.get('상태','대기'),'#94A3B8')
            st.markdown(f"""<div class="ml-card ml-card-accent"><div style="font-weight:700;font-size:0.88rem;color:#00A878;">✅ 이미 등록된 기관입니다</div><div style="font-size:0.79rem;margin-top:0.4rem;">기관명: <b>{my_org}</b> &nbsp;|&nbsp; 상태: <span style="color:{s_color};font-weight:700;">{existing.get('상태','')}</span>{f" &nbsp;|&nbsp; 승인일: {existing.get('승인일','')}" if existing.get('승인일') else ""}</div></div>""", unsafe_allow_html=True)
        elif already_applied:
            st.info(f"✅ {my_org}의 신청서가 이미 제출되었습니다. 주무관 검토 중입니다.")
        else:
            with st.form("agency_apply_form"):
                st.markdown("**기관 기본 정보**")
                f_a1,f_a2=st.columns(2)
                with f_a1:
                    f_name=st.text_input("기관(법인)명",value=my_org); f_ceo=st.text_input("대표자 성명"); f_biz=st.text_input("사업자등록번호")
                with f_a2:
                    f_addr=st.text_input("주소"); f_tel=st.text_input("대표번호"); f_email=st.text_input("기관 이메일")
                st.markdown("**직원 현황**")
                f_b1,f_b2,f_b3=st.columns(3)
                with f_b1: f_staff1=st.text_input("직원명 1",placeholder="홍길동")
                with f_b2: f_qual1=st.text_input("자격증 1",placeholder="상담심리사 1급")
                with f_b3: f_role1=st.text_input("담당업무 1",placeholder="센터장")
                st.markdown("**진료/상담 정보**")
                f_c1,f_c2=st.columns(2)
                with f_c1:
                    f_type=st.selectbox("지원 유형",["상담지원형","정신의료형","종합지원형"])
                    f_hours=st.text_input("운영 시간",placeholder="평일 09:00-18:00")
                with f_c2:
                    f_wait=st.text_input("평균 대기 시간",placeholder="약 3일")
                    f_appt=st.text_input("예약 방법",placeholder="유선, 네이버 예약")
                f_services=st.text_area("검사·치료 내용 (금액 포함)",height=80,placeholder="예) 개인 심리상담 회기당 80,000원")
                st.markdown("**동의 사항**")
                agree1=st.checkbox("제3자(학생, 학부모, 학교 등)에게 기관 정보 제공에 동의합니다")
                agree2=st.checkbox("치료·상담 비용은 월 1회 사후입금 방식에 동의합니다")
                agree3=st.checkbox("연 1회 Wee센터의 현장점검에 동의합니다")
                if st.form_submit_button("📤 연계기관 지정 신청서 제출"):
                    if not (agree1 and agree2 and agree3):
                        st.error("모든 동의 사항에 체크해주세요.")
                    elif not f_name or not f_addr or not f_tel:
                        st.error("기관명, 주소, 연락처는 필수 입력 항목입니다.")
                    else:
                        new_app={'기관명':f_name,'지역':f_addr[:4] if f_addr else '미입력','유형':f_type,'주소':f_addr,'전화':f_tel,'위도':37.39+random.uniform(-0.02,0.02),'경도':126.96+random.uniform(-0.02,0.02),'상태':'대기','승인일':'','신청일':datetime.now().strftime('%Y-%m-%d')}
                        st.session_state['hosp_applications'].append(new_app)
                        st.success("✅ 연계기관 지정 신청서가 제출되었습니다. 위(Wee)센터 주무관에서 검토 후 승인 처리됩니다.")
                        st.info("📍 승인 후 연계기관 지도에 자동으로 표시됩니다.")
                        st.rerun()

# ── Route ─────────────────────────────────────────────────────────────────────
if role == "담임교사":       page_teacher()
elif role == "위센터 주무관": page_wee()
elif role == "학부모":       page_parent()
elif role == "병원 관계자":   page_hospital()
