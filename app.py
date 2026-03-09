# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyogrio
import requests
import streamlit as st
from rapidfuzz import fuzz, process
from shapely.geometry import (
    Point,
    box,
    LineString,
    MultiLineString,
    GeometryCollection,
    Polygon,
    MultiPolygon,
)
from streamlit_folium import st_folium
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go

try:
    from zoneinfo import ZoneInfo
    SEOUL_TZ = ZoneInfo("Asia/Seoul")
except Exception:
    SEOUL_TZ = None


# =========================================================
# 0) 기본 설정
# =========================================================
st.set_page_config(
    page_title="대중교통 접근성 취약 진단 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
from pathlib import Path

GRID_ID_COL = "GRID_500M_"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data")))

DATA_DIR = DATA_ROOT / "02_routing" / "01_intercity" / "02_500m"
GRID_PATH = DATA_ROOT / "00_grid" / "500m.gpkg"

OD_PATH = DATA_DIR / "od_500m.parquet"
BESTCASE_PATH = DATA_DIR / "dashboard_baseset_bestcase.parquet"
GRID_DIAG_GPQ = DATA_DIR / "grid_500m_baseset_diag.geoparquet"
FACILITY_GPQ = DATA_DIR / "all_facilities.geoparquet"
MAP_HEIGHT = 780

# 서울특별시 성동구 왕십리로 222 부근
DEFAULT_CENTER = [37.5575, 127.0450]
DEFAULT_ZOOM = 14
DEFAULT_BOUNDS = (127.036, 37.551, 127.054, 37.564)

BUNDLE_THRESHOLD = 15.0

# Traveler profile 구조 취약 기준
FS_EPS = 0.01
FD_GAP_THRESHOLD = 5.0
TC_MEAN_THRESHOLD = 5.0
TC_STD_THRESHOLD = 5.0

MAX_GRID_RENDER = 8000
MAX_POINT_RENDER = 1800

GRID_RENDER_MIN_ZOOM = 12
POINT_RENDER_MIN_ZOOM = 14

DEFAULT_VWORLD_API_KEY = "407D633D-7C2A-3D60-8EFF-4735DE5A4030"


# =========================================================
# 1) 스타일
# =========================================================
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #4f7de8 !important;
        color: white !important;
        border: 1px solid #4571d3 !important;
        border-radius: 10px !important;
        padding: 0.52rem 0.95rem !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #426fd8 !important;
        border-color: #385fbb !important;
        color: white !important;
    }

    .stButton > button:focus,
    .stFormSubmitButton > button:focus {
        box-shadow: 0 0 0 0.2rem rgba(79,125,232,0.20) !important;
    }

    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stFormSubmitButton > button {
        width: 100%;
    }

    .metric-card {
        background: #f8fbff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        min-height: 108px;
    }

    .soft-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }

    .hint-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2) activity / color
# =========================================================
STANDARD_SET = ["library", "park", "m1", "ms", "grocery", "public", "pharmacy"]

ACTIVITY_META = {
    "library":  {"label": "도서관",               "cov_thr": 15},
    "park":     {"label": "공원",                 "cov_thr": 15},
    "m1":       {"label": "기초의료",             "cov_thr": 10},
    "ms":       {"label": "전문의료(통합)",        "cov_thr": 15},
    "m2":       {"label": "주요 외래과목",         "cov_thr": 15},
    "m3":       {"label": "감각·피부·여성·비뇨 외래", "cov_thr": 15},
    "m4":       {"label": "정신건강",             "cov_thr": 15},
    "m5":       {"label": "치과",                 "cov_thr": 15},
    "m6":       {"label": "한의원",               "cov_thr": 15},
    "m7":       {"label": "외과/응급/특수",        "cov_thr": 15},
    "grocery":  {"label": "식료품",               "cov_thr": 10},
    "public":   {"label": "공공서비스",           "cov_thr": 15},
    "pharmacy": {"label": "약국",                 "cov_thr": 10},
    "elderly":  {"label": "노인여가",             "cov_thr": 15},
    "nursery":  {"label": "보육",                 "cov_thr": 15},
    "primary":  {"label": "초등학교",             "cov_thr": 15},
    "junior":   {"label": "중학교",               "cov_thr": 15},
    "high":     {"label": "고등학교",             "cov_thr": 15},
}

PROFILE_POOL = [
    "library", "park", "m1", "m2", "m3", "m4", "m5", "m6", "m7",
    "grocery", "public", "pharmacy", "elderly", "nursery",
    "primary", "junior", "high"
]

DIAG_COLOR = {
    "양호": "#d9d9d9",
    "F(s)": "#fca5a5",
    "F(d)": "#fcd34d",
    "T(c)": "#93c5fd",
    "F(s)+F(d)": "#fb7185",
    "F(s)+T(c)": "#a78bfa",
    "F(d)+T(c)": "#34d399",
    "F(s)+F(d)+T(c)": "#f97316",
}

TS_HATCH_COLOR = "#ef4444"

FACILITY_KIND_COLOR = {
    "의료기관": "#ef4444",
    "약국": "#f97316",
    "식료품": "#eab308",
    "도서관": "#22c55e",
    "공원": "#10b981",
    "공공서비스": "#3b82f6",
    "노인여가": "#8b5cf6",
    "보육": "#ec4899",
    "초등학교": "#14b8a6",
    "중학교": "#0ea5e9",
    "고등학교": "#6366f1",
}

M_LABEL = {
    "m1": "기초의료시설",
    "m2": "주요 외래과목",
    "m3": "감각·피부·여성·비뇨 외래",
    "m4": "정신건강",
    "m5": "치과",
    "m6": "한의원",
}

MED_GROUP_MAP_RAW = {
    "가정의학과": "m1", "내과": "m1", "소아청소년과": "m1",
    "정형외과": "m2", "재활의학과": "m2", "마취통증의학과": "m2",
    "안과": "m3", "이비인후과": "m3", "피부과": "m3", "비뇨의학과": "m3", "신경과": "m3", "산부인과": "m3",
    "정신건강의학과": "m4",
    "치과": "m5", "통합치의학과": "m5", "소아치과": "m5", "치과교정과": "m5", "치과보존과": "m5",
    "치과보철과": "m5", "치주과": "m5", "구강내과": "m5",
    "사상체질과": "m6", "침구과": "m6", "한방내과": "m6", "한방부인과": "m6",
    "한방소아과": "m6", "한방신경정신과": "m6", "한방안·이비인후·피부과": "m6",
    "한방재활의학과": "m6",
}
# 정규화 후 맵
MED_GROUP_MAP = {re.sub(r"\s+", "", k): v for k, v in MED_GROUP_MAP_RAW.items()}


# =========================================================
# 3) 주소 보정 사전
# =========================================================
ADDRESS_ALIAS_MAP = {
    "서울": "서울특별시",
    "서울시": "서울특별시",
    "서울 특별시": "서울특별시",
    "부산": "부산광역시",
    "부산시": "부산광역시",
    "대구": "대구광역시",
    "대구시": "대구광역시",
    "인천": "인천광역시",
    "인천시": "인천광역시",
    "광주": "광주광역시",
    "광주시": "광주광역시",
    "대전": "대전광역시",
    "대전시": "대전광역시",
    "울산": "울산광역시",
    "울산시": "울산광역시",
    "세종": "세종특별자치시",
    "세종시": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
    "제주도": "제주특별자치도",
}

REPRESENTATIVE_QUERY_MAP = {
    "서울특별시": "서울특별시청",
    "부산광역시": "부산광역시청",
    "대구광역시": "대구광역시청",
    "인천광역시": "인천광역시청",
    "광주광역시": "광주광역시청",
    "대전광역시": "대전광역시청",
    "울산광역시": "울산광역시청",
    "세종특별자치시": "세종특별자치시청",
    "경기도": "경기도청",
    "강원특별자치도": "강원특별자치도청",
    "충청북도": "충청북도청",
    "충청남도": "충청남도청",
    "전북특별자치도": "전북특별자치도청",
    "전라남도": "전라남도청",
    "경상북도": "경상북도청",
    "경상남도": "경상남도청",
    "제주특별자치도": "제주특별자치도청",
}
FUZZY_CANDIDATES = sorted(set(list(ADDRESS_ALIAS_MAP.keys()) + list(ADDRESS_ALIAS_MAP.values())))


# =========================================================
# 4) session state
# =========================================================
def init_state():
    defaults = {
        "address_input": "",
        "selected_from_id": None,
        "selected_grid_bounds": None,
        "map_bounds": DEFAULT_BOUNDS,
        "map_center": DEFAULT_CENTER,
        "map_zoom": DEFAULT_ZOOM,
        "analysis_mode": "표준세트",
        "selected_activities": STANDARD_SET.copy(),
        "set_confirmed": False,
        "analysis_requested": False,
        "selected_time": None,
        "focus_bundle_to_id": None,
        "current_lon": None,
        "current_lat": None,
        "search_note": None,
        "search_matched_address": None,
        "last_map_click_key": None,
        "pending_fit_bounds": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =========================================================
# 5) 공통 유틸
# =========================================================
def now_kst():
    if SEOUL_TZ is not None:
        return datetime.now(SEOUL_TZ)
    return datetime.now()

def act_label(act: str) -> str:
    return ACTIVITY_META.get(act, {}).get("label", act)

def act_labels(acts: List[str]) -> List[str]:
    return [act_label(a) for a in acts]

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", "", str(x).strip())

def parse_facility_type_groups(s: str) -> List[str]:
    if not s:
        return []
    parts = [x.strip() for x in re.split(r"[,\n;/|]+", str(s)) if x.strip()]
    out = []
    for p in parts:
        norm = normalize_text(p)
        if norm in MED_GROUP_MAP:
            out.append(MED_GROUP_MAP[norm])
    return sorted(set(out))

def get_vworld_api_key() -> str:
    try:
        if "VWORLD_API_KEY" in st.secrets:
            return st.secrets["VWORLD_API_KEY"]
    except Exception:
        pass
    return DEFAULT_VWORLD_API_KEY

@st.cache_data(show_spinner=False)
def get_time_cols() -> List[str]:
    schema = pl.scan_parquet(OD_PATH).collect_schema()
    cols = list(schema.keys())
    return sorted([c for c in cols if re.fullmatch(r"pt\d{2}", c)], key=lambda x: int(x[2:]))

@st.cache_data(show_spinner=False)
def get_available_od_cols() -> List[str]:
    schema = pl.scan_parquet(OD_PATH).collect_schema()
    return list(schema.keys())

def build_diag_label(fs: bool, fd: bool, tc: bool) -> str:
    parts = []
    if fs:
        parts.append("F(s)")
    if fd:
        parts.append("F(d)")
    if tc:
        parts.append("T(c)")
    return "+".join(parts) if parts else "양호"

def build_diag_color(label: str) -> str:
    return DIAG_COLOR.get(label, "#d9d9d9")

def normalize_bounds_from_st_folium(bounds_obj) -> Optional[Tuple[float, float, float, float]]:
    if bounds_obj is None:
        return None
    if isinstance(bounds_obj, dict):
        if "_southWest" in bounds_obj and "_northEast" in bounds_obj:
            sw = bounds_obj["_southWest"]
            ne = bounds_obj["_northEast"]
            return (sw["lng"], sw["lat"], ne["lng"], ne["lat"])
    if isinstance(bounds_obj, list) and len(bounds_obj) == 2:
        sw, ne = bounds_obj
        return (sw[1], sw[0], ne[1], ne[0])
    return None

def bounds_to_fit(bounds: Tuple[float, float, float, float], pad_ratio: float = 0.12):
    minx, miny, maxx, maxy = bounds
    dx = max(maxx - minx, 0.001)
    dy = max(maxy - miny, 0.001)
    return [
        [miny - dy * pad_ratio, minx - dx * pad_ratio],
        [maxy + dy * pad_ratio, maxx + dx * pad_ratio],
    ]

def combine_bounds(bounds_list: List[Optional[Tuple[float, float, float, float]]]) -> Optional[Tuple[float, float, float, float]]:
    valid = [b for b in bounds_list if b is not None]
    if not valid:
        return None
    minx = min(b[0] for b in valid)
    miny = min(b[1] for b in valid)
    maxx = max(b[2] for b in valid)
    maxy = max(b[3] for b in valid)
    return (minx, miny, maxx, maxy)

def parse_time_to_minutes(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    m = re.match(r"^(\d{1,2})(?::(\d{1,2}))?(?::\d{1,2})?$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    hh = hh % 24
    return hh * 60 + mm

def detect_day_schedule(row: pd.Series) -> Tuple[str, str]:
    wd = now_kst().weekday()
    day_prefix = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][wd]
    return f"{day_prefix}_from", f"{day_prefix}_to"

def infer_open_status_and_hours(row: pd.Series) -> Tuple[str, str]:
    cols = {c.lower(): c for c in row.index}
    status_col = next((c for c in row.index if c.lower() in {"status", "current_status", "운영여부", "영업상태", "is_open", "open_yn"}), None)
    explicit_status = str(row.get(status_col, "")).strip() if status_col else ""

    hours_col = next((c for c in row.index if c.lower() in {"opening_hours", "operating_hours", "hours", "운영시간", "영업시간"}), None)
    hours_text = str(row.get(hours_col, "")).strip() if hours_col else ""

    day_from, day_to = detect_day_schedule(row)
    day_from_real = cols.get(day_from.lower())
    day_to_real = cols.get(day_to.lower())

    bf = cols.get("break_from")
    bt = cols.get("break_to")

    if day_from_real and day_to_real:
        smin = parse_time_to_minutes(row.get(day_from_real))
        emin = parse_time_to_minutes(row.get(day_to_real))

        if smin is None or emin is None or (smin == 0 and emin == 0):
            if hours_text:
                return ("미운영 또는 정보없음", hours_text)
            return ("미운영 또는 정보없음", "오늘 운영시간 정보 없음")

        hours_str = f"오늘 {row.get(day_from_real)} ~ {row.get(day_to_real)}"
        if bf and bt and str(row.get(bf, "")).strip() and str(row.get(bt, "")).strip():
            hours_str += f" / 휴게 {row.get(bf)} ~ {row.get(bt)}"

        now_dt = now_kst()
        now_min = now_dt.hour * 60 + now_dt.minute

        if smin <= now_min < emin:
            if bf and bt:
                bfm = parse_time_to_minutes(row.get(bf))
                btm = parse_time_to_minutes(row.get(bt))
                if bfm is not None and btm is not None and bfm <= now_min < btm:
                    return ("휴게시간", hours_str)
            return ("운영중", hours_str)
        return ("미운영", hours_str)

    if explicit_status:
        return (explicit_status, hours_text or "운영시간 정보 없음")
    if hours_text:
        return ("운영시간 참고", hours_text)
    return ("정보없음", "운영시간 정보 없음")

def parse_json_list_like(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    parsed = None
    try:
        parsed = json.loads(s)
    except Exception:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return [x.strip() for x in re.split(r"[,+/|]", s) if x.strip()]

def normalize_diag_token(s: str) -> str:
    x = str(s).strip().lower()
    mapping = {
        "f(s)": "F(s)",
        "fs": "F(s)",
        "f_s": "F(s)",
        "f(d)": "F(d)",
        "fd": "F(d)",
        "f_d": "F(d)",
        "t(c)": "T(c)",
        "tc": "T(c)",
        "t_c": "T(c)",
        "t(s)-v": "T(s)",
        "t(s)": "T(s)",
        "none": "양호",
    }
    return mapping.get(x, s)

def has_ts_label(labels: list[str]) -> bool:
    norm = [normalize_diag_token(x) for x in labels]
    return "T(s)" in norm


# =========================================================
# 6) precomputed 리소스
# =========================================================
@st.cache_resource(show_spinner=False)
def load_std_bestcase_resource() -> pd.DataFrame:
    if not BESTCASE_PATH.exists():
        return pd.DataFrame()
    df = pl.read_parquet(BESTCASE_PATH).to_pandas()
    df["from_id"] = df["from_id"].astype(str)
    return df.set_index("from_id", drop=False)

@st.cache_resource(show_spinner=False)
def load_full_grid_diag_resource() -> gpd.GeoDataFrame:
    if not GRID_DIAG_GPQ.exists():
        return gpd.GeoDataFrame()
    gdf = gpd.read_parquet(GRID_DIAG_GPQ)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    _ = gdf.sindex
    return gdf

@st.cache_resource(show_spinner=False)
def load_full_facility_resource() -> gpd.GeoDataFrame:
    if not FACILITY_GPQ.exists():
        return gpd.GeoDataFrame()
    gdf = gpd.read_parquet(FACILITY_GPQ)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    _ = gdf.sindex
    return gdf


# =========================================================
# 7) 주소 검색
# =========================================================
def normalize_address_input(q: str) -> str:
    s = str(q).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_address_token(q: str) -> str:
    s = normalize_address_input(q)
    return ADDRESS_ALIAS_MAP.get(s, s)

def fuzzy_correct_address_token(q: str, score_cutoff: int = 82) -> tuple[str, Optional[str], int]:
    s = normalize_address_input(q)
    matched = process.extractOne(s, FUZZY_CANDIDATES, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
    if matched is None:
        return s, None, 0
    matched_text, score, _ = matched
    corrected = ADDRESS_ALIAS_MAP.get(matched_text, matched_text)
    return corrected, matched_text, int(score)

def geocode_vworld_raw(query: str, addr_type: str) -> Tuple[Optional[Tuple[float, float, str]], Optional[str]]:
    api_key = get_vworld_api_key()
    base_url = "https://api.vworld.kr/req/address"
    params = {
        "service": "address",
        "request": "getcoord",
        "version": "2.0",
        "crs": "epsg:4326",
        "address": query,
        "refine": "true",
        "simple": "false",
        "format": "json",
        "type": addr_type,
        "key": api_key,
    }
    try:
        r = requests.get(base_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"VWORLD_HTTP_ERROR:{e}"

    resp = data.get("response", {})
    status = resp.get("status", "")
    if status != "OK":
        err = resp.get("error", {})
        code = err.get("code", "")
        text = err.get("text", "")
        if code:
            return None, f"{code}:{text}"
        return None, f"VWORLD_STATUS:{status}"

    result = resp.get("result", {})
    point = result.get("point", {})
    refined = resp.get("refined", {})

    x = point.get("x")
    y = point.get("y")
    if x is None or y is None:
        return None, "VWORLD_NO_POINT"

    matched = refined.get("text", query)
    return (float(x), float(y), matched), None

def try_geocode_vworld(query: str) -> Tuple[Optional[Tuple[float, float, str]], Optional[str]]:
    out, err1 = geocode_vworld_raw(query, "ROAD")
    if out is not None:
        return out, None
    out, err2 = geocode_vworld_raw(query, "PARCEL")
    if out is not None:
        return out, None
    return None, err2 or err1

def geocode_nominatim(query: str) -> Optional[Tuple[float, float, str]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": "pt-access-dashboard/0.1"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lon, lat, data[0].get("display_name", query)

def geocode_address(query: str) -> Optional[Tuple[float, float, str, str]]:
    q = normalize_address_input(query)
    if not q:
        return None

    m = re.match(r"^\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*$", q)
    if m:
        lat = float(m.group(1))
        lon = float(m.group(2))
        return lon, lat, "manual_latlon", "manual"

    candidates: List[Tuple[str, str]] = [(q, "exact")]

    normalized = canonicalize_address_token(q)
    if normalized != q:
        candidates.append((normalized, f"normalized:{q}->{normalized}"))

    fuzzy_corrected, _, score = fuzzy_correct_address_token(q)
    if fuzzy_corrected != q:
        candidates.append((fuzzy_corrected, f"fuzzy:{q}->{fuzzy_corrected}({score})"))

    broad_base = canonicalize_address_token(q)
    if broad_base in REPRESENTATIVE_QUERY_MAP:
        rep = REPRESENTATIVE_QUERY_MAP[broad_base]
        candidates.append((rep, f"representative:{broad_base}->{rep}"))

    if fuzzy_corrected in REPRESENTATIVE_QUERY_MAP:
        rep = REPRESENTATIVE_QUERY_MAP[fuzzy_corrected]
        candidates.append((rep, f"representative:{fuzzy_corrected}->{rep}"))

    seen = set()
    dedup = []
    for cand, note in candidates:
        if cand not in seen:
            dedup.append((cand, note))
            seen.add(cand)

    last_err = None
    for cand, note in dedup:
        out, err = try_geocode_vworld(cand)
        if out is not None:
            lon, lat, matched = out
            return lon, lat, matched, note
        last_err = err

    for cand, note in dedup:
        try:
            out = geocode_nominatim(cand)
            if out is not None:
                lon, lat, matched = out
                return lon, lat, matched, note
        except Exception:
            pass

    if last_err:
        st.warning(f"브이월드 검색 실패 사유: {last_err}")
    return None


# =========================================================
# 8) grid / geoparquet 읽기
# =========================================================
@st.cache_data(show_spinner=False)
def read_grid_by_point(lon: float, lat: float) -> Optional[gpd.GeoDataFrame]:
    layer_name = pyogrio.list_layers(GRID_PATH)[0][0]
    info = pyogrio.read_info(GRID_PATH, layer=layer_name)

    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(info["crs"])
    x, y = pt.iloc[0].x, pt.iloc[0].y

    gdf = pyogrio.read_dataframe(
        GRID_PATH,
        layer=layer_name,
        columns=[GRID_ID_COL],
        bbox=(x - 1200, y - 1200, x + 1200, y + 1200),
        use_arrow=True,
    )
    if gdf.empty:
        return None

    cand = gdf[gdf.geometry.intersects(pt.iloc[0])].copy()
    if cand.empty:
        return None

    cand[GRID_ID_COL] = cand[GRID_ID_COL].astype(str)
    cand = cand.sort_values(GRID_ID_COL).head(1).copy()
    return cand.to_crs(4326)

@st.cache_data(show_spinner=False)
def read_grid_by_id(grid_id: str) -> Optional[gpd.GeoDataFrame]:
    escaped = grid_id.replace("'", "''")
    layer_name = pyogrio.list_layers(GRID_PATH)[0][0]

    try:
        gdf = pyogrio.read_dataframe(
            GRID_PATH,
            layer=layer_name,
            where=f"{GRID_ID_COL} = '{escaped}'",
            columns=[GRID_ID_COL],
            use_arrow=True,
        )
    except Exception:
        gdf = pyogrio.read_dataframe(
            GRID_PATH,
            layer=layer_name,
            columns=[GRID_ID_COL],
            use_arrow=True
        )
        gdf[GRID_ID_COL] = gdf[GRID_ID_COL].astype(str)
        gdf = gdf[gdf[GRID_ID_COL] == grid_id].copy()

    if gdf.empty:
        return None
    gdf[GRID_ID_COL] = gdf[GRID_ID_COL].astype(str)
    return gdf.to_crs(4326)

@st.cache_data(show_spinner=False)
def read_visible_standard_geoparquet(bounds_wgs84: Tuple[float, float, float, float]):
    if not GRID_DIAG_GPQ.exists():
        return gpd.GeoDataFrame(), 0, False

    try:
        gdf = gpd.read_parquet(GRID_DIAG_GPQ, bbox=bounds_wgs84)
    except ValueError as e:
        if "Specifying 'bbox' not supported" not in str(e):
            raise
        full = load_full_grid_diag_resource()
        if full.empty:
            return gpd.GeoDataFrame(), 0, False

        bbox_geom = box(*bounds_wgs84)
        cand_idx = list(full.sindex.intersection(bbox_geom.bounds))
        gdf = full.iloc[cand_idx].copy()
        if not gdf.empty:
            gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()

    if gdf.empty:
        return gpd.GeoDataFrame(), 0, False

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    total_n = len(gdf)
    sampled = False
    if total_n > MAX_GRID_RENDER:
        step = math.ceil(total_n / MAX_GRID_RENDER)
        gdf = gdf.iloc[::step].copy()
        sampled = True

    return gdf, total_n, sampled

@st.cache_data(show_spinner=False)
def read_visible_grid_geometry(bounds_wgs84: Tuple[float, float, float, float]):
    layer_name = pyogrio.list_layers(GRID_PATH)[0][0]
    info = pyogrio.read_info(GRID_PATH, layer=layer_name)

    bbox_geom = gpd.GeoSeries([box(*bounds_wgs84)], crs="EPSG:4326").to_crs(info["crs"])
    bbox_ds = tuple(bbox_geom.total_bounds.tolist())

    gdf = pyogrio.read_dataframe(
        GRID_PATH,
        layer=layer_name,
        columns=[GRID_ID_COL],
        bbox=bbox_ds,
        use_arrow=True,
    )

    if gdf.empty:
        return gpd.GeoDataFrame(columns=[GRID_ID_COL, "geometry"], geometry="geometry", crs="EPSG:4326"), 0, False

    gdf[GRID_ID_COL] = gdf[GRID_ID_COL].astype(str)
    total_n = len(gdf)
    sampled = False
    if total_n > MAX_GRID_RENDER:
        step = math.ceil(total_n / MAX_GRID_RENDER)
        gdf = gdf.iloc[::step].copy()
        sampled = True

    return gdf.to_crs(4326), total_n, sampled

@st.cache_data(show_spinner=False)
def read_visible_facilities_geoparquet(bounds_wgs84: Tuple[float, float, float, float]):
    if not FACILITY_GPQ.exists():
        return pd.DataFrame(), 0, False

    try:
        gdf = gpd.read_parquet(FACILITY_GPQ, bbox=bounds_wgs84)
    except ValueError as e:
        if "Specifying 'bbox' not supported" not in str(e):
            raise

        full = load_full_facility_resource()
        if full.empty:
            return pd.DataFrame(), 0, False

        bbox_geom = box(*bounds_wgs84)
        cand_idx = list(full.sindex.intersection(bbox_geom.bounds))
        gdf = full.iloc[cand_idx].copy()
        if not gdf.empty:
            gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()

    if gdf.empty:
        return pd.DataFrame(), 0, False

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    total_n = len(gdf)
    sampled = False
    if total_n > MAX_POINT_RENDER:
        step = math.ceil(total_n / MAX_POINT_RENDER)
        gdf = gdf.iloc[::step].copy()
        sampled = True

    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type != "Point":
            geom = geom.centroid

        status, hours = infer_open_status_and_hours(row)
        kind = str(row.get("facility_kind", "시설"))
        nm = str(row.get("facility_name", ""))
        tp = str(row.get("facility_type", kind))
        color = FACILITY_KIND_COLOR.get(kind, "#6b7280")

        tooltip_html = f"""
        <div style="font-size:12px; line-height:1.4;">
            <b>시설종류</b>: {kind}<br>
            <b>시설명</b>: {nm if nm else '-'}<br>
            <b>시설유형</b>: {tp if tp else '-'}<br>
            <b>현재 운영여부</b>: {status}<br>
            <b>운영시간</b>: {hours}
        </div>
        """

        rows.append({
            "lat": geom.y,
            "lon": geom.x,
            "tooltip_html": tooltip_html,
            "color": color,
            "kind": kind,
        })

    return pd.DataFrame(rows), total_n, sampled


# =========================================================
# 9) 표준세트 그래프/셋 즉시 계산
# =========================================================
@st.cache_data(show_spinner=False)
def compute_single_origin_standard_from_od(from_id: str) -> pd.DataFrame:
    selected_acts = STANDARD_SET.copy()
    time_cols = get_time_cols()
    cov_thr = {a: ACTIVITY_META[a]["cov_thr"] for a in selected_acts}
    available_cols = get_available_od_cols()
    selected_acts = [a for a in selected_acts if a in available_cols]

    df = (
        pl.scan_parquet(OD_PATH)
        .filter(pl.col("from_id") == from_id)
        .select(["from_id", "to_id"] + time_cols + selected_acts)
        .with_columns(
            pl.col("from_id").cast(pl.Utf8, strict=False),
            pl.col("to_id").cast(pl.Utf8, strict=False),
            *[pl.col(t).cast(pl.Float32, strict=False) for t in time_cols],
            *[pl.col(a).fill_null(0).cast(pl.Int8, strict=False) for a in selected_acts],
        )
        .group_by(["from_id", "to_id"])
        .agg(
            *[pl.col(t).min().alias(t) for t in time_cols],
            *[pl.col(a).max().alias(a) for a in selected_acts],
        )
        .collect()
        .to_pandas()
    )

    if df.empty:
        return pd.DataFrame(columns=[
            "from_id", "time", "coverage_pct", "bundle_pct", "bundle_id",
            "reachable_set", "bundle_set", "coverage_loss_bestcase",
            "bundle_loss_bestcase", "coverage_ts_quadrant_time",
            "bundle_ts_quadrant_time", "coverage_ts_type_time",
            "bundle_ts_type_time", "is_best_cov_time", "is_best_bundle_time"
        ])

    cov_values = []
    bun_values = []
    bundle_ids = []
    reachable_sets = []
    bundle_sets = []

    for t in time_cols:
        reachable = []
        for a in selected_acts:
            thr = cov_thr[a]
            ok = (df[a].fillna(0).astype(int) > 0) & (df[t].fillna(np.inf) <= thr)
            if ok.any():
                reachable.append(act_label(a))

        coverage_pct = 100.0 * len(reachable) / max(len(selected_acts), 1)
        cov_values.append(coverage_pct)
        reachable_sets.append(", ".join(reachable) if reachable else "없음")

        counts = []
        for _, r in df.iterrows():
            c = 0
            bundle_set = []
            tt = safe_float(r[t])
            for a in selected_acts:
                if pd.notna(tt) and tt <= BUNDLE_THRESHOLD and int(r[a]) > 0:
                    c += 1
                    bundle_set.append(act_label(a))
            counts.append((c, tt if pd.notna(tt) else np.inf, str(r["to_id"]), bundle_set))

        counts_sorted = sorted(counts, key=lambda x: (-x[0], x[1], x[2]))
        best_count, _, best_to_id, best_bundle_set = counts_sorted[0]
        bundle_pct = 100.0 * best_count / max(len(selected_acts), 1)

        bun_values.append(bundle_pct)
        bundle_ids.append(best_to_id if best_count > 0 else None)
        bundle_sets.append(", ".join(best_bundle_set) if best_bundle_set else "없음")

    cov_values = np.array(cov_values, dtype=float)
    bun_values = np.array(bun_values, dtype=float)
    best_cov = np.nanmax(cov_values)
    best_bun = np.nanmax(bun_values)

    rows = []
    for i, t in enumerate(time_cols):
        rows.append({
            "from_id": from_id,
            "time": t,
            "coverage_pct": float(cov_values[i]),
            "bundle_pct": float(bun_values[i]),
            "bundle_id": bundle_ids[i],
            "reachable_set": reachable_sets[i],
            "bundle_set": bundle_sets[i],
            "coverage_loss_bestcase": float(best_cov - cov_values[i]),
            "bundle_loss_bestcase": float(best_bun - bun_values[i]),
            "coverage_ts_quadrant_time": None,
            "bundle_ts_quadrant_time": None,
            "coverage_ts_type_time": None,
            "bundle_ts_type_time": None,
            "is_best_cov_time": cov_values[i] == best_cov,
            "is_best_bundle_time": bun_values[i] == best_bun,
        })

    return pd.DataFrame(rows)


# =========================================================
# 10) traveler profile 계산
# =========================================================
def get_cov_thresholds(selected_acts: List[str]) -> Dict[str, float]:
    return {a: float(ACTIVITY_META[a]["cov_thr"]) for a in selected_acts}

@st.cache_data(show_spinner=True)
def compute_all_origin_metrics_custom(selected_acts_tuple: tuple) -> pd.DataFrame:
    selected_acts = list(selected_acts_tuple)
    time_cols = get_time_cols()
    cov_thr = get_cov_thresholds(selected_acts)
    available_cols = get_available_od_cols()
    selected_acts = [a for a in selected_acts if a in available_cols]

    lf = (
        pl.scan_parquet(OD_PATH)
        .select(["from_id", "to_id"] + time_cols + selected_acts)
        .with_columns(
            pl.col("from_id").cast(pl.Utf8, strict=False),
            pl.col("to_id").cast(pl.Utf8, strict=False),
            *[pl.col(t).cast(pl.Float32, strict=False) for t in time_cols],
            *[pl.col(a).fill_null(0).cast(pl.Int8, strict=False) for a in selected_acts],
        )
        .group_by(["from_id", "to_id"])
        .agg(
            *[pl.col(t).min().alias(t) for t in time_cols],
            *[pl.col(a).max().alias(a) for a in selected_acts],
        )
    )

    cov_agg_exprs = []
    for t in time_cols:
        for a in selected_acts:
            thr = cov_thr[a]
            cov_agg_exprs.append(
                (
                    (
                        pl.col(t).is_not_null() &
                        (pl.col(t) <= thr) &
                        (pl.col(a) > 0)
                    )
                    .cast(pl.Int8)
                    .max()
                    .alias(f"__{t}_{a}_cov")
                )
            )

    cov_lf = lf.group_by("from_id").agg(cov_agg_exprs)
    cov_lf = cov_lf.with_columns([
        (
            pl.sum_horizontal([pl.col(f"__{t}_{a}_cov") for a in selected_acts]) * (100.0 / len(selected_acts))
        ).alias(f"{t}_coverage")
        for t in time_cols
    ]).select(["from_id"] + [f"{t}_coverage" for t in time_cols])

    bundle_pair_lf = lf.with_columns([
        (
            pl.sum_horizontal([
                (
                    pl.col(t).is_not_null() &
                    (pl.col(t) <= BUNDLE_THRESHOLD) &
                    (pl.col(a) > 0)
                ).cast(pl.Int8)
                for a in selected_acts
            ]) * (100.0 / len(selected_acts))
        ).alias(f"__{t}_bundle_pair")
        for t in time_cols
    ])

    bundle_lf = bundle_pair_lf.group_by("from_id").agg([
        pl.col(f"__{t}_bundle_pair").max().alias(f"{t}_bundle")
        for t in time_cols
    ])

    out = cov_lf.join(bundle_lf, on="from_id", how="outer").collect(streaming=True).to_pandas()

    cov_arr = out[[f"{t}_coverage" for t in time_cols]].to_numpy(dtype=float)
    bun_arr = out[[f"{t}_bundle" for t in time_cols]].to_numpy(dtype=float)

    cov_best = np.nanmax(cov_arr, axis=1)
    bun_best = np.nanmax(bun_arr, axis=1)
    bun_loss = bun_best[:, None] - bun_arr
    bun_loss_mean = np.nanmean(bun_loss, axis=1)
    bun_loss_std = np.nanstd(bun_loss, axis=1)
    bundle_gap = cov_best - bun_best

    fs = cov_best < (100.0 - FS_EPS)
    fd = bundle_gap >= FD_GAP_THRESHOLD
    tc = (bun_loss_mean >= TC_MEAN_THRESHOLD) | (bun_loss_std >= TC_STD_THRESHOLD)

    labels = [build_diag_label(a, b, c) for a, b, c in zip(fs, fd, tc)]
    colors = [build_diag_color(x) for x in labels]

    out["best_cov_pct"] = cov_best
    out["best_bundle_pct"] = bun_best
    out["bundle_gap_best"] = bundle_gap
    out["coverage_loss_mean"] = np.nanmean(cov_best[:, None] - cov_arr, axis=1)
    out["coverage_loss_std"] = np.nanstd(cov_best[:, None] - cov_arr, axis=1)
    out["bundle_loss_mean"] = bun_loss_mean
    out["bundle_loss_std"] = bun_loss_std
    out["structure_diag_best"] = labels
    out["diag_color_best"] = colors
    out["has_ts_best"] = False
    return out

@st.cache_data(show_spinner=False)
def compute_timeseries_and_sets_from_od(from_id: str, selected_acts_tuple: tuple) -> pd.DataFrame:
    selected_acts = list(selected_acts_tuple)
    time_cols = get_time_cols()
    cov_thr = get_cov_thresholds(selected_acts)
    available_cols = get_available_od_cols()
    selected_acts = [a for a in selected_acts if a in available_cols]

    df = (
        pl.scan_parquet(OD_PATH)
        .filter(pl.col("from_id") == from_id)
        .select(["from_id", "to_id"] + time_cols + selected_acts)
        .with_columns(
            pl.col("from_id").cast(pl.Utf8, strict=False),
            pl.col("to_id").cast(pl.Utf8, strict=False),
            *[pl.col(t).cast(pl.Float32, strict=False) for t in time_cols],
            *[pl.col(a).fill_null(0).cast(pl.Int8, strict=False) for a in selected_acts],
        )
        .group_by(["from_id", "to_id"])
        .agg(
            *[pl.col(t).min().alias(t) for t in time_cols],
            *[pl.col(a).max().alias(a) for a in selected_acts],
        )
        .collect()
        .to_pandas()
    )

    if df.empty:
        return pd.DataFrame(columns=["time", "coverage_pct", "bundle_pct", "reachable_set", "bundle_set", "bundle_id"])

    cov_values = []
    bun_values = []

    for t in time_cols:
        reachable = []
        for a in selected_acts:
            thr = cov_thr[a]
            ok = (df[a].fillna(0).astype(int) > 0) & (df[t].fillna(np.inf) <= thr)
            if ok.any():
                reachable.append(act_label(a))
        coverage_pct = 100.0 * len(reachable) / max(len(selected_acts), 1)
        cov_values.append(coverage_pct)

        counts = []
        for _, r in df.iterrows():
            c = 0
            bundle_set = []
            tt = safe_float(r[t])
            for a in selected_acts:
                if pd.notna(tt) and tt <= BUNDLE_THRESHOLD and int(r[a]) > 0:
                    c += 1
                    bundle_set.append(act_label(a))
            counts.append((c, tt if pd.notna(tt) else np.inf, str(r["to_id"]), bundle_set))
        counts_sorted = sorted(counts, key=lambda x: (-x[0], x[1], x[2]))
        best_count, _, best_to_id, best_bundle_set = counts_sorted[0]
        bundle_pct = 100.0 * best_count / max(len(selected_acts), 1)
        bun_values.append(bundle_pct)

    cov_values = np.array(cov_values, dtype=float)
    bun_values = np.array(bun_values, dtype=float)
    best_cov = np.nanmax(cov_values)
    best_bun = np.nanmax(bun_values)

    rows = []
    for idx_t, t in enumerate(time_cols):
        reachable = []
        for a in selected_acts:
            thr = cov_thr[a]
            ok = (df[a].fillna(0).astype(int) > 0) & (df[t].fillna(np.inf) <= thr)
            if ok.any():
                reachable.append(act_label(a))

        counts = []
        for _, r in df.iterrows():
            c = 0
            bundle_set = []
            tt = safe_float(r[t])
            for a in selected_acts:
                if pd.notna(tt) and tt <= BUNDLE_THRESHOLD and int(r[a]) > 0:
                    c += 1
                    bundle_set.append(act_label(a))
            counts.append((c, tt if pd.notna(tt) else np.inf, str(r["to_id"]), bundle_set))

        counts_sorted = sorted(counts, key=lambda x: (-x[0], x[1], x[2]))
        best_count, _, best_to_id, best_bundle_set = counts_sorted[0]

        rows.append({
            "from_id": from_id,
            "time": t,
            "coverage_pct": float(cov_values[idx_t]),
            "bundle_pct": float(bun_values[idx_t]),
            "bundle_id": best_to_id if best_count > 0 else None,
            "reachable_set": ", ".join(reachable) if reachable else "없음",
            "bundle_set": ", ".join(best_bundle_set) if best_bundle_set else "없음",
            "coverage_loss_bestcase": float(best_cov - cov_values[idx_t]),
            "bundle_loss_bestcase": float(best_bun - bun_values[idx_t]),
            "coverage_ts_quadrant_time": None,
            "bundle_ts_quadrant_time": None,
            "coverage_ts_type_time": None,
            "bundle_ts_type_time": None,
            "is_best_cov_time": cov_values[idx_t] == best_cov,
            "is_best_bundle_time": bun_values[idx_t] == best_bun,
            "structure_diag_best": None,
            "has_ts_best": False,
        })

    return pd.DataFrame(rows)


# =========================================================
# 10) hatch overlay
# =========================================================
def iter_lines_from_geom(geom):
    if geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            if isinstance(g, (LineString, MultiLineString)):
                yield from iter_lines_from_geom(g)

def add_hatch_for_polygon(
    m: folium.Map,
    geom,
    color: str = TS_HATCH_COLOR,
    weight: float = 1.0,
    opacity: float = 0.55,
):
    if geom is None or geom.is_empty:
        return

    polygons = []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        return

    for poly in polygons:
        minx, miny, maxx, maxy = poly.bounds
        dx = maxx - minx
        dy = maxy - miny
        if dx <= 0 or dy <= 0:
            continue

        spacing = max(dx, dy) / 7.5
        spacing = max(spacing, 0.00035)

        start_x = minx - dy
        end_x = maxx + dy

        x = start_x
        while x <= end_x:
            line = LineString([(x, miny), (x + dy, maxy)])
            clipped = poly.intersection(line)
            for seg in iter_lines_from_geom(clipped):
                coords = [(y, x) for x, y in list(seg.coords)]
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                ).add_to(m)
            x += spacing


# =========================================================
# 11) 차트
# =========================================================
def make_line_figure(df: pd.DataFrame, y_col: str, hover_col: str, title: str, selected_time: Optional[str], line_color: str) -> go.Figure:
    marker_size = [13 if str(t) == str(selected_time) else 9 for t in df["time"]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df[y_col],
            mode="lines+markers",
            marker=dict(size=marker_size, color=line_color, line=dict(width=1, color="white")),
            line=dict(width=3, color=line_color),
            fill="tozeroy",
            fillcolor="rgba(79,125,232,0.08)" if line_color == "#4f7de8" else "rgba(245,158,11,0.08)",
            customdata=np.stack([df[hover_col].fillna("")], axis=-1),
            hovertemplate="%{x}<br>%{y:.1f}%<br>%{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=18, r=18, t=50, b=18),
        yaxis_title="%",
        xaxis_title="시간대",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(range=[0, 100], gridcolor="rgba(148,163,184,0.20)")
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


# =========================================================
# 12) 선택 bundle 격자 내 시설 요약
# =========================================================
def summarize_bundle_grid_facilities(bundle_grid_gdf: Optional[gpd.GeoDataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bundle_grid_gdf is None or bundle_grid_gdf.empty:
        return pd.DataFrame(), pd.DataFrame()

    full = load_full_facility_resource()
    if full.empty:
        return pd.DataFrame(), pd.DataFrame()

    poly = bundle_grid_gdf.to_crs(full.crs).geometry.iloc[0]
    cand_idx = list(full.sindex.intersection(poly.bounds))
    sub = full.iloc[cand_idx].copy()
    sub = sub[sub.geometry.intersects(poly)].copy()

    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        sub.groupby("facility_kind")
        .size()
        .reset_index(name="개수")
        .sort_values("개수", ascending=False)
    )

    med = sub[sub["facility_kind"] == "의료기관"].copy()
    med_rows = []
    if not med.empty:
        counter = {k: 0 for k in M_LABEL.keys()}
        for _, r in med.iterrows():
            groups = parse_facility_type_groups(str(r.get("facility_type", "")))
            for g in groups:
                if g in counter:
                    counter[g] += 1

        med_rows = pd.DataFrame({
            "의료유형": [f"{k} ({v})" for k, v in M_LABEL.items()],
            "개수": [counter[k] for k in M_LABEL.keys()],
        })

    return summary, med_rows


# =========================================================
# 13) 지도 레이어
# =========================================================
def gdf_to_geojson_data(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_json())

def render_grid_layer(m: folium.Map, gdf: gpd.GeoDataFrame, analysis_requested: bool):
    if gdf is None or gdf.empty:
        return

    layer = gdf.copy()

    if analysis_requested:
        layer["tooltip_text"] = (
            layer["from_id"].astype(str)
            + " / "
            + layer["structure_diag_best"].fillna("미계산").astype(str)
        )
    else:
        if GRID_ID_COL in layer.columns:
            layer["tooltip_text"] = layer[GRID_ID_COL].astype(str)
        else:
            layer["tooltip_text"] = layer["from_id"].astype(str)

    data = gdf_to_geojson_data(layer)

    def style_fn(feature):
        props = feature["properties"]
        if analysis_requested:
            fill_color = props.get("diag_color_best", "#d9d9d9")
            fill_opacity = 0.24
        else:
            fill_color = "#e5e7eb"
            fill_opacity = 0.03
        return {
            "fillColor": fill_color,
            "color": "#9ca3af",
            "weight": 0.5,
            "fillOpacity": fill_opacity,
        }

    folium.GeoJson(
        data=data,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["tooltip_text"],
            aliases=[""],
            labels=False,
            sticky=False,
        ),
    ).add_to(m)

    if analysis_requested and "has_ts_best" in layer.columns:
        ts_subset = layer[layer["has_ts_best"].fillna(False)].copy()
        if not ts_subset.empty:
            for _, row in ts_subset.iterrows():
                add_hatch_for_polygon(
                    m,
                    row.geometry,
                    color=TS_HATCH_COLOR,
                    weight=1.0,
                    opacity=0.55,
                )

def render_map(
    selected_grid: Optional[gpd.GeoDataFrame],
    bundle_grid: Optional[gpd.GeoDataFrame],
    visible_grids: Optional[gpd.GeoDataFrame],
    visible_facilities: pd.DataFrame,
    analysis_requested: bool,
):
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="CartoDB positron",
        control_scale=True,
    )

    render_grid_layer(m, visible_grids, analysis_requested=analysis_requested)

    if selected_grid is not None and not selected_grid.empty:
        row = selected_grid.iloc[0]
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#ecfccb",
                "color": "#65a30d",
                "weight": 3,
                "fillOpacity": 0.52,
            },
            tooltip=folium.Tooltip(f"선택 격자: {row[GRID_ID_COL]}")
        ).add_to(m)

    if bundle_grid is not None and not bundle_grid.empty:
        row = bundle_grid.iloc[0]
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#bfdbfe",
                "color": "#2563eb",
                "weight": 3,
                "fillOpacity": 0.42,
            },
            tooltip=folium.Tooltip(f"bundle 격자: {row[GRID_ID_COL]}")
        ).add_to(m)

    if visible_facilities is not None and not visible_facilities.empty:
        for _, r in visible_facilities.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=4,
                color=r["color"],
                weight=1,
                fill=True,
                fill_color=r["color"],
                fill_opacity=0.82,
                tooltip=folium.Tooltip(r["tooltip_html"], sticky=True),
            ).add_to(m)

    return st_folium(
        m,
        width=None,
        height=MAP_HEIGHT,
        returned_objects=["bounds", "last_clicked", "zoom"],
        key="main_map",
    )


# =========================================================
# 14) 선택 헬퍼
# =========================================================
def set_selected_grid(grid_gdf: gpd.GeoDataFrame, lon: Optional[float], lat: Optional[float], note: Optional[str], matched_address: Optional[str]):
    gid = str(grid_gdf.iloc[0][GRID_ID_COL])

    st.session_state.selected_from_id = gid
    st.session_state.current_lon = lon
    st.session_state.current_lat = lat
    st.session_state.search_note = note
    st.session_state.search_matched_address = matched_address

    tb = tuple(grid_gdf.total_bounds.tolist())
    st.session_state.selected_grid_bounds = tb
    st.session_state.pending_fit_bounds = tb
    st.session_state.selected_time = None
    st.session_state.focus_bundle_to_id = None

def fit_pending_bounds():
    fit_bounds = st.session_state.pending_fit_bounds
    if fit_bounds is None:
        return

    fb = bounds_to_fit(fit_bounds)
    st.session_state.map_bounds = fit_bounds
    st.session_state.map_center = [
        (fb[0][0] + fb[1][0]) / 2,
        (fb[0][1] + fb[1][1]) / 2,
    ]

    dx = fit_bounds[2] - fit_bounds[0]
    if dx < 0.01:
        st.session_state.map_zoom = 15
    elif dx < 0.03:
        st.session_state.map_zoom = 14
    elif dx < 0.08:
        st.session_state.map_zoom = 13
    else:
        st.session_state.map_zoom = 12

    st.session_state.pending_fit_bounds = None


# =========================================================
# 15) 사이드바
# =========================================================
st.title("대중교통 접근성 취약 진단 대시보드")

with st.sidebar:
    st.subheader("1) 주소 검색")

    with st.form("search_form"):
        addr = st.text_input(
            "주소 입력",
            value=st.session_state.address_input,
            placeholder="예: 서울특별시 성동구 왕십리로 222"
        )
        search_submit = st.form_submit_button("격자 찾기")

    if search_submit:
        st.session_state.address_input = addr
        try:
            geocoded = geocode_address(addr)
            if geocoded is None:
                st.error("주소를 찾지 못했습니다. 주소를 더 구체적으로 입력하거나, 지도를 직접 클릭해 격자를 선택하세요.")
            else:
                lon, lat, geocoded_name, match_note = geocoded
                grid_gdf = read_grid_by_point(lon, lat)

                if grid_gdf is None or grid_gdf.empty:
                    st.error("좌표는 찾았지만 해당 좌표가 포함된 격자를 찾지 못했습니다. 지도를 직접 클릭해보세요.")
                else:
                    set_selected_grid(
                        grid_gdf=grid_gdf,
                        lon=lon,
                        lat=lat,
                        note=match_note,
                        matched_address=geocoded_name,
                    )
                    st.success(f"선택 격자: {st.session_state.selected_from_id}")
                    st.caption(f"매칭 주소: {geocoded_name}")
                    fit_pending_bounds()

        except Exception as e:
            st.error(f"주소 검색 오류: {e}")

    st.divider()
    st.subheader("2) 분석 세트")

    mode = st.radio(
        "분석 방식",
        ["표준세트", "Traveler profile"],
        index=0 if st.session_state.analysis_mode == "표준세트" else 1,
    )
    st.session_state.analysis_mode = mode

    if mode == "표준세트":
        st.info("표준세트: " + ", ".join(act_labels(STANDARD_SET)))
        if st.button("세트 결정", key="confirm_standard"):
            st.session_state.selected_activities = STANDARD_SET.copy()
            st.session_state.set_confirmed = True
    else:
        chosen = st.multiselect(
            "activity pool",
            PROFILE_POOL,
            default=[a for a in st.session_state.selected_activities if a in PROFILE_POOL],
            format_func=act_label,
        )
        if st.button("세트 결정", key="confirm_profile"):
            if len(chosen) == 0:
                st.warning("최소 1개 activity를 선택해야 합니다.")
            else:
                st.session_state.selected_activities = chosen
                st.session_state.set_confirmed = True

    if st.session_state.set_confirmed:
        st.success("선택 세트: " + ", ".join(act_labels(st.session_state.selected_activities)))

    st.divider()
    final_submit = st.button("최종 조회", type="primary")


# =========================================================
# 16) 데이터 로드 / 분석
# =========================================================
selected_grid_gdf = None
bundle_grid_gdf = None
selected_origin_metrics = None
timeseries_df = None
analysis_df = None

bestcase_std_df = load_std_bestcase_resource()

if st.session_state.selected_from_id:
    selected_grid_gdf = read_grid_by_id(st.session_state.selected_from_id)

if final_submit:
    if not st.session_state.selected_from_id:
        st.error("먼저 주소 검색 또는 지도 클릭으로 격자를 선택하세요.")
    elif not st.session_state.set_confirmed:
        st.error("먼저 activity set을 결정하세요.")
    else:
        st.session_state.analysis_requested = True

if st.session_state.analysis_requested and st.session_state.selected_from_id and st.session_state.set_confirmed:
    acts = st.session_state.selected_activities

    if acts == STANDARD_SET:
        if not bestcase_std_df.empty and st.session_state.selected_from_id in bestcase_std_df.index:
            selected_origin_metrics = bestcase_std_df.loc[st.session_state.selected_from_id].to_dict()

        # 표준세트 그래프는 반드시 OD에서 즉시 계산
        timeseries_df = compute_single_origin_standard_from_od(st.session_state.selected_from_id)
        if not timeseries_df.empty:
            timeseries_df["time"] = timeseries_df["time"].astype(str)
            timeseries_df = timeseries_df.sort_values("time").reset_index(drop=True)
    else:
        analysis_df = compute_all_origin_metrics_custom(tuple(acts))
        row = analysis_df.loc[analysis_df["from_id"] == st.session_state.selected_from_id]
        if not row.empty:
            selected_origin_metrics = row.iloc[0].to_dict()

        timeseries_df = compute_timeseries_and_sets_from_od(
            st.session_state.selected_from_id,
            tuple(acts),
        )

if st.session_state.selected_time is None and timeseries_df is not None and not timeseries_df.empty:
    st.session_state.selected_time = str(timeseries_df["time"].iloc[0])


# =========================================================
# 17) 우측 패널
# =========================================================
col_map, col_side = st.columns([1.7, 1.0], gap="large")

with col_side:
    st.subheader("선택 격자 정보")

    if st.session_state.selected_from_id:
        st.write(f"**from_id**: `{st.session_state.selected_from_id}`")
    else:
        st.info("주소 검색 또는 지도 클릭으로 먼저 격자를 선택하세요.")

    if st.session_state.search_matched_address:
        st.write(f"**매칭 주소**: {st.session_state.search_matched_address}")

    top_cards = st.columns(2, gap="small")

    if selected_origin_metrics is not None:
        struct_label = selected_origin_metrics.get("structure_diag_best", selected_origin_metrics.get("diag_label", "미계산"))
        has_ts = bool(selected_origin_metrics.get("has_ts_best", False))

        with top_cards[0]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="small-muted">구조 취약 유형</div>
                    <div style="font-size:1.35rem; font-weight:700; margin-top:0.4rem;">{struct_label}{', T(s)' if has_ts else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with top_cards[1]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="small-muted">best bundle</div>
                    <div style="font-size:1.15rem; font-weight:700; margin-top:0.4rem;">{safe_float(selected_origin_metrics.get('best_bundle_pct')):.1f}%</div>
                    <div class="small-muted">@ {selected_origin_metrics.get('best_bundle_time', '-')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        info_cols = st.columns(2, gap="small")
        with info_cols[0]:
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="small-muted">best coverage</div>
                    <div style="font-size:1.15rem; font-weight:700;">{safe_float(selected_origin_metrics.get('best_cov_pct')):.1f}%</div>
                    <div class="small-muted">@ {selected_origin_metrics.get('best_cov_time', '-')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with info_cols[1]:
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="small-muted">bundle gap</div>
                    <div style="font-size:1.15rem; font-weight:700;">{safe_float(selected_origin_metrics.get('bundle_gap_best')):.1f}%</div>
                    <div class="small-muted">best 기준</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="hint-card">
                <div><b>coverage loss mean</b>: {safe_float(selected_origin_metrics.get('coverage_loss_mean')):.1f}</div>
                <div><b>coverage loss std</b>: {safe_float(selected_origin_metrics.get('coverage_loss_std')):.1f}</div>
                <div><b>bundle loss mean</b>: {safe_float(selected_origin_metrics.get('bundle_loss_mean')):.1f}</div>
                <div><b>bundle loss std</b>: {safe_float(selected_origin_metrics.get('bundle_loss_std')):.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if timeseries_df is not None and not timeseries_df.empty:
        cov_hover_col = "reachable_set"
        bun_hover_col = "bundle_set"

        cov_fig = make_line_figure(
            timeseries_df, "coverage_pct", cov_hover_col,
            "시간대별 PT coverage", st.session_state.selected_time, line_color="#4f7de8"
        )
        cov_click = plotly_events(
            cov_fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=300,
            key="coverage_click"
        )
        if cov_click:
            st.session_state.selected_time = str(cov_click[0]["x"])

        bun_fig = make_line_figure(
            timeseries_df, "bundle_pct", bun_hover_col,
            "시간대별 PT bundle", st.session_state.selected_time, line_color="#f59e0b"
        )
        bun_click = plotly_events(
            bun_fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=300,
            key="bundle_click"
        )
        if bun_click:
            st.session_state.selected_time = str(bun_click[0]["x"])

        manual_t = st.selectbox(
            "선택 시간대",
            options=list(timeseries_df["time"]),
            index=list(timeseries_df["time"]).index(st.session_state.selected_time)
            if st.session_state.selected_time in list(timeseries_df["time"])
            else 0,
        )
        st.session_state.selected_time = manual_t

        picked_row = timeseries_df.loc[timeseries_df["time"] == st.session_state.selected_time]
        if not picked_row.empty:
            picked_row = picked_row.iloc[0]

            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="small-muted">현재 시간대</div>
                    <div style="font-size:1.2rem; font-weight:700;">{picked_row['time']}</div>
                    <div class="small-muted">coverage {safe_float(picked_row.get('coverage_pct')):.1f}% · bundle {safe_float(picked_row.get('bundle_pct')):.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"**coverage 도달 가능 시설 종류**: {picked_row.get('reachable_set', '없음')}")
            st.markdown(f"**bundle 최대 조합 시설 종류**: {picked_row.get('bundle_set', '없음')}")

            st.caption(f"coverage loss(best case 기준): {safe_float(picked_row.get('coverage_loss_bestcase')):.1f}")
            st.caption(f"bundle loss(best case 기준): {safe_float(picked_row.get('bundle_loss_bestcase')):.1f}")

            if "coverage_ts_quadrant_time" in picked_row and pd.notna(picked_row["coverage_ts_quadrant_time"]):
                st.caption(f"coverage weakness quadrant: {picked_row['coverage_ts_quadrant_time']}")
            if "bundle_ts_quadrant_time" in picked_row and pd.notna(picked_row["bundle_ts_quadrant_time"]):
                st.caption(f"bundle weakness quadrant: {picked_row['bundle_ts_quadrant_time']}")

            if "coverage_ts_type_time" in picked_row and pd.notna(picked_row["coverage_ts_type_time"]):
                st.caption(f"coverage T(s): {picked_row['coverage_ts_type_time']}")
            if "bundle_ts_type_time" in picked_row and pd.notna(picked_row["bundle_ts_type_time"]):
                st.caption(f"bundle T(s): {picked_row['bundle_ts_type_time']}")

            bun_id = picked_row.get("bundle_id", None)
            st.markdown(f"**bundle_id**: `{bun_id}`" if bun_id not in [None, "", "None", "nan"] else "**bundle_id**: 없음")

            st.session_state.focus_bundle_to_id = None if bun_id in [None, "", "None", "nan"] else bun_id

            if st.session_state.focus_bundle_to_id:
                bundle_grid_gdf = read_grid_by_id(str(st.session_state.focus_bundle_to_id))
                if selected_grid_gdf is not None and bundle_grid_gdf is not None:
                    fit_bounds = combine_bounds([
                        tuple(selected_grid_gdf.total_bounds.tolist()),
                        tuple(bundle_grid_gdf.total_bounds.tolist()),
                    ])
                    st.session_state.pending_fit_bounds = fit_bounds

fit_pending_bounds()


# =========================================================
# 18) 현재 뷰 데이터
# =========================================================
current_bounds = st.session_state.map_bounds or DEFAULT_BOUNDS
current_zoom = st.session_state.map_zoom or DEFAULT_ZOOM

need_grid_layer = False
need_point_layer = False

# 초기엔 아무것도 안 그림
if st.session_state.analysis_requested and current_zoom >= GRID_RENDER_MIN_ZOOM:
    need_grid_layer = True

if (st.session_state.selected_from_id is not None or st.session_state.analysis_requested) and current_zoom >= POINT_RENDER_MIN_ZOOM:
    need_point_layer = True

visible_grids_gdf = gpd.GeoDataFrame()
visible_grid_total = 0
visible_grid_sampled = False

visible_facilities_df = pd.DataFrame()
visible_fac_total = 0
visible_fac_sampled = False

if need_grid_layer:
    if st.session_state.selected_activities == STANDARD_SET and GRID_DIAG_GPQ.exists():
        visible_grids_gdf, visible_grid_total, visible_grid_sampled = read_visible_standard_geoparquet(current_bounds)

        if not visible_grids_gdf.empty:
            if "has_ts_best" not in visible_grids_gdf.columns:
                if "diagnosis_best_json" in visible_grids_gdf.columns:
                    visible_grids_gdf["has_ts_best"] = visible_grids_gdf["diagnosis_best_json"].apply(
                        lambda x: has_ts_label(parse_json_list_like(x))
                    )
                else:
                    visible_grids_gdf["has_ts_best"] = False

            if "structure_diag_best" not in visible_grids_gdf.columns:
                visible_grids_gdf["structure_diag_best"] = "미계산"
            if "diag_color_best" not in visible_grids_gdf.columns:
                visible_grids_gdf["diag_color_best"] = "#d9d9d9"

    else:
        visible_grids_gdf, visible_grid_total, visible_grid_sampled = read_visible_grid_geometry(current_bounds)

        if st.session_state.analysis_requested and st.session_state.selected_activities != STANDARD_SET:
            if analysis_df is None:
                analysis_df = compute_all_origin_metrics_custom(tuple(st.session_state.selected_activities))

            join_df = analysis_df[["from_id", "structure_diag_best", "diag_color_best", "has_ts_best"]].copy()
            visible_grids_gdf = visible_grids_gdf.merge(
                join_df,
                left_on=GRID_ID_COL,
                right_on="from_id",
                how="left",
            )
            visible_grids_gdf["structure_diag_best"] = visible_grids_gdf["structure_diag_best"].fillna("미계산")
            visible_grids_gdf["diag_color_best"] = visible_grids_gdf["diag_color_best"].fillna("#d9d9d9")
            visible_grids_gdf["has_ts_best"] = visible_grids_gdf["has_ts_best"].fillna(False)

if need_point_layer:
    visible_facilities_df, visible_fac_total, visible_fac_sampled = read_visible_facilities_geoparquet(current_bounds)


# =========================================================
# 19) 지도
# =========================================================
with col_map:
    st.subheader("지도")

    if visible_grid_sampled:
        st.caption(f"현재 화면 범위 격자 수가 많아 {visible_grid_total:,}개 중 일부만 렌더링했습니다.")
    if visible_fac_sampled:
        st.caption(f"현재 화면 범위 시설 수가 많아 {visible_fac_total:,}개 중 일부만 렌더링했습니다.")

    map_data = render_map(
        selected_grid=selected_grid_gdf,
        bundle_grid=bundle_grid_gdf,
        visible_grids=visible_grids_gdf if need_grid_layer else gpd.GeoDataFrame(),
        visible_facilities=visible_facilities_df if need_point_layer else pd.DataFrame(),
        analysis_requested=st.session_state.analysis_requested,
    )

    new_bounds = map_data.get("bounds") if map_data else None
    new_bounds = normalize_bounds_from_st_folium(new_bounds)
    if new_bounds is not None:
        st.session_state.map_bounds = new_bounds

    new_zoom = map_data.get("zoom") if map_data else None
    if new_zoom is not None:
        try:
            st.session_state.map_zoom = int(new_zoom)
        except Exception:
            pass

    clicked = map_data.get("last_clicked") if map_data else None
    if clicked is not None:
        click_key = f"{round(clicked['lat'], 6)}|{round(clicked['lng'], 6)}"
        if click_key != st.session_state.last_map_click_key:
            st.session_state.last_map_click_key = click_key
            grid_gdf = read_grid_by_point(clicked["lng"], clicked["lat"])
            if grid_gdf is not None and not grid_gdf.empty:
                set_selected_grid(
                    grid_gdf=grid_gdf,
                    lon=clicked["lng"],
                    lat=clicked["lat"],
                    note="map_click",
                    matched_address="지도 클릭 선택",
                )
                st.rerun()

    st.caption("초기에는 가볍게 띄우기 위해 상세 레이어를 제한합니다. 확대하거나 격자를 선택하면 상세 레이어가 나타납니다.")


# =========================================================
# 20) 우측 하단: bundle 격자 내 시설 요약
# =========================================================
with col_side:
    st.divider()
    st.subheader("선택 bundle 격자 내 시설 요약")

    bundle_summary_df, med_summary_df = summarize_bundle_grid_facilities(bundle_grid_gdf)

    if bundle_summary_df.empty:
        st.write("선택 시간대의 bundle 격자가 없거나, 해당 격자 내 시설이 없습니다.")
    else:
        st.dataframe(bundle_summary_df, use_container_width=True, hide_index=True)

        if med_summary_df is not None and not med_summary_df.empty:
            st.markdown("**의료시설 세부 구성**")
            st.dataframe(med_summary_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("지표 해석")
    st.markdown(
        """
        - **coverage_pct**: 선택 activity set 중 해당 시간대에 reachable한 항목 비율(%)  
        - **bundle_pct**: 하나의 bundle 격자에서 함께 만족되는 activity 비율(%)  
        - **best case**: 해당 지표가 가장 높은 시간대  
          (`loss = best_value - current_value` 이므로, loss가 가장 작은 시간대와 같음)  
        """
    )