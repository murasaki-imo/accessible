from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import Normalize, PowerNorm
from shapely.geometry import box
import streamlit.components.v1 as st_components


# =========================================================
# 경로 자동 탐색
# =========================================================
def pick_first_existing_path(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


import os as _os

_DATA_ROOT_ENV = _os.environ.get("DATA_ROOT", "")

if _DATA_ROOT_ENV:
    # ── 서버 모드: DATA_ROOT=/app/data (docker-compose volume) ──────────────
    # data/ 폴더에 아래 파일들을 flat하게 넣어두면 됩니다:
    #   from_metrics_500m_intracity_classified.parquet
    #   500m.gpkg, station.gpkg, subway.gpkg
    #   all_facilities.geoparquet
    #   from_metrics_500m_intracity.parquet
    #   od_500m_intracity.parquet
    #   deficit_ref_sgg.csv  (optional)
    DATA_ROOT = Path(_DATA_ROOT_ENV)
    ROOT_OUT  = DATA_ROOT

    CLASSIFIED_PATH = pick_first_existing_path(
        DATA_ROOT / "from_metrics_500m_intracity_classified.parquet",
        DATA_ROOT / "500m_classified.parquet",
    )
    GRID_PATH    = DATA_ROOT / "500m.gpkg"
    STATION_PATH = DATA_ROOT / "station.gpkg"
    SUBWAY_PATH  = DATA_ROOT / "subway.gpkg"
    FAC_PATH     = pick_first_existing_path(
        DATA_ROOT / "all_facilities.geoparquet",
        DATA_ROOT / "all_activities.geoparquet",
    )
    INTRACITY_PATH = pick_first_existing_path(
        DATA_ROOT / "from_metrics_500m_intracity.parquet",
        DATA_ROOT / "from_metrics_500m_intracity_classified.parquet",
    )
    OD_PATH = pick_first_existing_path(
        DATA_ROOT / "od_500m_intracity.parquet",
        DATA_ROOT / "od.parquet",
    )
    DEFICIT_REF_NAT_PATH = pick_first_existing_path(
        DATA_ROOT / "deficit_ref_sgg.csv",
        DATA_ROOT / "deficit_ref_nat.csv",
    )

else:
    # ── 로컬 모드: Dropbox 절대경로 ─────────────────────────────────────────
    def _pick_dropbox_base() -> Path:
        for p in [Path(r"E:\Dropbox"), Path(r"C:\Users\82102\Dropbox"), Path.home() / "Dropbox"]:
            if p.exists():
                return p
        return Path(r"C:\Users\82102\Dropbox")

    _DB    = _pick_dropbox_base()
    _PAPER = _DB / r"01-대학원\02-Paper Work\01-개인연구\202603_격자 단위 다양한 시설 대중교통 접근성 결핍 진단"
    _RAW   = _PAPER / r"03-분석자료\01-기초자료\01-전처리\02_routing\02_intracity\02_500m"
    _VIZ   = _PAPER / r"03-분석자료\01-기초자료\02-삽도자료"

    ROOT_OUT = _PAPER / r"03-분석자료\01-기초자료\02-삽도자료\02_학회자료\01_교통학회"

    CLASSIFIED_PATH = pick_first_existing_path(
        _RAW / "from_metrics_500m_intracity_classified.parquet",
        _RAW / "500m_classified.parquet",
    )
    GRID_PATH    = _PAPER / r"03-분석자료\01-기초자료\01-전처리\02_routing\00_grid\500m.gpkg"
    STATION_PATH = _VIZ / r"01_재료\station.gpkg"
    SUBWAY_PATH  = _VIZ / r"01_재료\subway.gpkg"
    FAC_PATH     = pick_first_existing_path(
        _RAW / "all_facilities.geoparquet",
        _RAW / "all_activities.geoparquet",
    )
    INTRACITY_PATH = pick_first_existing_path(
        _RAW / "from_metrics_500m_intracity.parquet",
        _RAW / "from_metrics_500m_intracity_classified.parquet",
    )
    OD_PATH = pick_first_existing_path(
        _RAW / "od_500m_intracity.parquet",
        _RAW / "od.parquet",
    )
    DEFICIT_REF_NAT_PATH = pick_first_existing_path(
        _PAPER / r"01-학회자료\01_교통학회\03_분석결과\deficit_ref_sgg.csv",
        _PAPER / r"03-분석자료\01-기초자료\deficit_ref_sgg.csv",
    )

CACHE_DIR = ROOT_OUT / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_GRID        = CACHE_DIR / "grid_dashboard_5179.geoparquet"
CACHE_GRID_SIMPLE = CACHE_DIR / "grid_dashboard_simple_5179.geoparquet"
CACHE_SGG         = CACHE_DIR / "sgg_dashboard_5179.geoparquet"
CACHE_STATION     = CACHE_DIR / "station_5179.geoparquet"
CACHE_SUBWAY      = CACHE_DIR / "subway_5179.geoparquet"
CACHE_FAC         = CACHE_DIR / "facilities_5179.geoparquet"
CACHE_TS          = CACHE_DIR / "grid_timeseries.parquet"
CACHE_IDX         = CACHE_DIR / "grid_index_points_5179.parquet"
CACHE_CELL_DATA   = CACHE_DIR / "cell_detail_data.parquet"

CACHE_GEOJSON_DIR    = CACHE_DIR / "geojson_tiles"
CACHE_STATION_JSON   = CACHE_GEOJSON_DIR / "station.json"
CACHE_SUBWAY_JSON    = CACHE_GEOJSON_DIR / "subway.json"
CACHE_FAC_JSON_TPL   = str(CACHE_GEOJSON_DIR / "fac_{ftype}.json")
CACHE_FACILITY_ACCESS = CACHE_DIR / "grid_facility_access.parquet"

# =========================================================
# 상수 / 스타일
# =========================================================
PLOT_CRS = "EPSG:5179"
WEB_CRS  = "EPSG:4326"

GRID_ID_COL   = "GRID_500M_"
GRID_JOIN_COL = "from_id"
SGG_CODE_COL  = "from_sgg_key"
SGG_NAME_COL  = "from_sgg"

FAC_KIND_COL = "facility_kind"
FAC_TYPE_COL = "facility_type"
FAC_DEPT_COL = "department"

TIME_SLOTS = ["08", "10", "12", "14", "16", "18", "20", "22"]
COV_COLS   = [f"{t}_coverage" for t in TIME_SLOTS]
MAI_COLS   = [f"{t}_mai"      for t in TIME_SLOTS]

OD_FACILITY_COLS = ["pharmacy", "grocery", "library", "park", "public", "m1", "m2", "m3", "m4", "m5", "m6"]
OD_FACILITY_LABELS = {
    "pharmacy": "Pharmacy",
    "grocery":  "Grocery",
    "library":  "Library",
    "park":     "Park",
    "public":   "Public service",
    "m1":       "Primary care",       # 가정의학과, 내과, 소아청소년과
    "m2":       "Rehab & ortho",      # 정형외과, 재활의학과, 마취통증의학과
    "m3":       "Specialty clinic",   # 안과, 이비인후과, 피부과, 비뇨기, 신경과, 산부인과
    "m4":       "Mental health",      # 정신건강의학과
    "m5":       "Dental",             # 치과 계열
    "m6":       "Korean medicine",    # 한방 계열
}

# m2~m6을 "Specialist care"로 묶어 표시 (논문 기준: 하나라도 접근 가능하면 accessible)
# 세부 과목은 inaccessible 시 토글로 확인
SPECIALIST_COLS  = ["m2", "m3", "m4", "m5", "m6"]
SPECIALIST_LABEL = "Specialist care"
SPECIALIST_DETAIL_LABELS = {
    "m2": "Rehab & ortho",
    "m3": "Specialty clinic",
    "m4": "Mental health",
    "m5": "Dental",
    "m6": "Korean medicine",
}
# 패널/툴팁에서 실제로 표시할 시설 순서 (m2~m6 → specialist 하나로)
DISPLAY_FAC_COLS   = ["park", "library", "m1", "specialist", "grocery", "public", "pharmacy"]
DISPLAY_FAC_LABELS = {
    "park":       "Park",
    "library":    "Library",
    "m1":         "Primary care",
    "specialist": SPECIALIST_LABEL,
    "grocery":    "Grocery",
    "public":     "Public service",
    "pharmacy":   "Pharmacy",
}

# 시설별 Coverage 접근 기준시간 (분) — 국토부 제2차 국가도시재생기본방침 기준
FAC_COV_THRESH = {
    "park":     15, "library":  15,
    "m1":       10, "m2":       15, "m3":       15,
    "m4":       15, "m5":       15, "m6":       15,
    "grocery":  10, "public":   15, "pharmacy": 10,
}
# MAI는 항상 15분 기준 (출발지에서 15분 내 도달 가능한 to_id 중 최다 시설 그리드)
MAI_THRESH = 15

LAYER_LABEL_TO_KEY = {"F(s)": "fs", "F(d)": "fd", "T(c)": "tc", "T(f)": "tf", "Population": "pop"}
LAYER_KEY_TO_LABEL = {v: k for k, v in LAYER_LABEL_TO_KEY.items()}
LAYER_HELP = {
    "F(s)": "Facility siting / sub-optimal location problem",
    "F(d)": "Facility dispersion problem",
    "T(c)": "Transit connection problem",
    "T(f)": "Transit frequency problem",
    "Population": "Population share or density",
}

BASE_MAP_KEYS   = ["pop", "coverage", "mai"]
BASE_MAP_LABELS = {"pop": "Population", "coverage": "Coverage (avg.)", "mai": "MAI (avg.)"}

# ── 컬러맵 (요청 반영) ────────────────────────────
CMAPS = {
    "fs":       "RdPu",
    "fd":       "BuPu",
    "tc":       "PuRd",
    "tf":       "YlOrBr",
    "pop":      "Reds",      # ← 요청: Reds
    "coverage": "YlGnBu",    # ← 요청: YlGnBu
    "mai":      "BuPu",      # ← 요청: BuPu
}

# 결핍 격자 테두리 색상 (유형별 구분, fill 없음)
DEFICIT_COLORS = {
    "fs": "#E53935",   # 빨강 (이미지 좌상단)
    "fd": "#F4A100",   # 골드/주황 (이미지 우상단)
    "tc": "#7B1FA2",   # 보라 (이미지 좌하단)
    "tf": "#00BCD4",   # 시안 (격자색과 구별되는 밝은 청록)
}
DEFICIT_LABELS = {"fs": "F(s)", "fd": "F(d)", "tc": "T(c)", "tf": "T(f)"}

FACILITY_COLORS = {
    "park":     "#43A047", "library": "#1E88E5",
    "m1":       "#E53935", "m2":      "#8E24AA",
    "grocery":  "#FB8C00", "public":  "#00ACC1",
    "pharmacy": "#D81B60",
}
FACILITY_LABELS_EN = {
    "park":     "Park",
    "library":  "Library",
    "m1":       "Primary care",
    "m2":       "Specialist care",
    "grocery":  "Grocery",
    "public":   "Public service",
    "pharmacy": "Pharmacy",
}
FACILITY_ORDER = ["park", "library", "m1", "m2", "grocery", "public", "pharmacy"]

MED_GROUP_MAP_RAW = {
    "가정의학과": "m1", "내과": "m1", "소아청소년과": "m1",
    "정형외과": "m2", "재활의학과": "m2", "마취통증의학과": "m2",
    "안과": "m3", "이비인후과": "m3", "피부과": "m3",
    "비뇨의학과": "m3", "신경과": "m3", "산부인과": "m3",
    "정신건강의학과": "m4",
    "치과": "m5", "통합치의학과": "m5", "소아치과": "m5",
    "치과교정과": "m5", "치과보존과": "m5", "치과보철과": "m5",
    "치주과": "m5", "구강내과": "m5",
    "예방치과": "m8", "영상치의학과": "m8", "구강병리과": "m8", "구강악안면외과": "m8",
    "사상체질과": "m6", "침구과": "m6", "한방내과": "m6",
    "한방부인과": "m6", "한방소아과": "m6", "한방신경정신과": "m6",
    "한방안·이비인후·피부과": "m6", "한방재활의학과": "m6",
    "한방응급": "m7", "외과": "m7", "신경외과": "m7",
    "심장혈관흉부외과": "m7", "응급의학과": "m7", "결핵과": "m7",
    "방사선종양학과": "m7", "핵의학과": "m7", "병리과": "m7",
    "영상의학과": "m7", "진단검사의학과": "m7", "예방의학과": "m7",
    "직업환경의학과": "m7", "성형외과": "m7",
}
MED_ALLOWED_RAW    = {"m1", "m2", "m3", "m4", "m5", "m6"}
MED_SPECIALIZED_RAW = {"m2", "m3", "m4", "m5", "m6"}

COV_LINE_COLOR = "#5C6BC0"
MAI_LINE_COLOR = "#26A69A"

POP_MAP_MODE = "share"
MAP_HEIGHT   = 560


# =========================================================
# 유틸
# =========================================================
def normalize_code_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return out

def normalize_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def normalize_sgg_code(code) -> str:
    try:    return str(int(float(str(code).strip())))
    except: return str(code).strip()

def parse_deficit_tokens(val) -> Set[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    if isinstance(val, (list, tuple, set)):
        raw = list(val)
    else:
        s = str(val).strip()
        if s in ["", "[]", "{}"]: return set()
        try:
            obj = json.loads(s)
            raw = obj if isinstance(obj, list) else [obj]
        except:
            raw = re.findall(r"[FfTt]\([sScCdDfF]\)", s)
    out = set()
    for x in raw:
        t = str(x).strip().lower()
        if   t == "f(s)": out.add("F(s)")
        elif t == "f(d)": out.add("F(d)")
        elif t == "t(c)": out.add("T(c)")
        elif t == "t(f)": out.add("T(f)")
    return out

def parse_department_list(val) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)): return []
    if isinstance(val, list): return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if s in ["", "[]", "nan", "None"]: return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list): return [str(x).strip() for x in obj if str(x).strip()]
    except: pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list): return [str(x).strip() for x in obj if str(x).strip()]
    except: pass
    s2 = s.strip("[]")
    parts = [p.strip().strip('"').strip("'") for p in s2.split(",")]
    return [p for p in parts if p]

def normalize_facility_type_from_row(row: pd.Series) -> str:
    kind = str(row.get(FAC_KIND_COL, "")).strip(); ftype = str(row.get(FAC_TYPE_COL, "")).strip()
    kl = kind.lower(); fl = ftype.lower()
    if "공원" in kind or "park" in kl or "공원" in ftype or "park" in fl: return "park"
    if "도서관" in kind or "library" in kl or "도서관" in ftype or "library" in fl: return "library"
    if "약국" in kind or "pharmacy" in kl or "약국" in ftype or "pharmacy" in fl: return "pharmacy"
    if ("식료품" in kind or "마트" in kind or "시장" in kind or "편의점" in kind or
        "grocery" in kl or "market" in kl or "mart" in kl or "supermarket" in kl or
        "식료품" in ftype or "마트" in ftype or "시장" in ftype or "편의점" in ftype): return "grocery"
    if ("행정" in kind or "공공" in kind or "주민센터" in kind or "행정서비스" in kind or
        "행정" in ftype or "공공" in ftype or "주민센터" in ftype or "행정서비스" in ftype): return "public"
    dept_list  = parse_department_list(row.get(FAC_DEPT_COL))
    raw_groups = [MED_GROUP_MAP_RAW[d] for d in dept_list if d in MED_GROUP_MAP_RAW]
    raw_groups = [g for g in raw_groups if g in MED_ALLOWED_RAW]
    if "m1" in raw_groups: return "m1"
    if any(g in MED_SPECIALIZED_RAW for g in raw_groups): return "m2"
    if ("의료" in kind or "병원" in kind or "의원" in kind or "치과" in kind or "한의원" in kind or
        "의료" in ftype or "병원" in ftype or "의원" in ftype or "치과" in ftype or "한의원" in ftype):
        if any(x in ftype for x in ["보건소", "보건지소", "보건진료소", "보건의료원"]): return "m1"
        return "m2"
    return "exclude"


class PowerNormSafe(PowerNorm):
    pass

def compute_group_norm_from_series(values: pd.Series, gamma: float = 0.55, force_zero_min: bool = True):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0]
    vmin, vmax = (0.0, 1.0) if len(vals) == 0 else (0.0 if force_zero_min else float(vals.min()), float(vals.max()))
    if vmax <= vmin: vmax = vmin + 1e-9
    return vmin, vmax, PowerNormSafe(gamma=gamma, vmin=vmin, vmax=vmax)

def compute_group_pop_norm(values: pd.Series, share_mode: bool = True):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vmin, vmax = (0.0, 1.0) if len(vals) == 0 else (0.0 if share_mode else float(vals.min()), float(vals.max()))
    if vmax <= vmin: vmax = vmin + 1e-9
    norm = PowerNormSafe(gamma=0.6, vmin=vmin, vmax=vmax) if share_mode else Normalize(vmin=vmin, vmax=vmax)
    return vmin, vmax, norm

def compute_continuous_norm(values: pd.Series, gamma: float = 0.55):
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0]
    vmin = 0.0; vmax = float(vals.max()) if len(vals) else 100.0
    if vmax <= vmin: vmax = 100.0
    return vmin, vmax, PowerNormSafe(gamma=gamma, vmin=vmin, vmax=vmax)

def gradient_css_from_cmap(cmap_name: str) -> str:
    cmap  = matplotlib.colormaps[cmap_name]
    stops = [f"{mcolors.to_hex(cmap(p))} {int(p*100)}%" for p in np.linspace(0, 1, 7)]
    return ", ".join(stops)

def make_pretty_ticks(vmin, vmax, n=5):
    if vmax <= vmin: return np.array([vmin, vmax + 1e-9])
    return np.linspace(vmin, vmax, n)

def choose_tick_decimals(vmin, vmax):
    span = abs(vmax - vmin)
    if span < 0.05: return 3
    if span < 0.5:  return 2
    if span < 5:    return 1
    return 0

def build_dashboard_cache(progress_cb=None) -> Dict[str, str]:
    metrics = pd.read_parquet(CLASSIFIED_PATH).copy()
    metrics[GRID_JOIN_COL] = normalize_str_series(metrics[GRID_JOIN_COL])
    metrics[SGG_CODE_COL]  = normalize_code_series(metrics[SGG_CODE_COL])
    metrics[SGG_NAME_COL]  = normalize_str_series(metrics[SGG_NAME_COL])
    metrics["pop"] = pd.to_numeric(metrics["pop"], errors="coerce").fillna(0)
    for c in COV_COLS + MAI_COLS:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")
    for c in ["avg_coverage", "avg_mai", "cv_coverage", "cv_mai", "car_coverage", "car_mai"]:
        if c in metrics.columns:
            metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    metrics["_nat_tokens"] = metrics["nat_deficit"].apply(parse_deficit_tokens)
    metrics["_sgg_tokens"] = metrics["sgg_deficit"].apply(parse_deficit_tokens)
    for tok, key in {"F(s)": "fs", "F(d)": "fd", "T(c)": "tc", "T(f)": "tf"}.items():
        metrics[f"nat_has_{key}"] = metrics["_nat_tokens"].apply(lambda s: tok in s)
        metrics[f"sgg_has_{key}"] = metrics["_sgg_tokens"].apply(lambda s: tok in s)

    # ── national 기준 결핍 CSV 보강 ──────────────────────────────────────────
    # deficit_ref_sgg.csv 가 있으면 nat_has_* 컬럼을 CSV 값으로 덮어씀
    # 예상 컬럼: from_id (or grid_id), nat_has_fs, nat_has_fd, nat_has_tc, nat_has_tf
    if DEFICIT_REF_NAT_PATH.exists():
        try:
            ref = pd.read_csv(DEFICIT_REF_NAT_PATH)
            # from_id 컬럼 탐색 (다양한 이름 허용)
            id_col = next((c for c in ref.columns if c.lower() in ("from_id","grid_id","id","from_sgg_key")), None)
            if id_col:
                ref[id_col] = ref[id_col].astype(str).str.strip()
                # nat_has_* 컬럼이 있으면 덮어쓰기, bool/int 통일
                nat_cols = [c for c in ref.columns if c.startswith("nat_has_") and c in [f"nat_has_{k}" for k in ["fs","fd","tc","tf"]]]
                if nat_cols:
                    ref_sub = ref[[id_col] + nat_cols].rename(columns={id_col: GRID_JOIN_COL})
                    for c in nat_cols:
                        ref_sub[c] = ref_sub[c].astype(bool)
                    # metrics에서 기존 nat_has_* 제거 후 merge
                    drop_cols = [c for c in nat_cols if c in metrics.columns]
                    metrics = metrics.drop(columns=drop_cols)
                    metrics = metrics.merge(ref_sub, on=GRID_JOIN_COL, how="left")
                    for c in nat_cols:
                        if c in metrics.columns:
                            metrics[c] = metrics[c].fillna(False).astype(bool)
                        else:
                            metrics[c] = False
        except Exception:
            import traceback; traceback.print_exc()

    metrics["sgg_pop_total"]      = metrics.groupby(SGG_CODE_COL)["pop"].transform("sum")
    metrics["national_pop_total"] = float(metrics["pop"].sum())
    metrics["nat_pop_map"]   = np.where(metrics["national_pop_total"] > 0, metrics["pop"] / metrics["national_pop_total"] * 100.0, 0.0)
    metrics["sgg_pop_map"]   = np.where(metrics["sgg_pop_total"] > 0,      metrics["pop"] / metrics["sgg_pop_total"] * 100.0,      0.0)
    metrics["local_pop_map"] = metrics["sgg_pop_map"]

    for key in ["fs", "fd", "tc", "tf"]:
        metrics[f"nat_{key}_ratio"] = np.where(metrics[f"nat_has_{key}"] & (metrics["sgg_pop_total"] > 0), metrics["pop"] / metrics["sgg_pop_total"] * 100.0, 0.0)
        metrics[f"sgg_{key}_ratio"] = np.where(metrics[f"sgg_has_{key}"] & (metrics["sgg_pop_total"] > 0), metrics["pop"] / metrics["sgg_pop_total"] * 100.0, 0.0)

    # 시군구 가중 평균
    def _sgg_wmean(col):
        return (
            metrics.assign(_w=metrics[col] * metrics["pop"])
            .groupby(SGG_CODE_COL).agg(_w=("_w","sum"), pop=("pop","sum"))
            .assign(v=lambda x: np.where(x["pop"] > 0, x["_w"] / x["pop"], np.nan))["v"]
        )
    if "avg_coverage" in metrics.columns:
        metrics["sgg_avg_coverage"] = metrics[SGG_CODE_COL].map(_sgg_wmean("avg_coverage"))
    if "avg_mai" in metrics.columns:
        metrics["sgg_avg_mai"]      = metrics[SGG_CODE_COL].map(_sgg_wmean("avg_mai"))

    grid = gpd.read_file(GRID_PATH)
    if grid.crs is None: grid = grid.set_crs(epsg=4326)
    if str(grid.crs) != PLOT_CRS: grid = grid.to_crs(PLOT_CRS)
    grid[GRID_ID_COL] = normalize_str_series(grid[GRID_ID_COL])
    grid = grid[[GRID_ID_COL, "geometry"]].copy()

    gdf = grid.merge(metrics, left_on=GRID_ID_COL, right_on=GRID_JOIN_COL, how="inner")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PLOT_CRS)
    gdf["area_km2"]        = gdf.geometry.area / 1_000_000.0
    gdf["pop_density_km2"] = np.where(gdf["area_km2"] > 0, gdf["pop"] / gdf["area_km2"], np.nan)

    # idx_df: sgg_avg 포함해서 저장
    centroid = gdf.geometry.centroid
    idx_cols = {
        GRID_JOIN_COL: gdf[GRID_JOIN_COL], SGG_CODE_COL: gdf[SGG_CODE_COL], SGG_NAME_COL: gdf[SGG_NAME_COL],
        "x": centroid.x, "y": centroid.y, "pop": gdf["pop"],
        "avg_coverage": gdf["avg_coverage"] if "avg_coverage" in gdf.columns else np.nan,
        "avg_mai":      gdf["avg_mai"]      if "avg_mai"      in gdf.columns else np.nan,
        "cv_coverage":  gdf["cv_coverage"]  if "cv_coverage"  in gdf.columns else np.nan,
        "cv_mai":       gdf["cv_mai"]       if "cv_mai"       in gdf.columns else np.nan,
        "car_coverage": gdf["car_coverage"] if "car_coverage" in gdf.columns else np.nan,
        "car_mai":      gdf["car_mai"]      if "car_mai"      in gdf.columns else np.nan,
        "sgg_avg_coverage": gdf["sgg_avg_coverage"] if "sgg_avg_coverage" in gdf.columns else np.nan,
        "sgg_avg_mai":      gdf["sgg_avg_mai"]      if "sgg_avg_mai"      in gdf.columns else np.nan,
    }
    idx_df = pd.DataFrame(idx_cols)
    for c in COV_COLS: idx_df[c] = gdf[c]
    for c in MAI_COLS: idx_df[c] = gdf[c]
    idx_df.to_parquet(CACHE_IDX, index=False)

    safe_grid_cols = [
        GRID_JOIN_COL, SGG_CODE_COL, SGG_NAME_COL, "pop",
        "nat_deficit", "sgg_deficit",
        "nat_pop_map", "sgg_pop_map", "local_pop_map",
        "nat_fs_ratio", "nat_fd_ratio", "nat_tc_ratio", "nat_tf_ratio",
        "sgg_fs_ratio", "sgg_fd_ratio", "sgg_tc_ratio", "sgg_tf_ratio",
        "nat_has_fs", "nat_has_fd", "nat_has_tc", "nat_has_tf",
        "sgg_has_fs", "sgg_has_fd", "sgg_has_tc", "sgg_has_tf",
        "avg_coverage", "avg_mai", "cv_coverage", "cv_mai",
        "car_coverage", "car_mai",
        "sgg_avg_coverage", "sgg_avg_mai",
        "area_km2", "pop_density_km2",
        *COV_COLS, *MAI_COLS, "geometry",
    ]
    safe_grid_cols = [c for c in safe_grid_cols if c in gdf.columns]
    gdf_safe   = gdf[safe_grid_cols].copy()
    gdf_simple = gdf_safe.copy()
    gdf_simple["geometry"] = gdf_simple.geometry.simplify(25, preserve_topology=True)
    gdf_safe.to_parquet(CACHE_GRID,        index=False)
    gdf_simple.to_parquet(CACHE_GRID_SIMPLE, index=False)

    agg_rows = []
    for code, gg in gdf_safe.groupby(SGG_CODE_COL, dropna=True):
        row = {SGG_CODE_COL: code, SGG_NAME_COL: gg[SGG_NAME_COL].iloc[0],
               "sgg_pop_total": float(gg["pop"].sum()),
               "nat_pop_map": float(gg["nat_pop_map"].sum()),
               "sgg_pop_map": float(gg["sgg_pop_map"].sum())}
        for k in ["fs","fd","tc","tf"]:
            row[f"nat_{k}_ratio"] = float(gg[f"nat_{k}_ratio"].sum())
            row[f"sgg_{k}_ratio"] = float(gg[f"sgg_{k}_ratio"].sum())
        agg_rows.append(row)
    sgg_attr = pd.DataFrame(agg_rows)
    sgg_poly = gdf_safe[[SGG_CODE_COL, "geometry"]].dissolve(by=SGG_CODE_COL).reset_index()
    sgg_poly = sgg_poly.merge(sgg_attr, on=SGG_CODE_COL, how="left")
    sgg_poly = gpd.GeoDataFrame(sgg_poly, geometry="geometry", crs=PLOT_CRS)
    sgg_poly.to_parquet(CACHE_SGG, index=False)

    station = gpd.read_file(STATION_PATH)
    subway  = gpd.read_file(SUBWAY_PATH)
    fac     = gpd.read_parquet(FAC_PATH)
    for lyr in [station, subway, fac]:
        if lyr.crs is None: lyr.set_crs(epsg=4326, inplace=True)
    if str(station.crs) != PLOT_CRS: station = station.to_crs(PLOT_CRS)
    if str(subway.crs)  != PLOT_CRS: subway  = subway.to_crs(PLOT_CRS)
    if str(fac.crs)     != PLOT_CRS: fac     = fac.to_crs(PLOT_CRS)
    fac["fac_type_norm"] = fac.apply(normalize_facility_type_from_row, axis=1)
    fac = fac[fac["fac_type_norm"].isin(FACILITY_ORDER)].copy()
    station.to_parquet(CACHE_STATION, index=False)
    subway.to_parquet(CACHE_SUBWAY,  index=False)
    fac.to_parquet(CACHE_FAC,        index=False)

    ts_keep = [GRID_JOIN_COL, SGG_CODE_COL, SGG_NAME_COL,
               "avg_coverage", "avg_mai", "cv_coverage", "cv_mai", "car_coverage", "car_mai",
               "sgg_avg_coverage", "sgg_avg_mai",
               *COV_COLS, *MAI_COLS]
    ts_keep = [c for c in ts_keep if c in metrics.columns]
    metrics[ts_keep].to_parquet(CACHE_TS, index=False)

    # ── OD 기반 시설 접근 가능 여부 (시간대별 Coverage + MAI) ─────────────────
    if OD_PATH.exists():
        try:
            if progress_cb: progress_cb(0, 1, "Computing facility accessibility from OD file...")
            od = pd.read_parquet(OD_PATH)
            od["from_id"] = od["from_id"].astype(str).str.strip()
            od["to_id"]   = od["to_id"].astype(str).str.strip()

            PT_SLOT_COL = {s: f"pt{s}" for s in TIME_SLOTS}
            PT_SLOT_COL = {s: c for s, c in PT_SLOT_COL.items() if c in od.columns}
            FAC_COLS    = [c for c in OD_FACILITY_COLS if c in od.columns]

            if not FAC_COLS or not PT_SLOT_COL:
                raise ValueError(f"Required columns missing. PT: {list(PT_SLOT_COL)}, fac: {FAC_COLS}")

            # ── float32로 통일 (메모리 절약) ──────────────────────
            for c in PT_SLOT_COL.values():
                od[c] = od[c].astype(np.float32)
            for c in FAC_COLS:
                od[c] = od[c].astype(np.int8)

            # ── Coverage: groupby + transform → from_id별 슬롯×시설 집계 ──
            # 각 (from_id, slot, fac) 조합에 대해 threshold 내에 시설 존재 여부
            cov_result_cols = {}
            for slot, pt_col in PT_SLOT_COL.items():
                pt_vals = od[pt_col].values
                for fc in FAC_COLS:
                    thresh    = FAC_COV_THRESH.get(fc, 15)
                    col_name  = f"cov_{slot}_{fc}"
                    # 해당 행이 조건 만족 여부 (0/1)
                    od[col_name] = ((pt_vals <= thresh) & (od[fc].values > 0)).astype(np.int8)
                    # from_id 그룹 내 any → max()
                    cov_result_cols[col_name] = "max"

            # ── 슬롯별 최솟값 PT 계산 ──────────────────────────
            pt_cols_list = list(PT_SLOT_COL.values())
            od["_min_pt"] = od[pt_cols_list].min(axis=1).astype(np.float32)

            # ── MAI: 15분 내 to_id 중 최다 시설 (벡터화) ──────
            within_mask = od["_min_pt"] <= MAI_THRESH
            od_w = od[within_mask].copy()

            if not od_w.empty:
                od_w["_n_fac"] = od_w[FAC_COLS].gt(0).sum(axis=1).astype(np.int16)
                # from_id별 최대 시설 수
                max_fac = od_w.groupby("from_id")["_n_fac"].transform("max")
                # 최대 시설 수 & 최소 PT를 동시에 만족하는 행 선택
                best_mask = od_w["_n_fac"] == max_fac
                od_best   = od_w[best_mask].copy()
                # 동점 내 최소 PT 선택
                min_pt_in_best = od_best.groupby("from_id")["_min_pt"].transform("min")
                od_best = od_best[od_best["_min_pt"] == min_pt_in_best]
                # 중복 남아있을 경우 첫 행만
                od_best = od_best.groupby("from_id", as_index=False).first()
                od_best = od_best.set_index("from_id")

                # tie 여부: 위에서 best_mask 기준 count > 1이면 tie
                tie_counts = od_w[best_mask].groupby("from_id")["_min_pt"].count()
                od_best["mai_is_tie"] = (tie_counts > 1).astype(np.int8)
                od_best["mai_best_to_id"] = od_best["to_id"]
                mai_fac_cols = {f"mai_{fc}": od_best[fc].gt(0).astype(np.int8) for fc in FAC_COLS}
                for col, series in mai_fac_cols.items():
                    od_best[col] = series
                mai_df = od_best[["mai_is_tie", "mai_best_to_id"] + [f"mai_{fc}" for fc in FAC_COLS]].reset_index()
            else:
                mai_df = pd.DataFrame(columns=["from_id", "mai_is_tie", "mai_best_to_id"] + [f"mai_{fc}" for fc in FAC_COLS])

            # ── Coverage 집계: from_id 그룹별 max ──────────────
            cov_cols_list = list(cov_result_cols.keys())
            cov_df = od.groupby("from_id", as_index=False)[cov_cols_list].max()

            # ── 병합 ────────────────────────────────────────────
            result = cov_df.merge(mai_df, on="from_id", how="left")
            # MAI 없는 from_id → 0 채우기
            for fc in FAC_COLS:
                result[f"mai_{fc}"] = result[f"mai_{fc}"].fillna(0).astype(np.int8)
            result["mai_is_tie"]    = result["mai_is_tie"].fillna(0).astype(np.int8)

            result.rename(columns={"from_id": GRID_JOIN_COL}, inplace=True)
            result.to_parquet(CACHE_FACILITY_ACCESS, index=False)
            if progress_cb: progress_cb(1, 1, f"OD facility cache built ({len(result):,} grids).")
        except Exception:
            import traceback; traceback.print_exc()

    # ── 정적 GeoJSON 캐시 ──────────────────────────────────
    CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)

    def _write_json_safe(path: Path, data: dict) -> None:
        import tempfile, os, time
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            for attempt in range(5):
                try:
                    if path.exists(): path.unlink()
                    os.rename(tmp_path, str(path)); break
                except PermissionError:
                    time.sleep(0.3)
        except:
            try: os.unlink(tmp_path)
            except: pass
            raise

    def _gdf_to_geojson_dict(gdf: gpd.GeoDataFrame) -> dict:
        """iterrows 없이 벡터화 변환. bool/numpy 타입 자동 직렬화."""
        prop_cols = [c for c in gdf.columns if c != "geometry"]
        # numpy 타입 → Python 기본 타입 일괄 변환
        rec = gdf[prop_cols].copy()
        for c in prop_cols:
            if rec[c].dtype.kind in ("b",):          rec[c] = rec[c].astype(bool)
            elif rec[c].dtype.kind in ("i", "u"):    rec[c] = rec[c].astype(int)
            elif rec[c].dtype.kind in ("f",):        rec[c] = rec[c].astype(float)
        records = rec.where(rec.notna(), other=None).to_dict(orient="records")
        geoms   = gdf.geometry.values
        feats   = [
            {"type": "Feature", "geometry": g.__geo_interface__, "properties": p}
            for g, p in zip(geoms, records)
            if g is not None and not g.is_empty
        ]
        return {"type": "FeatureCollection", "features": feats}

    _write_json_safe(CACHE_STATION_JSON, _gdf_to_geojson_dict(station.to_crs(WEB_CRS)[["geometry"]]))
    _write_json_safe(CACHE_SUBWAY_JSON,  _gdf_to_geojson_dict(subway.to_crs(WEB_CRS)[["geometry"]]))
    fac_w = fac.to_crs(WEB_CRS)
    for ftype in FACILITY_ORDER:
        sub = fac_w[fac_w["fac_type_norm"] == ftype]
        if not sub.empty:
            _write_json_safe(Path(CACHE_FAC_JSON_TPL.format(ftype=ftype)),
                             _gdf_to_geojson_dict(sub[["geometry", "fac_type_norm"]]))

    # ── 시군구별 그리드 GeoJSON ──────────────────────────
    metric_cols = [
        "sgg_fs_ratio","sgg_fd_ratio","sgg_tc_ratio","sgg_tf_ratio",
        "nat_fs_ratio","nat_fd_ratio","nat_tc_ratio","nat_tf_ratio",
        "sgg_pop_map","nat_pop_map","local_pop_map",
        "avg_coverage","avg_mai",
        "nat_has_fs","nat_has_fd","nat_has_tc","nat_has_tf",
        "sgg_has_fs","sgg_has_fd","sgg_has_tc","sgg_has_tf",
        GRID_JOIN_COL, SGG_NAME_COL, SGG_CODE_COL,
    ]
    metric_cols = [c for c in metric_cols if c in gdf_safe.columns]
    gdf_web = gdf_simple.to_crs(WEB_CRS).copy()
    gdf_web[SGG_CODE_COL] = gdf_web[SGG_CODE_COL].apply(normalize_sgg_code)
    sgg_groups = list(gdf_web.groupby(SGG_CODE_COL))
    n_tiles = len(sgg_groups)
    for i, (sgg_code, grp) in enumerate(sgg_groups):
        key  = normalize_sgg_code(sgg_code)
        keep = [c for c in metric_cols if c in grp.columns] + ["geometry"]
        _write_json_safe(CACHE_GEOJSON_DIR / f"grid_{key}.json", _gdf_to_geojson_dict(grp[keep].copy()))
        if progress_cb:
            progress_cb(i + 1, n_tiles, f"GeoJSON tile: {key} ({i+1}/{n_tiles})")

    # ── 셀 상세 데이터 캐시 (sgg_avg 포함) ─────────────
    if INTRACITY_PATH.exists():
        try:
            ic = pd.read_parquet(INTRACITY_PATH)
            ic[GRID_JOIN_COL] = ic[GRID_JOIN_COL].astype(str).str.strip()
            keep_c = [GRID_JOIN_COL, "pop", "car_coverage", "car_mai",
                      "avg_coverage", "avg_mai", "cv_coverage", "cv_mai",
                      *COV_COLS, *MAI_COLS]
            ic_save = ic[[c for c in keep_c if c in ic.columns]].copy()
            # sgg_avg_coverage / sgg_avg_mai 를 metrics에서 merge
            sgg_ref = metrics[[GRID_JOIN_COL, "sgg_avg_coverage", "sgg_avg_mai"]].copy() if "sgg_avg_coverage" in metrics.columns else None
            if sgg_ref is not None:
                ic_save = ic_save.merge(sgg_ref, on=GRID_JOIN_COL, how="left")
            ic_save.to_parquet(CACHE_CELL_DATA, index=False)
        except Exception:
            pass

    return {"ok": "1"}


@st.cache_data(show_spinner=False)
def load_cached_data():
    grid        = gpd.read_parquet(CACHE_GRID)
    grid_simple = gpd.read_parquet(CACHE_GRID_SIMPLE)
    sgg         = gpd.read_parquet(CACHE_SGG)
    station     = gpd.read_parquet(CACHE_STATION)
    subway      = gpd.read_parquet(CACHE_SUBWAY)
    fac         = gpd.read_parquet(CACHE_FAC)
    ts          = pd.read_parquet(CACHE_TS)
    idx         = pd.read_parquet(CACHE_IDX)
    return grid, grid_simple, sgg, station, subway, fac, ts, idx

@st.cache_data(show_spinner=False)
def load_cell_detail_data() -> pd.DataFrame:
    src = CACHE_CELL_DATA if CACHE_CELL_DATA.exists() else (CACHE_IDX if CACHE_IDX.exists() else None)
    if src is None: return pd.DataFrame()
    df = pd.read_parquet(src)
    df[GRID_JOIN_COL] = df[GRID_JOIN_COL].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_facility_access_data() -> pd.DataFrame:
    if not CACHE_FACILITY_ACCESS.exists(): return pd.DataFrame()
    df = pd.read_parquet(CACHE_FACILITY_ACCESS)
    df[GRID_JOIN_COL] = df[GRID_JOIN_COL].astype(str).str.strip()
    return df


# =========================================================
# 지도 HTML 빌드
# =========================================================
def _read_json_safe(path) -> str:
    p = Path(path)
    if not p.exists(): return "null"
    with open(p, "r", encoding="utf-8") as fh: return fh.read()



def _hex_colors_for(base_feats, vcol, norm_obj, cmap_obj):
    raw = np.array(
        [(float(f["properties"][vcol]) if f["properties"].get(vcol) is not None else 0.0)
         for f in base_feats], dtype=np.float32)
    pos = raw > 0
    out = [None] * len(base_feats)
    if pos.any():
        rgba = cmap_obj(norm_obj(raw[pos]))
        for j, hi in enumerate(np.where(pos)[0]):
            out[hi] = mcolors.to_hex(rgba[j])
    return out


def _make_colorbar_html(cmap_name, vmin, vmax):
    ticks  = make_pretty_ticks(vmin, vmax, n=5)
    dec    = choose_tick_decimals(vmin, vmax)
    grad   = gradient_css_from_cmap(cmap_name)
    labels = "".join(
        f"<div style='text-align:center;font-size:10px;color:#666;'>{t:.{dec}f}%</div>"
        for t in ticks)
    return (
        f"<div style='padding:4px 8px 6px;background:#fff;border-top:1px solid #eee;'>"
        f"<div style='height:7px;border-radius:2px;margin-bottom:2px;"
        f"background:linear-gradient(to right,{grad});'></div>"
        f"<div style='display:grid;grid-template-columns:repeat(5,1fr);'>{labels}</div>"
        f"</div>")


@st.cache_data(show_spinner=False, max_entries=64)
def build_multi_map_html(
    sgg_code: str,
    metric_keys_str: str,   # "|".join(selected_metrics)
    center_lat: float,
    center_lng: float,
    zoom: int,
    height_px: int,
    deficit_colors_json: str = "{}",
    # basis / fac_visible 은 캐시 키 제외:
    #   basis → JS 가 localStorage('deficitBasis') 읽어 동적 recolor
    #   fac   → JS 가 localStorage('facVisible')  읽어 초기화/저장
) -> str:
    """Iframe HTML. sgg_code+metric_keys 기준으로만 캐시. sgg/nat 색상 양쪽 embed."""
    sgg_key     = normalize_sgg_code(sgg_code)
    metric_keys = [k for k in metric_keys_str.split("|") if k]
    n           = len(metric_keys)

    subway_js  = _read_json_safe(CACHE_SUBWAY_JSON)
    station_js = _read_json_safe(CACHE_STATION_JSON)

    fac_parts = []
    for ftype in FACILITY_ORDER:
        fj = _read_json_safe(Path(CACHE_FAC_JSON_TPL.format(ftype=ftype)))
        if fj != "null":
            fac_parts.append(
                '{"id":' + json.dumps(ftype) +
                ',"label":' + json.dumps(FACILITY_LABELS_EN.get(ftype, ftype)) +
                ',"c":' + json.dumps(FACILITY_COLORS.get(ftype, "#999")) +
                ',"d":' + fj + '}')
    fac_js = "[" + ",".join(fac_parts) + "]"

    grid_path = CACHE_GEOJSON_DIR / f"grid_{sgg_key}.json"
    base_gj   = None
    if grid_path.exists():
        with open(grid_path, "r", encoding="utf-8") as fh:
            base_gj = json.load(fh)
    base_feats = base_gj.get("features", []) if base_gj else []

    # ── 모든 metric × sgg/nat 색상+colorbar 미리 계산 ────────────────────────
    colors_by_mk = {}
    cbars_by_mk  = {}

    for mk in metric_keys:
        cmap_name = CMAPS.get(mk, "Blues")
        cmap_obj  = matplotlib.colormaps[cmap_name]
        colors_by_mk[mk] = {}
        cbars_by_mk[mk]  = {}
        for basis in ("sgg", "nat"):
            vcol = get_value_col(mk, basis)
            if mk in ("coverage", "mai"):
                vals_s = pd.Series(
                    [float(f["properties"].get(vcol) or 0) for f in base_feats],
                    dtype=np.float32)
                _, vmax, norm_obj = compute_continuous_norm(vals_s, gamma=0.6)
                vmin = 0.0
            elif mk == "pop":
                vals_s = pd.Series(
                    [float(f["properties"].get(vcol) or 0) for f in base_feats],
                    dtype=np.float32)
                vmin, vmax, norm_obj = compute_group_pop_norm(vals_s, share_mode=True)
            else:
                all_vals = []
                for k2 in ["fs","fd","tc","tf"]:
                    vc2 = f"{basis}_{k2}_ratio"
                    all_vals.extend(
                        [float(f["properties"].get(vc2) or 0) for f in base_feats])
                vmin, vmax, norm_obj = compute_group_norm_from_series(
                    pd.Series(all_vals, dtype=np.float32), gamma=0.55, force_zero_min=True)
            colors_by_mk[mk][basis] = _hex_colors_for(base_feats, vcol, norm_obj, cmap_obj)
            cbars_by_mk[mk][basis]  = _make_colorbar_html(cmap_name, vmin, vmax)

    # ── GeoJSON: _fs(sgg색)/_fn(nat색) 양쪽 embed ────────────────────────────
    grid_js_list = []
    if not base_feats:
        grid_js_list = ["null"] * n
    else:
        for mk in metric_keys:
            sc_ = colors_by_mk[mk]["sgg"]
            nc_ = colors_by_mk[mk]["nat"]
            new_feats = []
            for fi, feat in enumerate(base_feats):
                p = dict(feat["properties"])
                p["_fs"] = sc_[fi]
                p["_fn"] = nc_[fi]
                p["_o"]  = 0.80
                new_feats.append({"type":"Feature",
                                  "geometry":feat["geometry"],
                                  "properties":p})
            grid_js_list.append(
                json.dumps({"type":"FeatureCollection","features":new_feats}))

    cbars_js_arr = '[' + ','.join(
        json.dumps({"sgg": cbars_by_mk[mk]["sgg"], "nat": cbars_by_mk[mk]["nat"]})
        for mk in metric_keys) + ']'

    sgg_col = SGG_NAME_COL
    gj_col  = GRID_JOIN_COL

    cols_css = "display:block;" if n == 1 else "display:grid;grid-template-columns:1fr 1fr;gap:4px;"
    map_divs = ""
    for i, mk in enumerate(metric_keys):
        map_divs += (
            f'<div class="map-wrap">'
            f'<div class="map-title">{BASE_MAP_LABELS.get(mk, LAYER_KEY_TO_LABEL.get(mk, mk))}</div>'
            f'<div id="map{i}" class="map-box"></div>'
            f'<div id="cb{i}" class="colorbar-wrap">{cbars_by_mk[mk]["sgg"]}</div>'
            f'</div>')

    deficit_colors = json.loads(deficit_colors_json)

    # ── JS: 지도 초기화 ──────────────────────────────────────────────────────
    maps_init = ""
    for mi, mk in enumerate(metric_keys):
        sc_js  = json.dumps(sgg_col)
        jc_js  = json.dumps(gj_col)
        maps_init += (
            f"var map{mi}=L.map('map{mi}',{{"
            f"center:[{center_lat},{center_lng}],zoom:{zoom},"
            f"zoomControl:true,preferCanvas:true,renderer:L.canvas({{tolerance:3}})}});\n"
            f"L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',"
            f"{{attribution:'&copy; OpenStreetMap &copy; CARTO',subdomains:'abcd',maxZoom:19}}).addTo(map{mi});\n"
            f"(function(map,gd,sc,jc){{"
            f"if(!gd)return;"
            f"var glyr=L.geoJSON(gd,{{style:function(f){{"
            f"  var p=f.properties;"
            f"  var fc=(window.deficitBasis||'sgg')==='nat'?p._fn:p._fs;"
            f"  if(!fc)return{{fillOpacity:0,weight:0.8,color:'#999',opacity:0.45}};"
            f"  return{{fillColor:fc,fillOpacity:p._o||0,weight:0.8,color:'#888',opacity:0.5}};}},\n"
            f"onEachFeature:function(f,layer){{var p=f.properties,t='';"
            f"if(p[sc])t+='<b>'+p[sc]+'</b><br/>';"
            f"if(t)layer.bindTooltip(t,{{sticky:true,opacity:.95}});}},\n"
            f"renderer:L.canvas({{tolerance:2}})}}).addTo(map);\n"
            f"gridLayers.push(glyr);\n"
            f"var feats=gd.features||[];"
            f"map.on('click',function(e){{"
            f"var lat=e.latlng.lat,lng=e.latlng.lng,found=null;"
            f"for(var i=0;i<feats.length;i++){{"
            f"var f=feats[i];if(!f.geometry)continue;"
            f"var bb=getBBox(f.geometry);"
            f"if(lat<bb[1]||lat>bb[3]||lng<bb[0]||lng>bb[2])continue;"
            f"if(pointInPolygon(lat,lng,f.geometry)){{found=f;break;}}"
            f"}}"
            f"if(!found)return;"
            f"var fid=found.properties[jc];if(!fid)return;"
            f"if(window._hlGeoJSON){{map.removeLayer(window._hlGeoJSON);window._hlGeoJSON=null;}}"
            f"window._hlGeoJSON=L.geoJSON(found,{{style:function(){{"
            f"return{{fill:false,weight:3,color:'#1A73E8',opacity:1}};}}}}).addTo(map);"
            f"showCellPanel(String(fid));"
            f"}});"
            f"}})(map{mi},gridData[{mi}],{sc_js},{jc_js});\n"
            f"if(subwayData)L.geoJSON(subwayData,{{style:function(){{return{{color:'#424242',weight:2,opacity:0.6,fill:false}}}},renderer:L.canvas()}}).addTo(map{mi});\n"
            f"if(stationData)L.geoJSON(stationData,{{pointToLayer:function(f,ll){{"
            f"return L.circleMarker(ll,{{radius:4.5,color:'#333',weight:1.5,fillColor:'#fff',fillOpacity:1,opacity:1}});}},renderer:L.canvas()}}).addTo(map{mi});\n"
            f"facData.forEach(function(x){{if(!x.d)return;"
            f"if(!facLayers[x.id])facLayers[x.id]={{maps:[],color:x.c,label:x.label}};"
            f"var lyr=L.geoJSON(x.d,{{pointToLayer:function(f,ll){{"
            f"return L.circleMarker(ll,{{radius:4,color:'#fff',weight:1,fillColor:x.c,fillOpacity:.88,opacity:.88}});}},renderer:L.canvas()}});"
            f"facLayers[x.id].maps.push({{map:map{mi},layer:lyr}});"
            f"if(facVisible.indexOf(x.id)>=0)lyr.addTo(map{mi});}});\n"
            f"Object.keys(deficitInfo).forEach(function(dk){{"
            f"  if(!deficitLayers[dk])deficitLayers[dk]={{maps:[],color:deficitInfo[dk].c,label:deficitInfo[dk].label}};"
            f"  (function(dkLocal,dcolLocal){{"
            f"    function makeStyleFns(basis){{"
            f"      var col=basis+'_has_'+dkLocal;"
            f"      var wFn=function(f){{var p=f.properties;"
            f"        var isD=p[col]===true||p[col]==='true'||p[col]===1;"
            f"        if(!isD)return{{fill:false,weight:0,opacity:0}};"
            f"        return{{fill:false,weight:5,color:'#ffffff',opacity:1}};}};"
            f"      var cFn=function(f){{var p=f.properties;"
            f"        var isD=p[col]===true||p[col]==='true'||p[col]===1;"
            f"        if(!isD)return{{fill:false,weight:0,opacity:0}};"
            f"        return{{fill:false,weight:2.5,color:dcolLocal,opacity:1}};}};"
            f"      return{{white:wFn,color:cFn}};"
            f"    }}"
            f"    var fns=makeStyleFns(window.deficitBasis||'sgg');"
            f"    var dlyrW=L.geoJSON(gridData[{mi}],{{style:fns.white,renderer:L.canvas({{tolerance:0}})}});"
            f"    var dlyr=L.geoJSON(gridData[{mi}],{{style:fns.color,renderer:L.canvas({{tolerance:0}})}});"
            f"    deficitLayers[dkLocal].maps.push({{map:map{mi},layer:dlyr,whiteLayer:dlyrW,makeStyleFns:makeStyleFns}});"
            f"    if(deficitVisible.indexOf(dkLocal)>=0){{dlyrW.addTo(map{mi});dlyr.addTo(map{mi});}}"
            f"  }})(dk,deficitInfo[dk].c);"
            f"}});\n"
            f"allMaps.push(map{mi});\n"
        )

    sync_js = (
        "allMaps.forEach(function(src){"
        "allMaps.forEach(function(dst){"
        "if(src===dst)return;"
        "src.on('move',function(){dst.setView(src.getCenter(),src.getZoom(),{animate:false});});"
        "});});"
    )

    fac_toggle_js = (
        "var ctrl=document.getElementById('fc');\n"
        "(function(){"
        "  var btn=document.createElement('button');"
        "  btn.textContent='All Off';"
        "  btn.style.cssText='font-size:9px;padding:1px 7px;margin-bottom:4px;cursor:pointer;"
        "border:1px solid #ccc;border-radius:3px;background:#f5f5f5;color:#555;display:block;width:100%;';"
        "  btn.addEventListener('click',function(){"
        "    var cbs=ctrl.querySelectorAll('input[type=checkbox]');"
        "    cbs.forEach(function(cb){if(cb.checked){cb.checked=false;cb.dispatchEvent(new Event('change'));}});"
        "    try{localStorage.setItem('facVisible',JSON.stringify([]));}catch(e){}"
        "  });"
        "  ctrl.appendChild(btn);"
        "})();\n"
        "Object.keys(facLayers).forEach(function(id){"
        "  var info=facLayers[id];"
        "  var lbl=document.createElement('label');"
        "  var cb=document.createElement('input');cb.type='checkbox';cb.setAttribute('data-fid',id);"
        "  cb.checked=(facVisible.indexOf(id)>=0);"
        "  cb.addEventListener('change',function(){"
        "    var on=this.checked;"
        "    info.maps.forEach(function(x){if(on)x.layer.addTo(x.map);else x.map.removeLayer(x.layer);});"
        "    var vis=Object.keys(facLayers).filter(function(k){"
        "      var c=ctrl.querySelector('input[data-fid=\"'+k+'\"]');return c&&c.checked;});"
        "    try{localStorage.setItem('facVisible',JSON.stringify(vis));}catch(e){}"
        "  });"
        "  var dot=document.createElement('span');dot.className='dot';dot.style.background=info.color;"
        "  lbl.appendChild(cb);lbl.appendChild(dot);"
        "  lbl.appendChild(document.createTextNode(info.label));"
        "  ctrl.appendChild(lbl);"
        "});"
    )

    rows    = 1 if n <= 2 else 2
    panel_h = (height_px + 26 + 42) * rows

    panel_css = (
        ".outer-wrap{display:flex;flex-direction:row;gap:0;width:100%;}"
        ".maps-section{flex:1 1 0;min-width:0;}"
        ".cell-panel{"
        "flex:0 0 268px;width:268px;min-width:268px;"
        "background:#fefefe;border-left:1px solid #e8e8e8;"
        "padding:12px 14px 10px 14px;font-family:'Inter',system-ui,sans-serif;"
        "display:none;overflow-y:auto;max-height:" + str(panel_h) + "px;"
        "}"
        ".cell-panel.visible{display:block;}"
        ".cp-sec{font-size:7.5px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;"
        "margin-top:10px;margin-bottom:3px;}"
        ".cp-metrics{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin-bottom:2px;}"
        ".cp-m{background:#f9f9f9;border:1px solid #eee;border-radius:4px;padding:5px 7px;}"
        ".cp-m-label{font-size:7.5px;color:#aaa;margin-bottom:1px;font-weight:500;}"
        ".cp-m-val{font-size:14px;font-weight:700;color:#1a1a1a;line-height:1.2;}"
        ".cp-ref{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px 6px;}"
        ".cp-r-label{font-size:7.5px;color:#bbb;font-weight:500;}"
        ".cp-r-val{font-size:11px;font-weight:600;color:#555;}"
        ".cp-close{float:right;cursor:pointer;font-size:14px;color:#ccc;margin-left:5px;padding:2px;}"
        ".cp-close:hover{color:#555;}"
        ".cp-id{font-size:11px;font-weight:700;color:#1a1a1a;border-bottom:2px solid #3F51B5;padding-bottom:3px;}"
        ".cp-id-sub{font-size:7px;color:#bbb;letter-spacing:.6px;margin-bottom:4px;text-transform:uppercase;}"
        ".cp-sgg-ref{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;}"
        ".cp-sgg-m{background:#f0f4ff;border:1px solid #dde5f5;border-radius:4px;padding:4px 7px;}"
        ".cp-sgg-label{font-size:7.5px;color:#8a9ac0;margin-bottom:1px;font-weight:500;}"
        ".cp-sgg-val{font-size:12px;font-weight:700;color:#3F51B5;}"
        ".cp-inacc-tags{display:flex;flex-wrap:wrap;gap:3px;margin-top:2px;}"
        ".cp-deficit-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:2px;}"
        ".cp-inacc-tag{font-size:8.5px;padding:2px 5px;border-radius:10px;"
        "background:#FFF3E0;color:#E65100;border:1px solid #FFB74D;font-weight:600;}"
        ".cp-inacc-ok{font-size:8.5px;color:#43A047;font-weight:600;}"
        "#cp-chart{display:block;width:100%!important;height:145px!important;}"
        ".fac-ctrl{position:absolute;bottom:48px;right:8px;z-index:1000;"
        "background:rgba(255,255,255,.97);border:1px solid #ddd;border-radius:6px;"
        "padding:6px 10px;font-size:11px;line-height:1.9;"
        "box-shadow:0 2px 8px rgba(0,0,0,.10);min-width:145px;}"
        ".fac-hdr{font-weight:700;font-size:9.5px;color:#666;margin-bottom:2px;letter-spacing:.5px;text-transform:uppercase;}"
        ".fac-ctrl label{display:flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap;color:#444;}"
        ".fac-ctrl input{cursor:pointer;margin:0;width:12px;height:12px;}"
        ".dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;border:1px solid rgba(0,0,0,.12);}"
        ".map-wrap{position:relative;}"
        ".map-title{font-size:11px;font-weight:600;color:#555;padding:3px 8px;background:#fafafa;"
        "letter-spacing:.3px;border-bottom:1px solid #eee;}"
        ".colorbar-wrap{background:#fff;border-top:none;}"
        ".map-box{width:100%;height:" + str(height_px) + "px;}"
        ".maps-grid{" + cols_css + "width:100%;}"
        ".leaflet-tooltip{font-size:11px;background:rgba(255,255,255,.97);"
        "border:1px solid #ddd;padding:4px 8px;box-shadow:0 2px 6px rgba(0,0,0,.10);color:#333;}"
    )

    chart_cdn = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'

    panel_js = (
        "var _cpChart=null;\n"
        "var TIME_SLOTS=['08:00','10:00','12:00','14:00','16:00','18:00','20:00','22:00'];\n"
        "var SLOT_KEYS=['08','10','12','14','16','18','20','22'];\n"
        "var COV_COLS=['08_coverage','10_coverage','12_coverage','14_coverage','16_coverage','18_coverage','20_coverage','22_coverage'];\n"
        "var MAI_COLS=['08_mai','10_mai','12_mai','14_mai','16_mai','18_mai','20_mai','22_mai'];\n"
        "var FAC_LABEL_MAP=" + json.dumps(OD_FACILITY_LABELS) + ";\n"
        "var FAC_OD_COLS="   + json.dumps(OD_FACILITY_COLS)   + ";\n"
        "var FAC_COV_THRESH=" + json.dumps(FAC_COV_THRESH)    + ";\n"
        "var MAI_THRESH=" + str(MAI_THRESH) + ";\n"
        # 표시용: m2~m6 → specialist 하나로 묶음
        "var DISPLAY_FAC_COLS=" + json.dumps(DISPLAY_FAC_COLS) + ";\n"
        "var DISPLAY_FAC_LABELS=" + json.dumps(DISPLAY_FAC_LABELS) + ";\n"
        "var SPECIALIST_COLS=" + json.dumps(SPECIALIST_COLS) + ";\n"
        "var SPECIALIST_DETAIL_LABELS=" + json.dumps(SPECIALIST_DETAIL_LABELS) + ";\n"
        "var SPECIALIST_LABEL='" + SPECIALIST_LABEL + "';\n"
        "var DEFICIT_LABEL_MAP={fs:'F(s)',fd:'F(d)',tc:'T(c)',tf:'T(f)'};\n"
        "var DEFICIT_COLOR_MAP=" + json.dumps(DEFICIT_COLORS) + ";\n"
        # DEFICIT_BASIS는 localStorage에서 동적으로 읽음 (basis 전환 시 뷰포트 유지)
        "function getDeficitBasis(){return window.deficitBasis||'sgg';}\n"
        "function fv(v,suf,dec){"
        "  if(v===null||v===undefined)return 'N/A';"
        "  var f=parseFloat(v);if(isNaN(f))return 'N/A';"
        "  return f.toFixed(dec!==undefined?dec:2)+(suf!==undefined?suf:'%');}\n"
        "function showCellPanel(fid){\n"
        "  var d=(window.cellData||{})[fid];\n"
        "  var panel=document.getElementById('cell-panel');\n"
        "  document.getElementById('cp-fid').textContent=fid;\n"
        "  if(!d){panel.classList.add('visible');return;}\n"
        # PT metrics
        "  document.getElementById('cp-avg-cov').textContent=fv(d.avg_coverage);\n"
        "  document.getElementById('cp-avg-mai').textContent=fv(d.avg_mai);\n"
        "  document.getElementById('cp-cv-cov').textContent=fv(d.cv_coverage,'');\n"
        "  document.getElementById('cp-cv-mai').textContent=fv(d.cv_mai,'');\n"
        # SGG avg
        "  document.getElementById('cp-sgg-cov').textContent=fv(d.sgg_avg_coverage);\n"
        "  document.getElementById('cp-sgg-mai').textContent=fv(d.sgg_avg_mai);\n"
        # Deficit types
        "  var defEl=document.getElementById('cp-deficit-tags');\n"
        "  var defTypes=[];\n"
        "  ['fs','fd','tc','tf'].forEach(function(dk){\n"
        "    var col=getDeficitBasis()+'_has_'+dk;\n"
        "    if(d[col]===true||d[col]==='true'||d[col]===1)defTypes.push(dk);\n"
        "  });\n"
        "  if(defTypes.length>0){\n"
        "    defEl.innerHTML=defTypes.map(function(dk){\n"
        "      var c=DEFICIT_COLOR_MAP[dk]||'#999';\n"
        "      return '<span style=\"display:inline-flex;align-items:center;gap:3px;font-size:9px;font-weight:700;padding:2px 7px;border-radius:10px;border:1.5px solid '+c+';color:'+c+';background:'+c+'18;\">'+DEFICIT_LABEL_MAP[dk]+'</span>';\n"
        "    }).join('');\n"
        "  } else {\n"
        "    defEl.innerHTML='<span style=\"font-size:9px;color:#aaa;\">None</span>';\n"
        "  }\n"
        # Reference
        "  document.getElementById('cp-pop').textContent=fv(d.pop,'',0);\n"
        "  document.getElementById('cp-car-cov').textContent=fv(d.car_coverage);\n"
        "  document.getElementById('cp-car-mai').textContent=fv(d.car_mai);\n"
        # Inaccessible facilities (static panel) — m2~m6 → "Specialist care" 묶음 표시
        "  var acc=(window.facAccessData||{})[fid];\n"
        "  var tagsEl=document.getElementById('cp-inacc-tags');\n"
        "  function buildInaccHTML(getAccFn){\n"
        "    var rawAcc={};\n"
        "    FAC_OD_COLS.forEach(function(fc){rawAcc[fc]=getAccFn(fc);});\n"
        "    var specOk=SPECIALIST_COLS.some(function(fc){return rawAcc[fc]===true;});\n"
        "    var det=SPECIALIST_COLS.filter(function(fc){return !rawAcc[fc];});\n"
        # non-specialist inaccessible list
        "    var inaccDcs=DISPLAY_FAC_COLS.filter(function(dc){return dc!=='specialist'&&!rawAcc[dc];});\n"
        # specialist 완전 accessible (세부 모두 ok)이면 패널 포함 안 함
        "    var showSpec=!(specOk&&det.length===0);\n"
        "    var hasAnyInacc=inaccDcs.length>0||(!specOk);\n"
        "    if(!hasAnyInacc&&!showSpec)return '<span class=\"cp-inacc-ok\">✓ All accessible</span>';\n"
        "    var html='';\n"
        # non-specialist inaccessible tags
        "    html+=inaccDcs.map(function(dc){\n"
        "      return '<span class=\"cp-inacc-tag\">'+(DISPLAY_FAC_LABELS[dc]||dc)+'</span>';\n"
        "    }).join('');\n"
        # specialist tag: accessible→초록, inaccessible→기존 태그색
        "    if(showSpec){\n"
        "      var tid='sp-detail-'+Math.random().toString(36).slice(2,7);\n"
        "      var dh=det.map(function(fc){\n"
        "        return '<span style=\"font-size:8px;margin-right:3px;\">· '+SPECIALIST_DETAIL_LABELS[fc]+'</span>';\n"
        "      }).join('');\n"
        "      if(specOk){\n"
        # accessible이지만 세부 inaccessible 있음 → 초록 태그 + ℹ
        "        html+='<span class=\"cp-inacc-tag\" style=\"background:#E8F5E9;color:#2E7D32;border-color:#A5D6A7;cursor:default;\">'\n"
        "          +'✓ '+SPECIALIST_LABEL\n"
        "          +' <span onclick=\"(function(){var e=document.getElementById(\\''+tid+'\\');e.style.display=e.style.display===\\'none\\'?\\'block\\':\\'none\\';})()\" '\n"
        "          +'style=\"cursor:pointer;font-size:8px;color:#2E7D32;border:1px solid #A5D6A7;border-radius:3px;padding:0 3px;margin-left:2px;\" title=\"Show inaccessible sub-types\">ℹ</span>'\n"
        "          +'</span>'\n"
        "          +(det.length>0\n"
        "            ?'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:#E8F5E9;border-radius:4px;padding:3px 6px;font-size:8px;color:#2E7D32;\">Inaccessible: '+dh+'</div>'\n"
        "            :'');\n"
        "      } else {\n"
        # inaccessible → 주황 태그 + ℹ
        "        html+='<span class=\"cp-inacc-tag\" style=\"cursor:default;\">'\n"
        "          +SPECIALIST_LABEL\n"
        "          +' <span onclick=\"(function(){var e=document.getElementById(\\''+tid+'\\');e.style.display=e.style.display===\\'none\\'?\\'block\\':\\'none\\';})()\" '\n"
        "          +'style=\"cursor:pointer;font-size:8px;color:#8E24AA;border:1px solid #CE93D8;border-radius:3px;padding:0 3px;margin-left:2px;\" title=\"Show inaccessible sub-types\">ℹ</span>'\n"
        "          +'</span>'\n"
        "          +(det.length>0\n"
        "            ?'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:#F3E5F5;border-radius:4px;padding:3px 6px;font-size:8px;color:#6A1B9A;\">Inaccessible: '+dh+'</div>'\n"
        "            :'');\n"
        "      }\n"
        "    }\n"
        "    if(!html)return '<span class=\"cp-inacc-ok\">✓ All accessible</span>';\n"
        "    return html;\n"
        "  }\n"
        "  (function(){\n"
        "    if(!acc){\n"
        "      tagsEl.innerHTML='<span style=\"font-size:8.5px;color:#bbb;\">OD data not available</span>';\n"
        "      return;\n"
        "    }\n"
        "    var hasCovSlot=FAC_OD_COLS.some(function(fc){return acc['cov_08_'+fc]!==undefined;});\n"
        "    tagsEl.innerHTML=buildInaccHTML(function(fc){\n"
        "      if(hasCovSlot){\n"
        "        return SLOT_KEYS.some(function(s){return acc['cov_'+s+'_'+fc]===1;});\n"
        "      } else {\n"
        "        return !(acc['pt_'+fc]===0||acc['pt_'+fc]===false);\n"
        "      }\n"
        "    });\n"
        "  })();\n"
        # Chart
        "  var covVals=COV_COLS.map(function(c){var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  var maiVals=MAI_COLS.map(function(c){var v=d[c];return(v!=null&&!isNaN(parseFloat(v)))?parseFloat(v):null;});\n"
        "  if(_cpChart){_cpChart.destroy();_cpChart=null;}\n"
        "  var ctx=document.getElementById('cp-chart').getContext('2d');\n"
        "  var covBySlot={},maiBySlot={};\n"
        "  SLOT_KEYS.forEach(function(s){\n"
        "    var cv=d[s+'_coverage'],mv=d[s+'_mai'];\n"
        "    covBySlot[s]=(cv!=null&&!isNaN(parseFloat(cv)))?parseFloat(cv):null;\n"
        "    maiBySlot[s]=(mv!=null&&!isNaN(parseFloat(mv)))?parseFloat(mv):null;\n"
        "  });\n"
        # ── getCovRawAcc / getCovInacc / getMaiInacc / renderInaccItems (specialist 묶음) ──
        # specialist는 항상 items에 포함. specOk=true여도 세부 inaccessible 과목 표시
        "  function getCovRawAcc(slot){\n"
        "    var hasCovSlot=acc&&FAC_OD_COLS.some(function(fc){return acc['cov_'+slot+'_'+fc]!==undefined;});\n"
        "    var hasPtFac=acc&&FAC_OD_COLS.some(function(fc){return acc['pt_'+fc]!==undefined;});\n"
        "    var raw={};\n"
        "    FAC_OD_COLS.forEach(function(fc){\n"
        "      if(!acc){raw[fc]=false;return;}\n"
        "      if(hasCovSlot){raw[fc]=acc['cov_'+slot+'_'+fc]===1;}\n"
        "      else if(hasPtFac){raw[fc]=!(acc['pt_'+fc]===0||acc['pt_'+fc]===false);}\n"
        "      else{raw[fc]=false;}\n"
        "    });\n"
        "    return raw;\n"
        "  }\n"
        "  function getCovInacc(slot){\n"
        "    var covV=covBySlot[slot];\n"
        "    if(!acc){\n"
        "      if(covV===null||covV===0)return {items:DISPLAY_FAC_COLS.map(function(dc){return {dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]};}).filter(function(){return true;}),unknown:false};\n"
        "      if(covV>=99.99)return {items:[],unknown:false};\n"
        "      return {items:[],unknown:true};\n"
        "    }\n"
        "    var raw=getCovRawAcc(slot);\n"
        "    var items=[];\n"
        "    DISPLAY_FAC_COLS.forEach(function(dc){\n"
        "      if(dc==='specialist'){\n"
        # specialist: specOk 계산, 항상 push (accessible이더라도)
        "        var specOk=SPECIALIST_COLS.some(function(fc){return raw[fc]===true;});\n"
        "        var det=SPECIALIST_COLS.filter(function(fc){return !raw[fc];});\n"
        # accessible이고 세부 inaccessible도 없으면 skip (완전 접근 가능)
        "        if(specOk&&det.length===0)return;\n"
        "        items.push({dc:'specialist',label:SPECIALIST_LABEL,accessible:specOk,specDetail:det});\n"
        "      } else {\n"
        "        if(!raw[dc])items.push({dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]});\n"
        "      }\n"
        "    });\n"
        "    return {items:items,unknown:false};\n"
        "  }\n"
        "  function getMaiInacc(){\n"
        "    var maiV=d.avg_mai;\n"
        "    if(!acc){\n"
        "      if(maiV===null||maiV===0)return {items:DISPLAY_FAC_COLS.map(function(dc){return {dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]};}).filter(function(){return true;}),unknown:false,tie:false};\n"
        "      if(maiV>=99.99)return {items:[],unknown:false,tie:false};\n"
        "      return {items:[],unknown:true,tie:false};\n"
        "    }\n"
        "    var hasNewMai=FAC_OD_COLS.some(function(fc){return acc['mai_'+fc]!==undefined;});\n"
        "    var hasPtFacM=FAC_OD_COLS.some(function(fc){return acc['pt_'+fc]!==undefined;});\n"
        "    var rawM={};\n"
        "    FAC_OD_COLS.forEach(function(fc){\n"
        "      if(hasNewMai){rawM[fc]=acc['mai_'+fc]===1;}\n"
        "      else if(hasPtFacM){rawM[fc]=!(acc['pt_'+fc]===0||acc['pt_'+fc]===false);}\n"
        "      else{rawM[fc]=false;}\n"
        "    });\n"
        "    var items=[];\n"
        "    DISPLAY_FAC_COLS.forEach(function(dc){\n"
        "      if(dc==='specialist'){\n"
        "        var specOkM=SPECIALIST_COLS.some(function(fc){return rawM[fc]===true;});\n"
        "        var detM=SPECIALIST_COLS.filter(function(fc){return !rawM[fc];});\n"
        "        if(specOkM&&detM.length===0)return;\n"
        "        items.push({dc:'specialist',label:SPECIALIST_LABEL,accessible:specOkM,specDetail:detM});\n"
        "      } else {\n"
        "        if(!rawM[dc])items.push({dc:dc,label:DISPLAY_FAC_LABELS[dc]||dc,accessible:false,specDetail:[]});\n"
        "      }\n"
        "    });\n"
        "    return {items:items,unknown:false,tie:acc['mai_is_tie']===1};\n"
        "  }\n"
        # renderInaccItems: accessible specialist → 초록 태그 + ℹ, inaccessible → 빨강/청록 태그 + ℹ
        "  function renderInaccItems(items,pillBg,pillColor,pillBorder,detBg,detColor){\n"
        "    if(!items||items.length===0)return '';\n"
        "    return items.map(function(item){\n"
        "      if(item.dc==='specialist'){\n"
        "        var tid='sd'+Math.random().toString(36).slice(2,7);\n"
        # accessible 여부에 따라 태그 색상 결정
        "        var bg=item.accessible?'#E8F5E9':pillBg;\n"
        "        var fg=item.accessible?'#2E7D32':pillColor;\n"
        "        var bd=item.accessible?'#A5D6A7':pillBorder;\n"
        "        var ps='font-size:8.5px;background:'+bg+';color:'+fg+';border:1px solid '+bd+';border-radius:8px;padding:1px 5px;display:inline-block;';\n"
        "        var prefix=item.accessible?'✓ ':'';\n"
        "        var dh=item.specDetail.map(function(fc){return'<span style=\"font-size:8px;margin-right:3px;\">· '+SPECIALIST_DETAIL_LABELS[fc]+'</span>';}).join('');\n"
        "        return '<span style=\"'+ps+'cursor:default;\">'+prefix+item.label\n"
        "          +' <span onclick=\"(function(){var e=document.getElementById(\\''+tid+'\\');e.style.display=e.style.display===\\'none\\'?\\'block\\':\\'none\\';})()\" '\n"
        "          +'style=\"cursor:pointer;font-size:7.5px;color:'+fg+';border:1px solid '+bd+';border-radius:3px;padding:0 3px;margin-left:2px;\" title=\"Show inaccessible sub-types\">ℹ</span>'\n"
        "          +'</span>'\n"
        "          +(item.specDetail.length>0\n"
        "            ?'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:'+detBg+';border-radius:4px;padding:2px 6px;font-size:8px;color:'+detColor+';\">'+'Inaccessible: '+dh+'</div>'\n"
        "            :'<div id=\"'+tid+'\" style=\"display:none;width:100%;margin-top:2px;background:#E8F5E9;border-radius:4px;padding:2px 6px;font-size:8px;color:#2E7D32;\">All sub-types accessible</div>'\n"
        "          );\n"
        "      }\n"
        "      var ps='font-size:8.5px;background:'+pillBg+';color:'+pillColor+';border:1px solid '+pillBorder+';border-radius:8px;padding:1px 5px;display:inline-block;';\n"
        "      return '<span style=\"'+ps+'\">'+item.label+'</span>';\n"
        "    }).join('');\n"
        "  }\n"
        "  _cpChart=new Chart(ctx,{type:'line',"
        "    data:{labels:TIME_SLOTS,datasets:["
        "      {label:'Coverage (%)',data:covVals,borderColor:'" + COV_LINE_COLOR + "',"
        "backgroundColor:'" + COV_LINE_COLOR + "15',"
        "pointBackgroundColor:'" + COV_LINE_COLOR + "',pointBorderColor:'#fff',pointBorderWidth:1.2,"
        "pointRadius:4,pointHoverRadius:6,borderWidth:2,tension:0.15,fill:false},"
        "      {label:'MAI (%)',data:maiVals,borderColor:'" + MAI_LINE_COLOR + "',"
        "backgroundColor:'" + MAI_LINE_COLOR + "15',"
        "pointBackgroundColor:'" + MAI_LINE_COLOR + "',pointBorderColor:'#fff',pointBorderWidth:1.2,"
        "pointRadius:4,pointHoverRadius:6,borderWidth:2,tension:0.15,fill:false}"
        "    ]},"
        "    options:{responsive:true,maintainAspectRatio:false,"
        # mode:'index' + intersect:false → x축 어디서든 해당 시간대 감지
        "      interaction:{mode:'index',intersect:false,axis:'x'},"
        # top padding 넉넉히 줘서 100% 점 잘림 방지
        "      layout:{padding:{top:12,bottom:2,left:0,right:0}},"
        "      plugins:{"
        "        tooltip:{"
        "          enabled:false,"
        "          external:function(context){"
        "            var wrap=document.getElementById('cp-chart-wrap');"
        "            var tip=document.getElementById('cp-tooltip');"
        "            if(!tip){"
        "              tip=document.createElement('div');tip.id='cp-tooltip';"
        "              tip.style.cssText='position:absolute;background:#fff;border:1px solid #e0e0e0;"
        "border-radius:7px;padding:9px 11px;font-size:10px;line-height:1.55;"
        "box-shadow:0 3px 12px rgba(0,0,0,0.13);z-index:9999;width:230px;"
        "word-break:keep-all;white-space:normal;pointer-events:none;';"
        "              wrap.appendChild(tip);"
        "            }"
        "            var model=context.tooltip;"
        # opacity===0이면 숨기되, 짧은 지연 후 숨겨서 깜빡임 방지
        "            if(model.opacity===0){"
        "              if(tip._hideTimer)clearTimeout(tip._hideTimer);"
        "              tip._hideTimer=setTimeout(function(){tip.style.display='none';},120);"
        "              return;"
        "            }"
        "            if(tip._hideTimer){clearTimeout(tip._hideTimer);tip._hideTimer=null;}"
        "            var idx=model.dataPoints&&model.dataPoints[0]?model.dataPoints[0].dataIndex:null;"
        "            if(idx===null){tip.style.display='none';return;}"
        "            var slot=SLOT_KEYS[idx];"
        "            var covV=covBySlot[slot],maiV=maiBySlot[slot];"
        "            var inaccCov=getCovInacc(slot);"
        "            var inaccMai=getMaiInacc();"
        "            var h='';"
        "            h+='<div style=\"font-weight:700;font-size:11px;color:#222;margin-bottom:6px;"
        "border-bottom:1.5px solid #f0f0f0;padding-bottom:4px;\">'+TIME_SLOTS[idx]+'</div>';"
        "            h+='<div style=\"display:flex;align-items:center;gap:5px;margin-bottom:1px;\">';"
        "            h+='<span style=\"width:9px;height:9px;border-radius:50%;background:" + COV_LINE_COLOR + ";flex-shrink:0;\"></span>';"
        "            h+='<span style=\"font-weight:700;color:" + COV_LINE_COLOR + ";font-size:10px;\">Coverage: '+(covV!=null?covV.toFixed(1)+'%':'N/A')+'</span></div>';"
        "            h+='<div style=\"font-size:8px;color:#888;margin-left:14px;margin-bottom:3px;\">Inaccessible within threshold</div>';"
        "            if(inaccCov.unknown){h+='<div style=\"font-size:9px;color:#B0651A;margin-left:14px;margin-bottom:5px;\">⚠ Details unavailable</div>';}"
        "            else if(inaccCov.items.length===0){h+='<div style=\"font-size:9px;color:#43A047;margin-left:14px;margin-bottom:5px;\">✓ All accessible</div>';}"
        "            else{h+='<div style=\"margin-left:14px;margin-bottom:5px;display:flex;flex-wrap:wrap;gap:2px;\">'+renderInaccItems(inaccCov.items,'#FFEBEE','#b71c1c','#ef9a9a','#FFEBEE','#b71c1c')+'</div>';}"
        "            h+='<div style=\"border-top:1px solid #f5f5f5;padding-top:5px;margin-top:2px;\">';"
        "            h+='<div style=\"display:flex;align-items:center;gap:5px;margin-bottom:1px;\">';"
        "            h+='<span style=\"width:9px;height:9px;border-radius:50%;background:" + MAI_LINE_COLOR + ";flex-shrink:0;\"></span>';"
        "            h+='<span style=\"font-weight:700;color:" + MAI_LINE_COLOR + ";font-size:10px;\">MAI: '+(maiV!=null?maiV.toFixed(1)+'%':'N/A')+'</span></div>';"
        "            var maiSub='Best reachable grid (within 15 min)'+(inaccMai.tie?' *':'');"
        "            h+='<div style=\"font-size:8px;color:#888;margin-left:14px;margin-bottom:3px;\">'+maiSub+'</div>';"
        "            if(inaccMai.tie){h+='<div style=\"font-size:7.5px;color:#aaa;margin-left:14px;margin-bottom:2px;\">* Nearest of tied grids</div>';}"
        "            if(inaccMai.unknown){h+='<div style=\"font-size:9px;color:#B0651A;margin-left:14px;\">⚠ Details unavailable</div>';}"
        "            else if(inaccMai.items.length===0){h+='<div style=\"font-size:9px;color:#43A047;margin-left:14px;\">✓ All facility types present</div>';}"
        "            else{h+='<div style=\"margin-left:14px;display:flex;flex-wrap:wrap;gap:2px;\">'+renderInaccItems(inaccMai.items,'#E0F2F1','#00695C','#80CBC4','#E0F2F1','#00695C')+'</div>';}"
        "            h+='</div>';"
        "            tip.innerHTML=h;"
        "            var wrapW=wrap.offsetWidth||240;"
        "            var posX=model.caretX+14;"
        "            if(posX+235>wrapW)posX=model.caretX-244;"
        "            if(posX<0)posX=2;"
        "            var posY=model.caretY-20;"
        "            if(posY<0)posY=2;"
        "            tip.style.left=posX+'px';"
        "            tip.style.top=posY+'px';"
        "            tip.style.display='block';"
        "          }"
        "        },"
        "        legend:{position:'top',align:'start',labels:{font:{size:9},boxWidth:9,padding:6}}"
        "      },"
        "      scales:{"
        "        x:{grid:{color:'#f4f4f4'},ticks:{font:{size:8}}},"
        # y 최대 103으로 → 100% 점 상단 잘림 방지, suggestedMax로 여유 확보
        "        y:{min:0,max:103,grid:{color:'#f4f4f4'},ticks:{font:{size:8},callback:function(v){return v<=100?v+'%':'';},"
        "stepSize:20}}"
        "      }"
        "    }});\n"
        "  panel.classList.add('visible');\n"
        "  allMaps.forEach(function(m){m.invalidateSize();});\n"
        "}\n"
        "document.getElementById('cp-close').addEventListener('click',function(){\n"
        "  document.getElementById('cell-panel').classList.remove('visible');\n"
        "  if(_cpChart){_cpChart.destroy();_cpChart=null;}\n"
        "  if(window._hlGeoJSON){allMaps.forEach(function(m){try{m.removeLayer(window._hlGeoJSON);}catch(ex){}});window._hlGeoJSON=null;}\n"
        "  allMaps.forEach(function(m){m.invalidateSize();});\n"
        "});\n"
    )


    panel_html = (
        '<div class="cell-panel" id="cell-panel">'
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;">'
        '<div style="flex:1;min-width:0;"><div class="cp-id-sub">Selected Cell</div>'
        '<div class="cp-id" id="cp-fid"></div></div>'
        '<span class="cp-close" id="cp-close">&#x2715;</span></div>'
        # PT
        '<div class="cp-sec" style="color:#3F51B5;">Public Transit</div>'
        '<div class="cp-metrics">'
        '<div class="cp-m"><div class="cp-m-label">avg. Coverage</div><div class="cp-m-val" id="cp-avg-cov">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">avg. MAI</div><div class="cp-m-val" id="cp-avg-mai">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">CV Coverage</div><div class="cp-m-val" id="cp-cv-cov">—</div></div>'
        '<div class="cp-m"><div class="cp-m-label">CV MAI</div><div class="cp-m-val" id="cp-cv-mai">—</div></div>'
        '</div>'
        # SGG avg
        '<div class="cp-sec" style="color:#3F51B5;">Municipality Avg.</div>'
        '<div class="cp-sgg-ref">'
        '<div class="cp-sgg-m"><div class="cp-sgg-label">City Coverage</div><div class="cp-sgg-val" id="cp-sgg-cov">—</div></div>'
        '<div class="cp-sgg-m"><div class="cp-sgg-label">City MAI</div><div class="cp-sgg-val" id="cp-sgg-mai">—</div></div>'
        '</div>'
        # Deficit types
        '<div class="cp-sec" style="color:#555;margin-top:10px;">Deficit Type</div>'
        '<div class="cp-deficit-tags" id="cp-deficit-tags"></div>'
        # Reference
        '<div class="cp-sec" style="color:#555;">Reference</div>'
        '<div class="cp-ref">'
        '<div class="cp-r"><div class="cp-r-label">Population</div><div class="cp-r-val" id="cp-pop">—</div></div>'
        '<div class="cp-r"><div class="cp-r-label">Car Cov.</div><div class="cp-r-val" id="cp-car-cov">—</div></div>'
        '<div class="cp-r"><div class="cp-r-label">Car MAI</div><div class="cp-r-val" id="cp-car-mai">—</div></div>'
        '</div>'
        # Inaccessible
        '<div class="cp-sec" style="color:#E65100;">Inaccessible Facilities</div>'
        '<div class="cp-inacc-tags" id="cp-inacc-tags"></div>'
        # Chart
        '<div class="cp-sec" style="color:#555;margin-top:10px;">Time-of-Day Profile</div>'
        '<div id="cp-chart-wrap" style="position:relative;margin-top:2px;"><canvas id="cp-chart"></canvas></div>'
        '</div>'
    )



    html_parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8"/>',
        '<style>',
        '*{box-sizing:border-box;margin:0;padding:0;}',
        'body{background:#fff;font-family:"Inter",system-ui,sans-serif;}',
        panel_css,
        '</style>',
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>',
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>',
        chart_cdn,
        '</head><body>',
        '<div class="outer-wrap">',
        '<div class="maps-section">',
        '<div class="maps-grid">',
        map_divs,
        '</div>',
        '<div class="fac-ctrl" id="fc"><div class="fac-hdr">Facilities</div></div>',
        '</div>',
        panel_html,
        '</div>',
        '<script>',
        'var gridData=[' + ",".join(grid_js_list) + '];',
        'var subwayData='  + subway_js  + ';',
        'var stationData=' + station_js + ';',
        'var facData='     + fac_js + ';',
        'var colorbarsData=' + cbars_js_arr + ';',
        # fac_visible: localStorage 우선, 없으면 전부 표시
        '(function(){try{'
        '  var s=localStorage.getItem("facVisible");'
        '  window._initFacVisible=s?JSON.parse(s):null;'
        '}catch(e){window._initFacVisible=null;}})();',
        'var facVisible=window._initFacVisible||' + json.dumps(list(FACILITY_ORDER)) + ';',
        'var deficitInfo=' + json.dumps({
            k: {"c": v, "label": DEFICIT_LABELS[k]}
            for k, v in deficit_colors.items()
        }) + ';',
        'var deficitVisible=[];',
        'var facLayers={};',
        'var deficitLayers={};',
        'var gridLayers=[];',
        'var allMaps=[];',
        'window._hlGeoJSON=null;',
        'function getBBox(geom){'
        'var mn=[Infinity,Infinity],mx=[-Infinity,-Infinity];'
        'function pt(c){if(c[0]<mn[0])mn[0]=c[0];if(c[1]<mn[1])mn[1]=c[1];'
        'if(c[0]>mx[0])mx[0]=c[0];if(c[1]>mx[1])mx[1]=c[1];}'
        'function ring(r){for(var i=0;i<r.length;i++)pt(r[i]);}'
        'var t=geom.type,c=geom.coordinates;'
        'if(t==="Polygon")for(var i=0;i<c.length;i++)ring(c[i]);'
        'else if(t==="MultiPolygon")for(var i=0;i<c.length;i++)for(var j=0;j<c[i].length;j++)ring(c[i][j]);'
        'return[mn[0],mn[1],mx[0],mx[1]];}',
        'function pointInRing(lat,lng,ring){'
        'var inside=false;'
        'for(var i=0,j=ring.length-1;i<ring.length;j=i++){'
        'var xi=ring[i][0],yi=ring[i][1],xj=ring[j][0],yj=ring[j][1];'
        'if(((yi>lat)!=(yj>lat))&&(lng<(xj-xi)*(lat-yi)/(yj-yi)+xi))inside=!inside;}'
        'return inside;}',
        'function pointInPolygon(lat,lng,geom){'
        'var t=geom.type,c=geom.coordinates;'
        'if(t==="Polygon"){return pointInRing(lat,lng,c[0]);}'
        'if(t==="MultiPolygon"){for(var i=0;i<c.length;i++){if(pointInRing(lat,lng,c[i][0]))return true;}return false;}'
        'return false;}',
        maps_init,
        sync_js,
        fac_toggle_js,
        'window.toggleDeficit=function(dk,on){'
        '  if(!deficitLayers[dk])return;'
        '  deficitLayers[dk].maps.forEach(function(x){'
        '    if(on){if(x.whiteLayer)x.whiteLayer.addTo(x.map);x.layer.addTo(x.map);}'
        '    else{if(x.whiteLayer)x.map.removeLayer(x.whiteLayer);x.map.removeLayer(x.layer);}'
        '  });'
        '};'
        # basis 변경: grid recolor + colorbar 교체 + deficit recolor (뷰포트 유지)
        'window.updateDeficitBasis=function(newBasis){'
        '  window.deficitBasis=newBasis;'
        '  gridLayers.forEach(function(lyr){lyr.setStyle(lyr.options.style);});'
        '  colorbarsData.forEach(function(cb,i){'
        '    var el=document.getElementById("cb"+i);'
        '    if(el)el.innerHTML=cb[newBasis]||cb["sgg"];'
        '  });'
        '  Object.keys(deficitLayers).forEach(function(dk){'
        '    deficitLayers[dk].maps.forEach(function(x){'
        '      if(!x.makeStyleFns)return;'
        '      var fns=x.makeStyleFns(newBasis);'
        '      x.whiteLayer.setStyle(fns.white);x.layer.setStyle(fns.color);'
        '    });'
        '  });'
        '};'
        'window.addEventListener("storage",function(ev){'
        '  if(ev.key==="deficitState"&&ev.newValue){'
        '    try{var st=JSON.parse(ev.newValue);'
        '      Object.keys(st).forEach(function(dk){window.toggleDeficit(dk,st[dk]);});'
        '    }catch(e){}'
        '  }'
        '  if(ev.key==="deficitBasis"&&ev.newValue){'
        '    try{window.updateDeficitBasis(ev.newValue);}catch(e){}'
        '  }'
        '});'
        'window.addEventListener("message",function(ev){'
        '  try{'
        '    if(ev.data&&ev.data.type==="deficitState"){'
        '      var st=ev.data.state;'
        '      Object.keys(st).forEach(function(dk){window.toggleDeficit(dk,st[dk]);});'
        '    }'
        '  }catch(e){}'
        '});'
        '(function(){'
        '  try{'
        '    var b=localStorage.getItem("deficitBasis");'
        '    if(b&&b!=="sgg"){window.deficitBasis=b;window.updateDeficitBasis(b);}'
        '    else{window.deficitBasis="sgg";}'
        '    var s=localStorage.getItem("deficitState");'
        '    if(s){var st=JSON.parse(s);Object.keys(st).forEach(function(dk){if(st[dk])window.toggleDeficit(dk,true);});}'
        '  }catch(e){}'
        '})();',
        panel_js,
        '</script></body></html>',
    ]
    return "\n".join(html_parts)


# =========================================================
# norm / 값 컬럼
# =========================================================
def get_value_col(metric_key: str, basis_key: str) -> str:
    if metric_key == "pop":      return "local_pop_map" if basis_key == "sgg" else "nat_pop_map"
    if metric_key == "coverage": return "avg_coverage"
    if metric_key == "mai":      return "avg_mai"
    return f"{basis_key}_{metric_key}_ratio"

def get_norm_for_group(group: gpd.GeoDataFrame, metric_key: str, basis_key: str):
    """norm 계산 (Local 스케일 고정 — scale_mode 제거됨)."""
    if metric_key == "coverage":
        return compute_continuous_norm(group.get("avg_coverage", pd.Series([], dtype=float)), gamma=0.6)[2]
    if metric_key == "mai":
        return compute_continuous_norm(group.get("avg_mai", pd.Series([], dtype=float)), gamma=0.6)[2]
    if metric_key == "pop":
        return compute_group_pop_norm(group.get("local_pop_map" if basis_key == "sgg" else "nat_pop_map", pd.Series([], dtype=float)), share_mode=True)[2]
    vals = pd.concat([pd.to_numeric(group.get(f"{basis_key}_{k}_ratio", pd.Series()), errors="coerce") for k in ["fs","fd","tc","tf"]], axis=0)
    return compute_group_norm_from_series(vals, gamma=0.55, force_zero_min=True)[2]


def render_metric_maps(
    map_prefix: str,
    group_gdf: gpd.GeoDataFrame,
    aggregate_gdf: gpd.GeoDataFrame,
    selected_metrics: List[str],
    basis_key: str,
    station_gdf, subway_gdf, fac_gdf,
    initial_center: Tuple[float, float],
    initial_zoom: int = 11,
    click_source_gdf: Optional[gpd.GeoDataFrame] = None,
    selected_sgg_code: str = "",
    compare_partner_gdf: Optional[gpd.GeoDataFrame] = None,
    cell_df: Optional[pd.DataFrame] = None,
    fac_access_df: Optional[pd.DataFrame] = None,
):
    if "fac_visible" not in st.session_state:
        st.session_state["fac_visible"] = list(FACILITY_ORDER)

    # basis_key는 sidebar에서 localStorage로 JS에 전달되므로 여기선 cell_data 용도로만 사용

    # ── cell data JSON (sgg_avg 포함) ─────────────────
    cell_data_json = "{}"
    data_src = cell_df if (cell_df is not None and not cell_df.empty) else None
    if data_src is not None:
        if SGG_CODE_COL in data_src.columns:
            sub = data_src[data_src[SGG_CODE_COL].apply(normalize_sgg_code) == normalize_sgg_code(str(selected_sgg_code))]
        else:
            valid_ids = set(group_gdf[GRID_JOIN_COL].astype(str).str.strip())
            sub = data_src[data_src[GRID_JOIN_COL].isin(valid_ids)]

        if not sub.empty:
            # sgg_avg: from group_gdf (캐시된 값 우선, 없으면 계산)
            sgg_cov = sgg_mai = None
            # sgg_avg_coverage
            if "sgg_avg_coverage" in group_gdf.columns:
                vals = group_gdf["sgg_avg_coverage"].dropna()
                if len(vals): sgg_cov = float(vals.iloc[0])
            elif "sgg_avg_coverage" in sub.columns:
                vals = sub["sgg_avg_coverage"].dropna()
                if len(vals): sgg_cov = float(vals.iloc[0])
            # sgg_avg_mai
            if "sgg_avg_mai" in group_gdf.columns:
                vals = group_gdf["sgg_avg_mai"].dropna()
                if len(vals): sgg_mai = float(vals.iloc[0])
            elif "sgg_avg_mai" in sub.columns:
                vals = sub["sgg_avg_mai"].dropna()
                if len(vals): sgg_mai = float(vals.iloc[0])
            # 재계산 fallback: avg_coverage * pop / total_pop
            if sgg_cov is None and "avg_coverage" in sub.columns and "pop" in sub.columns:
                pop_s = pd.to_numeric(sub["pop"], errors="coerce").fillna(0)
                cov_s = pd.to_numeric(sub["avg_coverage"], errors="coerce")
                total = pop_s.sum()
                if total > 0: sgg_cov = float((cov_s * pop_s).sum() / total)
            if sgg_mai is None and "avg_mai" in sub.columns and "pop" in sub.columns:
                pop_s = pd.to_numeric(sub["pop"], errors="coerce").fillna(0)
                mai_s = pd.to_numeric(sub["avg_mai"], errors="coerce")
                total = pop_s.sum()
                if total > 0: sgg_mai = float((mai_s * pop_s).sum() / total)

            wanted = [GRID_JOIN_COL, "pop", "avg_coverage", "avg_mai",
                      "cv_coverage", "cv_mai", "car_coverage", "car_mai"] + COV_COLS + MAI_COLS
            sub = sub[[c for c in wanted if c in sub.columns]].copy()

            # 숫자 컬럼 float 변환 + inf/nan → None (vectorized)
            data_cols = [c for c in sub.columns if c != GRID_JOIN_COL]
            for c in data_cols:
                if sub[c].dtype.kind in ("f", "i", "u"):
                    sub[c] = pd.to_numeric(sub[c], errors="coerce")
            sub = sub.replace([np.inf, -np.inf], np.nan)

            # deficit 플래그: group_gdf에서 한 번에 join
            deficit_flag_cols = []
            for dk in ["fs", "fd", "tc", "tf"]:
                for bp in ["sgg", "nat"]:
                    col = f"{bp}_has_{dk}"
                    if col in group_gdf.columns:
                        deficit_flag_cols.append(col)
            if deficit_flag_cols and GRID_JOIN_COL in group_gdf.columns:
                def_ref = group_gdf[[GRID_JOIN_COL] + deficit_flag_cols].copy()
                def_ref[GRID_JOIN_COL] = def_ref[GRID_JOIN_COL].astype(str).str.strip()
                for c in deficit_flag_cols:
                    def_ref[c] = def_ref[c].astype(bool)
                sub = sub.merge(def_ref, on=GRID_JOIN_COL, how="left")
                for c in deficit_flag_cols:
                    sub[c] = sub[c].fillna(False)

            # sgg_avg 상수 컬럼 추가
            sub["sgg_avg_coverage"] = sgg_cov
            sub["sgg_avg_mai"]      = sgg_mai

            # to_dict → JSON (pandas가 NaN을 None으로 못 변환하므로 직접 처리)
            sub[GRID_JOIN_COL] = sub[GRID_JOIN_COL].astype(str)
            records = sub.where(sub.notna(), other=None).to_dict(orient="records")
            cell_dict = {r[GRID_JOIN_COL]: {k: v for k, v in r.items() if k != GRID_JOIN_COL} for r in records}
            cell_data_json = json.dumps(cell_dict, default=lambda x: None if (isinstance(x, float) and (x != x or abs(x) == float('inf'))) else x)

    # ── facility access JSON ─────────────────────────
    fac_access_json = "{}"
    if fac_access_df is not None and not fac_access_df.empty:
        valid_ids = set(group_gdf[GRID_JOIN_COL].astype(str).str.strip())
        fac_sub   = fac_access_df[fac_access_df[GRID_JOIN_COL].isin(valid_ids)].copy()
        if not fac_sub.empty:
            new_cov_cols = [c for c in fac_sub.columns if c.startswith("cov_")]
            new_mai_cols = [c for c in fac_sub.columns if c.startswith("mai_") and c not in ("mai_is_tie","mai_best_to_id")]
            legacy_cols  = [c for c in fac_sub.columns if c.startswith("pt_")]
            keep_cols    = [GRID_JOIN_COL] + new_cov_cols + new_mai_cols + legacy_cols
            if "mai_is_tie"     in fac_sub.columns: keep_cols.append("mai_is_tie")
            if "mai_best_to_id" in fac_sub.columns: keep_cols.append("mai_best_to_id")
            fac_sub = fac_sub[[c for c in keep_cols if c in fac_sub.columns]].copy()
            # int8/int16 컬럼 → int, str 컬럼 유지
            for c in fac_sub.columns:
                if c in (GRID_JOIN_COL, "mai_best_to_id"): continue
                if fac_sub[c].dtype.kind in ("i", "u", "b"):
                    fac_sub[c] = fac_sub[c].astype(int)
            fac_sub[GRID_JOIN_COL] = fac_sub[GRID_JOIN_COL].astype(str)
            records = fac_sub.where(fac_sub.notna(), other=None).to_dict(orient="records")
            fac_dict = {r[GRID_JOIN_COL]: {k: v for k, v in r.items() if k != GRID_JOIN_COL} for r in records}
            fac_access_json = json.dumps(fac_dict)

    n        = len(selected_metrics)
    rows     = 1 if n <= 2 else 2
    iframe_h = (MAP_HEIGHT + 42 + 26) * rows + 16

    html = build_multi_map_html(
        sgg_code=str(selected_sgg_code),
        metric_keys_str="|".join(selected_metrics),
        center_lat=initial_center[0],
        center_lng=initial_center[1],
        zoom=initial_zoom,
        height_px=MAP_HEIGHT,
        deficit_colors_json=json.dumps(DEFICIT_COLORS),
    )
    html_final = html.replace(
        '</body></html>',
        f'<script>'
        f'window.cellData={cell_data_json};'
        f'window.facAccessData={fac_access_json};'
        f'</script></body></html>',
    )
    st_components.html(html_final, height=iframe_h, scrolling=False)


# =========================================================
# 앱 시작
# =========================================================
st.set_page_config(page_title="PT Deficit Dashboard", layout="wide")
st.markdown(
    '<h1 style="font-size:20px;font-weight:700;color:#1a1a1a;margin-bottom:2px;">'
    'PT Accessibility Deficit Dashboard</h1>'
    '<p style="font-size:12px;color:#999;margin-bottom:14px;">Grid-level public transit accessibility analysis</p>',
    unsafe_allow_html=True,
)

required_paths = [CLASSIFIED_PATH, GRID_PATH, STATION_PATH, SUBWAY_PATH, FAC_PATH]
missing_paths  = [str(p) for p in required_paths if not p.exists()]
if missing_paths:
    st.error("Required input files missing:\n\n" + "\n".join(missing_paths))
    st.stop()


def _dropbox_safe_clear_geojson_dir():
    import os, time
    if not CACHE_GEOJSON_DIR.exists():
        CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True); return
    for f in list(CACHE_GEOJSON_DIR.iterdir()):
        if f.is_file():
            for attempt in range(6):
                try: f.unlink(missing_ok=True); break
                except PermissionError: time.sleep(0.4 * (attempt + 1))
    CACHE_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)


def _run_build_with_progress():
    _dropbox_safe_clear_geojson_dir()
    for fn in [load_cached_data, load_cell_detail_data, load_facility_access_data]:
        fn.clear()
    try: build_multi_map_html.clear()
    except: pass
    status = st.empty(); pbar = st.progress(0); stxt = st.empty()
    status.info("⚙️ Building cache...")
    pbar.progress(5)
    def _pcb(step, total, msg):
        pbar.progress(min(int(10 + 85 * step / max(total, 1)), 95))
        stxt.caption("🗺️ " + msg)
    build_dashboard_cache(progress_cb=_pcb)
    pbar.progress(100); stxt.empty()
    status.success("✅ Cache build complete!")


with st.sidebar:
    st.header("Setup")
    if st.button("Build / refresh cached data", use_container_width=True, key="btn_refresh_cache"):
        _run_build_with_progress()
        st.rerun()

_tiles_ok   = CACHE_GEOJSON_DIR.exists() and any(CACHE_GEOJSON_DIR.glob("grid_*.json"))
cache_ready = _tiles_ok and all(p.exists() for p in [CACHE_GRID, CACHE_SGG, CACHE_STATION, CACHE_SUBWAY, CACHE_FAC, CACHE_TS, CACHE_IDX])
if not cache_ready:
    st.info("⚙️ First run — building cache (this may take a few minutes)...")
    _pbar = st.progress(0); _stxt = st.empty()
    def _auto_pcb(step, total, msg):
        _pbar.progress(int(100 * step / max(total, 1)))
        _stxt.caption("🗺️ " + msg)
    build_dashboard_cache(progress_cb=_auto_pcb)
    _pbar.progress(100); _stxt.empty()
    st.rerun()

with st.spinner("Loading data..."):
    grid_gdf, grid_simple_gdf, sgg_gdf, station_gdf, subway_gdf, fac_gdf, ts_df, idx_df = load_cached_data()
    cell_df      = load_cell_detail_data()
    fac_access_df = load_facility_access_data()

# sgg_avg merge fallback
if not cell_df.empty and "sgg_avg_coverage" not in cell_df.columns:
    if "sgg_avg_coverage" in grid_gdf.columns:
        sgg_ref  = grid_gdf[[GRID_JOIN_COL, "sgg_avg_coverage", "sgg_avg_mai"]].copy()
        cell_df  = cell_df.merge(sgg_ref, on=GRID_JOIN_COL, how="left")
if not cell_df.empty and SGG_CODE_COL not in cell_df.columns:
    if SGG_CODE_COL in grid_gdf.columns:
        sgg_code_ref = grid_gdf[[GRID_JOIN_COL, SGG_CODE_COL]].copy()
        cell_df      = cell_df.merge(sgg_code_ref, on=GRID_JOIN_COL, how="left")


sgg_options    = sorted(grid_gdf[[SGG_CODE_COL, SGG_NAME_COL]].drop_duplicates().itertuples(index=False, name=None), key=lambda x: x[1])
sgg_name_to_code = {name: code for code, name in sgg_options}

def _sido(n): return n.split("_")[0] if "_" in n else n
def _sgg(n):  return n.split("_",1)[1] if "_" in n else n

sido_list    = sorted(set(_sido(n) for n in sgg_name_to_code))
sido_to_names = {}
for name in sgg_name_to_code:
    sido_to_names.setdefault(_sido(name), []).append(name)
for k in sido_to_names:
    sido_to_names[k] = sorted(sido_to_names[k], key=_sgg)

def sgg_selector(prefix, la="Province", lb="Municipality"):
    sido_sel = st.selectbox(la, sido_list, key=prefix + "_sido")
    names_in = sido_to_names.get(sido_sel, list(sgg_name_to_code))
    disp     = [_sgg(n) for n in names_in]
    sgg_disp = st.selectbox(lb, disp, key=prefix + "_sgg")
    full_key  = sido_sel + "_" + sgg_disp   # 데이터 조회용 (원본 키)
    full_disp = sido_sel + " " + sgg_disp   # 표시용
    return full_disp, sgg_name_to_code.get(full_key)


# ── 사이드바 ──────────────────────────────────────────
with st.sidebar:
    st.header("Display")
    compare_mode = st.toggle("Compare two municipalities", value=False, key="toggle_compare")

    if not compare_mode:
        st.markdown("**Municipality**")
        selected_full, selected_code = sgg_selector("single")
    else:
        st.markdown("**Municipality A**")
        full1, code1 = sgg_selector("cmp_a", "Province A", "City/County A")
        st.markdown("**Municipality B**")
        full2, code2 = sgg_selector("cmp_b", "Province B", "City/County B")

    basis_label = st.selectbox("Benchmark basis",
                               ["Municipality-based benchmark", "National benchmark"],
                               index=0, key="benchmark_basis")
    basis = "sgg" if basis_label.startswith("Municipality") else "nat"

    # basis 변경 → localStorage에 deficitBasis 저장 (iframe 재생성 없이 뷰포트 유지)
    _basis_prev_key = "basis_prev"
    _basis_prev = st.session_state.get(_basis_prev_key, None)
    if _basis_prev is not None and _basis_prev != basis:
        _basis_script = (
            f"<script>(function(){{"
            f"  try{{localStorage.setItem('deficitBasis','{basis}');"
            f"  window.dispatchEvent(new StorageEvent('storage',{{key:'deficitBasis',newValue:'{basis}'}}));"
            f"  }}catch(e){{}}"
            f"}})();</script>"
        )
        st_components.html(_basis_script, height=0, scrolling=False)
    st.session_state[_basis_prev_key] = basis

    st.markdown("---")
    st.markdown("**Base map layers**")
    selected_metric_labels = st.multiselect(
        "Base map layers",
        ["Coverage (avg.)", "MAI (avg.)", "Population"],
        default=["Coverage (avg.)", "MAI (avg.)", "Population"],
        key="base_metric_multiselect",
        label_visibility="collapsed",
    )
    label_to_key_base = {"Population": "pop", "Coverage (avg.)": "coverage", "MAI (avg.)": "mai"}
    if not selected_metric_labels: selected_metric_labels = ["Population"]
    selected_metric_keys = [label_to_key_base[x] for x in selected_metric_labels]

    st.markdown("---")
    st.markdown("**Deficit overlay**")
    st.caption("Overlays deficit cell borders on the base map.")

    # ── Deficit 체크박스: 순수 HTML/JS → Python 재실행 없이 토글 ──────────────
    # 클릭 → localStorage('deficitState') 업데이트 → iframe storage 이벤트 → toggleDeficit
    _deficit_items = [
        ("fs", "F(s) — Facility siting",     DEFICIT_COLORS["fs"]),
        ("fd", "F(d) — Facility dispersion",  DEFICIT_COLORS["fd"]),
        ("tc", "T(c) — Transit connection",   DEFICIT_COLORS["tc"]),
        ("tf", "T(f) — Transit frequency",    DEFICIT_COLORS["tf"]),
    ]
    _checkbox_html = """
<style>
.dcb-row{display:flex;align-items:center;gap:7px;margin-bottom:5px;cursor:pointer;user-select:none;}
.dcb-box{width:14px;height:14px;border:1.5px solid #888;border-radius:3px;
         display:flex;align-items:center;justify-content:center;flex-shrink:0;background:#fff;}
.dcb-box.checked{background:#444;border-color:#444;}
.dcb-box.checked::after{content:'';width:8px;height:8px;background:#fff;
  clip-path:polygon(14% 44%,0 65%,50% 100%,100% 16%,80% 0,43% 62%);display:block;}
.dcb-swatch{width:18px;height:3px;border-radius:2px;flex-shrink:0;}
.dcb-label{font-size:12px;color:#444;}
</style>
<div id="dcb-root">
""" + "".join(
        f'<div class="dcb-row" onclick="dcbToggle(\'{k}\')" id="dcb-row-{k}">'
        f'  <div class="dcb-box" id="dcb-box-{k}"></div>'
        f'  <span class="dcb-swatch" style="background:{color};"></span>'
        f'  <span class="dcb-label">{label}</span>'
        f'</div>'
        for k, label, color in _deficit_items
    ) + """
</div>
<script>
(function(){
  var STATE={fs:false,fd:false,tc:false,tf:false};
  // 기존 localStorage 상태 복원
  try{var s=localStorage.getItem('deficitState');if(s){var p=JSON.parse(s);Object.assign(STATE,p);}}catch(e){}
  function render(){
    Object.keys(STATE).forEach(function(k){
      var box=document.getElementById('dcb-box-'+k);
      if(box){if(STATE[k])box.classList.add('checked');else box.classList.remove('checked');}
    });
  }
  window.dcbToggle=function(k){
    STATE[k]=!STATE[k];
    render();
    try{
      var v=JSON.stringify(STATE);
      localStorage.setItem('deficitState',v);
      // iframe에 직접 postMessage도 함께 전송 (cross-origin fallback)
      document.querySelectorAll('iframe').forEach(function(f){
        try{f.contentWindow.postMessage({type:'deficitState',state:STATE},'*');}catch(e){}
      });
    }catch(e){}
  };
  render();
})();
</script>
"""
    st_components.html(_checkbox_html, height=len(_deficit_items) * 28 + 20, scrolling=False)

    st.markdown("---")
    st.caption("Stations, subway lines, and facility points shown on each map.")


# ── 메인 맵 렌더링 ────────────────────────────────────
def _render(code, full_name, compare_partner_gdf=None):
    group        = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == code].copy()
    group_detail = grid_gdf[grid_gdf[SGG_CODE_COL] == code].copy()
    sgg_group    = sgg_gdf[sgg_gdf[SGG_CODE_COL] == code].copy()
    center       = group_detail.to_crs(WEB_CRS).geometry.centroid.unary_union.centroid
    render_metric_maps(
        map_prefix=f"map_{code}",
        group_gdf=group, aggregate_gdf=sgg_group,
        selected_metrics=selected_metric_keys,
        basis_key=basis, station_gdf=station_gdf, subway_gdf=subway_gdf, fac_gdf=fac_gdf,
        initial_center=(center.y, center.x), initial_zoom=11,
        click_source_gdf=group_detail,
        selected_sgg_code=str(code),
        compare_partner_gdf=compare_partner_gdf,
        cell_df=cell_df,
        fac_access_df=fac_access_df,
    )

if not compare_mode:
    if not selected_code: st.warning("Municipality not found."); st.stop()
    _render(selected_code, selected_full)
else:
    if not code1 or not code2: st.warning("Municipality not found."); st.stop()
    group1 = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == code1].copy()
    group2 = grid_simple_gdf[grid_simple_gdf[SGG_CODE_COL] == code2].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(full1)
        _render(code1, full1, compare_partner_gdf=group2)
    with c2:
        st.subheader(full2)
        _render(code2, full2, compare_partner_gdf=group1)
