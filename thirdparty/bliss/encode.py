import re
import numpy as np
from typing import Any, Dict
import json
from pathlib import Path

# Ordered from oldest → newest
_OS_CODE_NAMES = [
    "Froyo", "Gingerbread", "Honeycomb", "Ice Cream Sandwich",
    "Jelly Bean", "KitKat", "Lollipop", "Marshmallow",
    "Nougat", "Oreo", "Pie"
]
# Map each code-name to a representative *major* Android version
_CODE_TO_MAJOR = {
    "Froyo": 2,  "Gingerbread": 2,  "Honeycomb": 3,  "Ice Cream Sandwich": 4,
    "Jelly Bean": 4,  "KitKat": 4,  "Lollipop": 5,  "Marshmallow": 6,
    "Nougat": 7,  "Oreo": 8,  "Pie": 9
}

_latest_rank = len(_OS_CODE_NAMES) - 1          # normaliser for 0-1 scaling
_version_re = re.compile(r"(\d+)")             # grabs the major part of “4.4.2”

def encode_os_osVersion(static_meta: Dict[str, Any]) -> np.ndarray:
    """
    Return a vector **[modernity , update_score]** in the range [0,1]².

    modernity    – how new is the newest OS mentioned in the long *os* string  
    update_score – how up-to-date is the installed *osVersion* relative to that
                   newest OS (1 = fully up-to-date, 0 = very outdated)
    """
    # ---------- 1. modernity from the long ‘os’ description ----------------
    os_field = str(static_meta.get("os", ""))

    best_rank  = -1          # −1 ⇒ “unknown / non-Android”
    best_major = 0
    for rank, name in enumerate(_OS_CODE_NAMES):
        if name.lower() in os_field.lower():
            best_rank = max(best_rank, rank)
            best_major = max(best_major, _CODE_TO_MAJOR[name])

    # Fallback: if nothing matched, we’ll base everything on osVersion only
    modernity = (best_rank / _latest_rank) if best_rank >= 0 else 0.0

    # ---------- 2. installed major version ---------------------------------
    inst_str = str(static_meta.get("osVersion", ""))
    m = _version_re.search(inst_str)
    installed_major = int(m.group(1)) if m else 0

    # If we never identified *best_major* above, use installed_major
    if best_major == 0:
        best_major = installed_major

    # ---------- 3. update_score --------------------------------------------
    if best_major == 0:
        update_score = 0.0
    else:
        update_score = np.clip(installed_major / best_major, 0.0, 1.0)

    return np.asarray([modernity, update_score], dtype=np.float32)

# Ordered list (oldest → newest) so the index is stable
_BRANDS = [
    "Amazon", "LeEco", "YU", "Acer", "HP", "Samsung", "BQ", "Motorola",
    "Tesco", "Doogee", "iBall", "Asus", "Lenovo", "HTC", "LG", "Huawei",
    "Sony", "Silent Circle", "UMI", "Tecno", "Google", "OnePlus",
    "alcatel", "Wiko", "Nvidia", "Micromax", "ZTE", "BlackBerry"
]
_BRAND_TO_IDX = {b.lower(): i for i, b in enumerate(_BRANDS)}

def encode_brand(static_meta: Dict[str, Any]) -> np.ndarray:
    """
    One-hot encode the *brand* field.

    Returns
    -------
    np.ndarray   shape = (len(_BRANDS),)  with exactly one 1 and rest 0s.
                 If the brand is unseen/unknown the vector is all zeros.
    """
    brand_raw = str(static_meta.get("brand", "")).strip().lower()
    idx = _BRAND_TO_IDX.get(brand_raw, None)

    vec = np.zeros(len(_BRANDS), dtype=np.float32)
    if idx is not None:
        vec[idx] = 1.0
    return vec

# ------------------------------------------------------------------
# RAM  -------------------------------------------------------------
# ------------------------------------------------------------------
_MIN_GB = 0.25                # 256 MB
_MAX_GB = 4.0                 # 4 GB (largest in dataset)

# capture *both* the number and its unit so we can treat MB vs GB
_RAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(gb|mb)", re.I)

def encode_ram(static_meta: Dict[str, Any]) -> np.ndarray:
    txt = str(static_meta.get("RAM", "None")).lower()
    sizes = []

    for num, unit in _RAM_RE.findall(txt):
        val = float(num)
        if unit.lower() == "mb":
            val /= 1024.0          # convert MB → GB
        sizes.append(val)

    ram_gb = max(sizes) if sizes else _MIN_GB
    ram_gb = np.clip(ram_gb, _MIN_GB, _MAX_GB)
    ram_norm = (ram_gb - _MIN_GB) / (_MAX_GB - _MIN_GB)
    return np.asarray([ram_norm], dtype=np.float32)


# ------------------------------------------------------------------
# Internal storage  -------------------------------------------------
# ------------------------------------------------------------------
_MIN_GB_INT = 0.5              # 512 MB
_MAX_GB_INT = 128.0            # 128 GB (largest in dataset)

_INT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(gb|mb)", re.I)

def encode_internal_memory(static_meta: Dict[str, Any]) -> np.ndarray:
    txt = str(static_meta.get("internal_memory", "None")).lower()
    sizes = []

    for num, unit in _INT_RE.findall(txt):
        val = float(num)
        if unit.lower() == "mb":
            val /= 1024.0          # MB → GB
        sizes.append(val)

    mem_gb = max(sizes) if sizes else _MIN_GB_INT
    mem_gb = np.clip(mem_gb, _MIN_GB_INT, _MAX_GB_INT)
    mem_norm = (mem_gb - _MIN_GB_INT) / (_MAX_GB_INT - _MIN_GB_INT)
    return np.asarray([mem_norm], dtype=np.float32)

# --- constants -------------------------------------------------------------
_MIN_MAH = 1300.0     # smallest realistic phone/tablet cell in list
_MAX_MAH = 9600.0     # largest value seen
_VOLTAGE = 3.7        # typical Li-Ion / Li-Po nominal voltage

# patterns
_MAH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mAh", re.I)
_WH_RE  = re.compile(r"(\d+(?:\.\d+)?)\s*Wh",  re.I)

def encode_battery(static_meta: Dict[str, Any]) -> np.ndarray:
    """
    Parse the *battery* field and return one normalised feature in [0,1].

    • Picks the **largest** capacity mentioned (handles multiple options).  
    • Accepts “… mAh” directly, or “… Wh” which it converts using 3.7 V.  
    • If neither unit is present, uses `_MIN_MAH` as a safe default.
    """
    txt = str(static_meta.get("battery", "")).lower()

    # --- 1. direct mAh values --------------------------------------------
    mah_vals = [float(m.group(1)) for m in _MAH_RE.finditer(txt)]

    # --- 2. Wh → mAh conversion ------------------------------------------
    for m in _WH_RE.finditer(txt):
        wh = float(m.group(1))
        mah_vals.append((wh * 1000.0) / _VOLTAGE)

    # fallback if no numbers found
    capacity = max(mah_vals) if mah_vals else _MIN_MAH

    # clip & normalise
    capacity = np.clip(capacity, _MIN_MAH, _MAX_MAH)
    cap_norm = (capacity - _MIN_MAH) / (_MAX_MAH - _MIN_MAH)

    return np.asarray([cap_norm], dtype=np.float32)


_CPU_MIN, _CPU_MAX = 4_000_000_000, 67_200_000_000
_GPU_MIN, _GPU_MAX = 0, 360_000_000_000
_THR_MIN, _THR_MAX = 72, 433

def encode_int_features(static_meta: Dict[str, Any]) -> np.ndarray:
    """Normalise **cpu_flops, gpu_flops, peak_throughput** to [0,1]^3."""
    cpu  = float(static_meta.get("cpu_flops", _CPU_MIN))
    gpu  = float(static_meta.get("gpu_flops", _GPU_MIN))
    thr  = float(static_meta.get("peak_throughput", _THR_MIN))

    cpu_n = (np.clip(cpu, _CPU_MIN, _CPU_MAX) - _CPU_MIN) / (_CPU_MAX - _CPU_MIN)
    gpu_n = (np.clip(gpu, _GPU_MIN, _GPU_MAX) - _GPU_MIN) / (_GPU_MAX - _GPU_MIN)
    thr_n = (np.clip(thr, _THR_MIN, _THR_MAX) - _THR_MIN) / (_THR_MAX - _THR_MIN)

    return np.asarray([cpu_n, gpu_n, thr_n], dtype=np.float32)

_cluster_file = Path(__file__).resolve().parent / "clusters.json"
_model_to_cluster: Dict[str, int] = {}
_cluster_ids: Dict[int, int] = {}   # original id → 0-based index

try:
    with _cluster_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # make cluster-id indices compact 0..K-1
    for idx, obj in enumerate(data):
        cid = int(obj["id"])
        _cluster_ids[cid] = idx
        for model in obj["models"]:
            _model_to_cluster[model.strip().lower()] = idx

    _N_CLUSTERS = len(_cluster_ids)
except FileNotFoundError:
    _model_to_cluster = {}
    _N_CLUSTERS = 0
    print(f"[Bliss] Warning: cluster.json not found at {_cluster_file}")

# ------------------------------------------------------------------
# Model-cluster one-hot encoder
# ------------------------------------------------------------------


def encode_model_cluster(static_meta: Dict[str, Any]) -> np.ndarray:
    """
    One-hot encode the *model* string based on cluster.json.

    Returns
    -------
    np.ndarray  shape = (_N_CLUSTERS,)  • all zeros if model not in map.
    """
    vec = np.zeros(_N_CLUSTERS, dtype=np.float32)

    if _N_CLUSTERS == 0:
        return vec  # graceful degrade when file missing

    model_raw = str(static_meta.get("model", "")).strip().lower()
    idx = _model_to_cluster.get(model_raw, None)
    if idx is not None:
        vec[idx] = 1.0
    return vec



def encode_static_metadata(static_meta: Dict[str, Any]) -> np.ndarray:
    """
    Aggregate all static-feature encodings into a single 1-D vector.

    Order of concatenation  (→ feature index ranges stay stable):
    1. [2 ]  encode_os_osVersion
    2. [28]  encode_brand
    3. [27 ]  encode_model_cluster   (27 = #clusters in clusters.json)
    4. [1 ]  encode_ram
    5. [1 ]  encode_internal_memory
    6. [1 ]  encode_battery
    7. [3 ]  encode_int_features    (cpu_flops, gpu_flops, throughput)
    ----------------------------------------------------------------
    N_total = 2 + 28 + 27 + 1 + 1 + 1 + 3 = 63
    """

    vecs = [
        encode_os_osVersion(static_meta),          # (2,)
        encode_brand(static_meta),                 # (28,)
        encode_model_cluster(static_meta),         # (27,)
        encode_ram(static_meta),                   # (1,)
        encode_internal_memory(static_meta),       # (1,)
        encode_battery(static_meta),               # (1,)
        encode_int_features(static_meta)           # (3,)
    ]
    return np.concatenate(vecs).astype(np.float32)
