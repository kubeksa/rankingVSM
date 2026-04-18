
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ranking VSM", page_icon="🏐", layout="wide")

POSITION_ALIASES = {
    "przyjmująca": "OH", "przyjmujaca": "OH", "przyjmujący": "OH", "przyjmujacy": "OH",
    "środkowa": "MB", "srodkowa": "MB", "środkowy": "MB", "srodkowy": "MB",
    "atakująca": "OPP", "atakujaca": "OPP", "atakujący": "OPP", "atakujacy": "OPP",
    "libero": "L", "rozgrywająca": "S", "rozgrywajaca": "S", "rozgrywający": "S", "rozgrywajacy": "S",
    "oh": "OH", "mb": "MB", "opp": "OPP", "op": "OPP", "l": "L", "lib": "L", "s": "S",
}
POSITION_CODE_MAP = {0: "L", 1: "OH", 2: "OPP", 3: "MB", 4: "S"}
POSITION_LABELS = {"OH": "Przyjmujące", "MB": "Środkowe", "OPP": "Atakujące", "L": "Libero", "S": "Rozgrywające"}

POSITION_PROFILES = {
    "OH": {
        "minimums": {"reception_total": 1, "attack_total": 1},
        "group_weights": {"receiving": 0.40, "attacking": 0.35, "scoring": 0.10, "balance": 0.15},
        "metric_weights": {
            "receiving": {"reception_positive_pct": 0.60, "reception_perfect_pct": 0.40},
            "attacking": {"attack_efficiency_pct": 0.60, "attack_success_pct": 0.40},
            "scoring": {"block_points_per_set": 0.50, "serve_aces_per_set": 0.50},
            "balance": {"point_balance_per_set": 1.00},
        },
        "raw_sort": ["attack_efficiency_pct", "attack_success_pct", "reception_positive_pct", "reception_perfect_pct", "point_balance", "block_points", "serve_aces"],
    },
    "MB": {
        "minimums": {"attack_total": 1},
        "group_weights": {"attacking": 0.35, "blocking": 0.35, "serve": 0.10, "balance": 0.20},
        "metric_weights": {
            "attacking": {"attack_efficiency_pct": 0.60, "attack_success_pct": 0.40},
            "blocking": {"block_points_per_set": 0.65, "block_touches_per_set": 0.35},
            "serve": {"serve_aces_per_set": 1.00},
            "balance": {"point_balance_per_set": 1.00},
        },
        "raw_sort": ["attack_efficiency_pct", "attack_success_pct", "block_points", "block_touches", "serve_aces", "point_balance"],
    },
    "OPP": {
        "minimums": {"attack_total": 1},
        "group_weights": {"attacking": 0.50, "balance": 0.25, "blocking": 0.10, "serve": 0.15},
        "metric_weights": {
            "attacking": {"attack_efficiency_pct": 0.65, "attack_success_pct": 0.35},
            "balance": {"point_balance_per_set": 1.00},
            "blocking": {"block_points_per_set": 1.00},
            "serve": {"serve_aces_per_set": 1.00},
        },
        "raw_sort": ["attack_efficiency_pct", "attack_success_pct", "point_balance", "block_points", "serve_aces"],
    },
    "L": {
        "minimums": {"reception_total": 1},
        "group_weights": {"reception": 0.55, "defense": 0.30, "workload": 0.15},
        "metric_weights": {
            "reception": {"reception_positive_pct": 0.60, "reception_perfect_pct": 0.40},
            "defense": {"dig_count_per_set": 1.00},
            "workload": {"reception_total_per_set": 1.00},
        },
        "raw_sort": ["reception_positive_pct", "reception_perfect_pct", "reception_total", "dig_count"],
    },
    "S": {
        "minimums": {"set_to_attack_count": 1},
        "group_weights": {"attack_after_set": 0.75, "block": 0.15, "serve": 0.10},
        "metric_weights": {
            "attack_after_set": {"attack_after_set_pct": 1.00},
            "block": {"block_points_per_set": 1.00},
            "serve": {"serve_aces_per_set": 1.00},
        },
        "raw_sort": ["attack_after_set_pct", "set_to_kill_count", "block_points", "serve_aces"],
    },
}

OUTPUT_COLUMNS = {
    "OH": ["rank","player_name","team_code","matches","sets_played","reception_positive_count","reception_positive_pct","reception_perfect_count","reception_perfect_pct","attack_kills","attack_success_pct","attack_efficiency_count","attack_efficiency_pct","block_points","serve_aces","point_balance","ranking_score"],
    "MB": ["rank","player_name","team_code","matches","sets_played","attack_kills","attack_success_pct","attack_efficiency_count","attack_efficiency_pct","block_points","block_touches","serve_aces","point_balance","ranking_score"],
    "OPP": ["rank","player_name","team_code","matches","sets_played","attack_kills","attack_success_pct","attack_efficiency_count","attack_efficiency_pct","block_points","serve_aces","point_balance","ranking_score"],
    "L": ["rank","player_name","team_code","matches","sets_played","reception_positive_count","reception_positive_pct","reception_perfect_count","reception_perfect_pct","reception_total","dig_count","ranking_score"],
    "S": ["rank","player_name","team_code","matches","sets_played","set_to_attack_count","set_to_kill_count","attack_after_set_pct","block_points","serve_aces","ranking_score"],
}

DISPLAY_COLUMN_NAMES = {
    "rank": "Miejsce", "player_name": "Zawodniczka", "team_code": "Drużyna", "matches": "Mecze", "sets_played": "Sety",
    "reception_positive_count": "Przyjęcie + (liczba)", "reception_positive_pct": "Przyjęcie + (%)",
    "reception_perfect_count": "Przyjęcie # (liczba)", "reception_perfect_pct": "Przyjęcie # (%)",
    "attack_kills": "A#", "attack_success_pct": "Skuteczność ataku (%)",
    "attack_efficiency_count": "Efektywność ataku (liczba)", "attack_efficiency_pct": "Efektywność ataku (%)",
    "block_points": "B#", "block_touches": "B+", "serve_aces": "S#", "point_balance": "Bilans punktowy",
    "ranking_score": "Weighted score", "reception_total": "Liczba przyjęć", "dig_count": "D#",
    "set_to_attack_count": "Ataki po wystawie", "set_to_kill_count": "A# po wystawie",
    "attack_after_set_pct": "Skuteczność po wystawie (%)", "raw_rank": "RAW", "weighted_rank": "WEIGHTED",
    "delta_rank": "Zmiana vs RAW",
}

def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0

def normalize_position(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, int):
        return POSITION_CODE_MAP.get(value)
    text = str(value).strip().lower()
    return POSITION_ALIASES.get(text, str(value).upper())

def first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None

def get_nested(d: Any, *path, default=None):
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur

def percent_rank(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    if series.empty:
        return series
    if series.nunique(dropna=False) <= 1:
        return pd.Series([50.0] * len(series), index=series.index)
    return (series.rank(method="average", pct=True) * 100.0).round(4)

def count_eval(df: pd.DataFrame, code: str) -> int:
    return int((df["evaluation_code"] == code).sum())

def count_skill(df: pd.DataFrame, skill: str) -> int:
    return int((df["skill"] == skill).sum())

def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out

def normalize_player_number(raw_player: Any) -> Any:
    if raw_player is None:
        return None
    text = str(raw_player).strip()
    return int(text) if text.isdigit() else text

def pretty_label(col: str) -> str:
    return DISPLAY_COLUMN_NAMES.get(col, col)

def make_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_pct"):
            out[col] = (out[col].fillna(0) * 100).round(2)
        elif col == "ranking_score":
            out[col] = out[col].fillna(0).round(2)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].fillna(0).round(2)
    return out

def rename_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: pretty_label(c) for c in df.columns})

def extract_team_and_player_metadata(vsm: dict) -> Tuple[Dict[Tuple[str, Any], dict], Dict[str, str]]:
    player_lookup: Dict[Tuple[str, Any], dict] = {}
    side_to_team_code: Dict[str, str] = {}
    team_root = vsm.get("team", {}) or {}
    for side_name in ("home", "away"):
        side = team_root.get(side_name, {}) or {}
        team_code = str(first_not_none(side.get("code"), side.get("name"), side_name))
        side_to_team_code[side_name] = team_code
        for p in side.get("players", []) or []:
            shirt_number = normalize_player_number(first_not_none(p.get("shirtNumber"), p.get("number"), p.get("playerNumber"), p.get("jerseyNumber")))
            first_name = first_not_none(p.get("firstName"), "")
            last_name = first_not_none(p.get("lastName"), "")
            full_name = f"{first_name} {last_name}".strip()
            normalized_position = POSITION_CODE_MAP.get(p.get("position"))
            player_lookup[(team_code, shirt_number)] = {
                "player_name": full_name if full_name else None,
                "player_number": shirt_number,
                "position": normalized_position,
                "team_code": team_code,
            }
    return player_lookup, side_to_team_code

def parse_vsm_to_dataframe(path: str | Path, positions_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    positions_map = positions_map or {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        vsm = json.load(f)
    if not isinstance(vsm, dict) or not isinstance(vsm.get("scout"), dict):
        return pd.DataFrame()

    match_id = path.stem
    player_lookup, side_to_team_code = extract_team_and_player_metadata(vsm)
    rows: List[dict] = []
    sets = get_nested(vsm, "scout", "sets", default=[])
    if not isinstance(sets, list):
        return pd.DataFrame()

    def map_team_code(raw_team: Any) -> Optional[str]:
        if raw_team is None:
            return None
        raw_team = str(raw_team)
        if raw_team in ("home", "away"):
            return side_to_team_code.get(raw_team, raw_team)
        if raw_team == "a":
            return side_to_team_code.get("away", "away")
        if raw_team == "*":
            return side_to_team_code.get("home", "home")
        return raw_team

    for set_idx, set_obj in enumerate(sets, start=1):
        if not isinstance(set_obj, dict):
            continue
        set_number = first_not_none(set_obj.get("setNumber"), set_obj.get("number"), set_obj.get("set"), set_idx)
        events = set_obj.get("events", [])
        if not isinstance(events, list):
            continue
        for event_idx, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                continue
            exchange = event.get("exchange")
            if not isinstance(exchange, dict):
                continue
            plays = exchange.get("plays")
            if not isinstance(plays, list):
                continue
            for play_idx, play in enumerate(plays, start=1):
                if not isinstance(play, dict):
                    continue
                skill = play.get("skill")
                effect = play.get("effect")
                if not skill:
                    continue
                team_code = map_team_code(first_not_none(play.get("team"), play.get("teamCode"), play.get("teamName")))
                player_number = normalize_player_number(first_not_none(play.get("playerNumber"), play.get("number"), play.get("shirtNumber"), play.get("player"), get_nested(play, "player", "number")))
                player_name = first_not_none(play.get("playerName"), play.get("name"), get_nested(play, "player", "name"))
                meta = None
                if team_code is not None and player_number is not None:
                    meta = player_lookup.get((str(team_code), player_number))
                inferred_position = None
                if meta:
                    player_name = first_not_none(player_name, meta.get("player_name"))
                    team_code = first_not_none(team_code, meta.get("team_code"))
                    inferred_position = meta.get("position")
                position = first_not_none(normalize_position(play.get("position")), inferred_position, positions_map.get(player_name) if player_name else None)
                rows.append({
                    "match_id": match_id, "set_number": set_number, "exchange_id": event_idx, "play_index": play_idx,
                    "team_code": str(team_code) if team_code is not None else None, "player_name": player_name,
                    "player_number": player_number, "position": normalize_position(position), "skill": str(skill),
                    "effect": str(effect) if effect is not None else "", "evaluation_code": f"{skill}{effect}" if effect is not None else str(skill),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return ensure_columns(df, ["match_id","set_number","exchange_id","play_index","team_code","player_name","player_number","position","skill","effect","evaluation_code"])

def load_many_vsm_files(paths: List[str | Path], positions_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = parse_vsm_to_dataframe(path, positions_map=positions_map)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def compute_common_metrics(df_player: pd.DataFrame) -> dict:
    all_attacks = count_skill(df_player, "A")
    all_receptions = count_skill(df_player, "R")
    all_sets = count_skill(df_player, "E")
    attack_kills = count_eval(df_player, "A#")
    attack_slash = count_eval(df_player, "A/")
    attack_errors = count_eval(df_player, "A=")
    reception_positive_count = count_eval(df_player, "R#") + count_eval(df_player, "R+")
    reception_perfect_count = count_eval(df_player, "R#")
    block_points = count_eval(df_player, "B#")
    block_touches = count_eval(df_player, "B+")
    serve_aces = count_eval(df_player, "S#")
    dig_count = count_eval(df_player, "D#")
    points = count_eval(df_player, "S#") + count_eval(df_player, "A#") + count_eval(df_player, "B#")
    errors = count_eval(df_player, "F=") + count_eval(df_player, "S=") + count_eval(df_player, "A=") + count_eval(df_player, "B/") + count_eval(df_player, "E=")
    point_balance = points - errors
    matches = df_player["match_id"].nunique()
    sets_played = df_player["set_number"].nunique()
    return {
        "matches": int(matches), "sets_played": int(sets_played), "attack_total": all_attacks, "attack_kills": attack_kills,
        "attack_success_pct": safe_div(attack_kills, all_attacks),
        "attack_efficiency_count": attack_kills - attack_slash - attack_errors,
        "attack_efficiency_pct": safe_div(attack_kills - attack_slash - attack_errors, all_attacks),
        "reception_total": all_receptions, "reception_positive_count": reception_positive_count,
        "reception_positive_pct": safe_div(reception_positive_count, all_receptions),
        "reception_perfect_count": reception_perfect_count,
        "reception_perfect_pct": safe_div(reception_perfect_count, all_receptions),
        "set_total": all_sets, "block_points": block_points, "block_touches": block_touches, "serve_aces": serve_aces,
        "dig_count": dig_count, "points_total": points, "errors_total": errors, "point_balance": point_balance,
        "block_points_per_set": safe_div(block_points, sets_played), "block_touches_per_set": safe_div(block_touches, sets_played),
        "serve_aces_per_set": safe_div(serve_aces, sets_played), "point_balance_per_set": safe_div(point_balance, sets_played),
        "dig_count_per_set": safe_div(dig_count, sets_played), "reception_total_per_set": safe_div(all_receptions, sets_played),
    }

def compute_setter_followup_metrics(df_all: pd.DataFrame, player_name: str, team_code: Optional[str]) -> dict:
    subset = df_all.copy()
    if team_code is not None:
        subset = subset[subset["team_code"] == team_code]
    subset = subset.sort_values(["match_id", "set_number", "exchange_id", "play_index"]).reset_index(drop=True)
    set_to_attack_count = 0
    set_to_kill_count = 0
    for _, ex in subset.groupby(["match_id", "set_number", "exchange_id"], sort=False):
        ex = ex.sort_values("play_index").reset_index(drop=True)
        for i in range(len(ex) - 1):
            current = ex.iloc[i]
            nxt = ex.iloc[i + 1]
            if current["player_name"] == player_name and current["skill"] == "E" and current["team_code"] == nxt["team_code"] and nxt["skill"] == "A":
                set_to_attack_count += 1
                if nxt["effect"] == "#":
                    set_to_kill_count += 1
    return {
        "set_to_attack_count": set_to_attack_count,
        "set_to_kill_count": set_to_kill_count,
        "attack_after_set_pct": safe_div(set_to_kill_count, set_to_attack_count),
    }

def infer_position_from_group(df_player: pd.DataFrame, fallback: Optional[str] = None) -> Optional[str]:
    non_null = df_player["position"].dropna()
    if not non_null.empty:
        return normalize_position(non_null.mode().iloc[0])
    return normalize_position(fallback)

def compute_player_stats(df: pd.DataFrame, positions_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    positions_map = positions_map or {}
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (player_name, team_code), g in df.groupby(["player_name", "team_code"], dropna=False):
        if player_name is None:
            continue
        base = compute_common_metrics(g)
        position = infer_position_from_group(g, fallback=positions_map.get(player_name))
        non_null_num = g["player_number"].dropna()
        player_number = non_null_num.iloc[0] if not non_null_num.empty else None
        row = {"player_name": player_name, "player_number": player_number, "team_code": team_code, "position": position, **base}
        if position == "S":
            row.update(compute_setter_followup_metrics(df, player_name, team_code))
        else:
            row.update({"set_to_attack_count": 0, "set_to_kill_count": 0, "attack_after_set_pct": 0.0})
        rows.append(row)
    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats["position"] = stats["position"].apply(normalize_position)
    return stats

def apply_minimums(df_pos: pd.DataFrame, minimums: Dict[str, Any]) -> pd.DataFrame:
    out = df_pos.copy()
    for col, min_value in minimums.items():
        if col in out.columns:
            out = out[out[col] >= min_value]
    return out

def build_raw_ranking(stats_df: pd.DataFrame, position: str) -> pd.DataFrame:
    position = normalize_position(position)
    profile = POSITION_PROFILES[position]
    df = stats_df[stats_df["position"] == position].copy()
    df = apply_minimums(df, profile["minimums"])
    sort_cols = [c for c in profile["raw_sort"] if c in df.columns]
    if not sort_cols:
        return df
    df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

def build_weighted_ranking(stats_df: pd.DataFrame, position: str) -> pd.DataFrame:
    position = normalize_position(position)
    profile = POSITION_PROFILES[position]
    df = stats_df[stats_df["position"] == position].copy()
    df = apply_minimums(df, profile["minimums"])
    if df.empty:
        return df
    metric_names = set()
    for _, metrics in profile["metric_weights"].items():
        metric_names.update(metrics.keys())
    for metric in metric_names:
        if metric not in df.columns:
            df[metric] = 0.0
        df[f"{metric}_score"] = percent_rank(df[metric].fillna(0.0))
    for group_name, metrics in profile["metric_weights"].items():
        group_score = pd.Series([0.0] * len(df), index=df.index)
        for metric, weight in metrics.items():
            group_score += df[f"{metric}_score"] * weight
        df[f"{group_name}_score"] = group_score
    final_score = pd.Series([0.0] * len(df), index=df.index)
    for group_name, group_weight in profile["group_weights"].items():
        final_score += df[f"{group_name}_score"] * group_weight
    df["ranking_score"] = final_score.round(4)
    df = df.sort_values(by=["ranking_score"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df

def select_output_columns(df: pd.DataFrame, position: str) -> pd.DataFrame:
    cols = [c for c in OUTPUT_COLUMNS[position] if c in df.columns]
    return df[cols].copy()

def get_rank_map(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or "player_name" not in df.columns or "rank" not in df.columns:
        return {}
    return {str(row["player_name"]): int(row["rank"]) for _, row in df[["player_name", "rank"]].dropna().iterrows()}

def build_rank_comparison_df(raw_df: pd.DataFrame, weighted_df: pd.DataFrame) -> pd.DataFrame:
    raw_map = get_rank_map(raw_df)
    weighted_map = get_rank_map(weighted_df)
    rows = []
    for player in sorted(set(raw_map.keys()) | set(weighted_map.keys())):
        raw_rank = raw_map.get(player)
        weighted_rank = weighted_map.get(player)
        if raw_rank is None or weighted_rank is None:
            continue
        rows.append({"player_name": player, "raw_rank": raw_rank, "weighted_rank": weighted_rank, "delta_rank": weighted_rank - raw_rank})
    return pd.DataFrame(rows)

def build_all_rankings(stats_df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    result = {}
    for pos in POSITION_PROFILES.keys():
        raw_full = build_raw_ranking(stats_df, pos)
        weighted_full = build_weighted_ranking(stats_df, pos)
        result[pos] = {
            "raw_full": raw_full,
            "weighted_full": weighted_full,
            "raw": select_output_columns(raw_full, pos),
            "weighted": select_output_columns(weighted_full, pos),
            "compare": build_rank_comparison_df(raw_full, weighted_full),
        }
    return result

@st.cache_data(show_spinner=False)
def process_uploaded_files(file_payloads: List[Tuple[str, bytes]], manual_positions_text: str):
    positions_map: Dict[str, str] = {}
    text = manual_positions_text.strip()
    if text:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            name, pos = line.split("=", 1)
            name = name.strip()
            pos = normalize_position(pos.strip())
            if name and pos in POSITION_PROFILES:
                positions_map[name] = pos

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths: List[Path] = []
        for name, content in file_payloads:
            path = Path(tmpdir) / name
            path.write_bytes(content)
            tmp_paths.append(path)

        plays_df = load_many_vsm_files(tmp_paths, positions_map=positions_map)
        if plays_df.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        stats_df = compute_player_stats(plays_df, positions_map=positions_map)
        rankings = build_all_rankings(stats_df)
        return plays_df, stats_df, rankings

def build_excel_bytes(stats_df: pd.DataFrame, rankings: Dict[str, Dict[str, pd.DataFrame]]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for pos in ["OH", "MB", "OPP", "L", "S"]:
            raw_df = rankings.get(pos, {}).get("raw", pd.DataFrame()).copy()
            weighted_df = rankings.get(pos, {}).get("weighted", pd.DataFrame()).copy()
            compare_df = rankings.get(pos, {}).get("compare", pd.DataFrame()).copy()
            (raw_df if not raw_df.empty else pd.DataFrame({"info": [f"Brak danych dla {pos} raw"]})).to_excel(writer, sheet_name=f"{pos}_raw", index=False)
            (weighted_df if not weighted_df.empty else pd.DataFrame({"info": [f"Brak danych dla {pos} weighted"]})).to_excel(writer, sheet_name=f"{pos}_weighted", index=False)
            (compare_df if not compare_df.empty else pd.DataFrame({"info": [f"Brak danych dla {pos} compare"]})).to_excel(writer, sheet_name=f"{pos}_compare", index=False)
        if not stats_df.empty:
            stats_df.to_excel(writer, sheet_name="all_stats", index=False)
    output.seek(0)
    return output.getvalue()

def render_main_table(df: pd.DataFrame):
    st.dataframe(rename_display_columns(make_display_dataframe(df)), use_container_width=True, hide_index=True)

def main():
    st.title("🏐 Ranking VSM")
    st.caption("Szybka wersja webowa w Streamlit")

    with st.sidebar:
        st.header("Dane wejściowe")
        uploaded_files = st.file_uploader("Wrzuć pliki .vsm", type=["vsm", "json"], accept_multiple_files=True)
        manual_positions_text = st.text_area("Awaryjne ręczne przypisanie pozycji", value="", help="Format: Imię Nazwisko = OH/MB/OPP/L/S", height=120)

    if not uploaded_files:
        st.info("Najpierw wrzuć co najmniej jeden plik .vsm.")
        return

    file_payloads = [(f.name, f.getvalue()) for f in uploaded_files]

    with st.spinner("Liczę rankingi..."):
        plays_df, stats_df, rankings = process_uploaded_files(file_payloads, manual_positions_text)

    if plays_df.empty or stats_df.empty or not rankings:
        st.error("Nie udało się wczytać żadnych zagrań z plików.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pliki", len(uploaded_files))
    c2.metric("Mecze", int(plays_df["match_id"].nunique()))
    c3.metric("Zagrania", int(len(plays_df)))
    c4.metric("Zawodniczki", int(stats_df["player_name"].nunique()))

    st.download_button(
        "Pobierz Excel",
        data=build_excel_bytes(stats_df, rankings),
        file_name="rankingi_zawodniczek.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tab_raw, tab_weighted, tab_analysis = st.tabs(["RAW", "WEIGHTED", "ANALIZA"])

    with tab_raw:
        st.subheader("Ranking RAW")
        pos = st.selectbox("Pozycja", ["OH", "MB", "OPP", "L", "S"], key="raw_pos")
        raw_df = rankings[pos]["raw"].copy()
        if raw_df.empty:
            st.warning("Brak danych dla tej pozycji.")
        else:
            render_main_table(raw_df)
            metric_candidates = [c for c in raw_df.columns if c != "rank" and pd.api.types.is_numeric_dtype(raw_df[c])]
            default_metric = "ranking_score" if "ranking_score" in metric_candidates else metric_candidates[0]
            metric = st.selectbox("Metryka wykresu", metric_candidates, index=metric_candidates.index(default_metric), key="raw_metric")
            top_n = st.slider("Liczba zawodniczek", 5, min(20, len(raw_df)), min(10, len(raw_df)), key="raw_topn")
            plot_df = raw_df.sort_values(by=metric, ascending=(metric == "rank")).head(top_n).copy()
            y_vals = plot_df[metric].fillna(0).tolist()
            if metric.endswith("_pct"):
                y_vals = [v * 100 for v in y_vals]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(plot_df["player_name"], y_vals)
            ax.set_title(f"TOP {len(plot_df)} – {pretty_label(metric)}")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

    with tab_weighted:
        st.subheader("Ranking WEIGHTED")
        pos = st.selectbox("Pozycja", ["OH", "MB", "OPP", "L", "S"], key="weighted_pos")
        weighted_df = rankings[pos]["weighted"].copy()
        if weighted_df.empty:
            st.warning("Brak danych dla tej pozycji.")
        else:
            render_main_table(weighted_df)
            top_n = st.slider("Liczba zawodniczek", 5, min(20, len(weighted_df)), min(10, len(weighted_df)), key="weighted_topn")
            plot_df = weighted_df.head(top_n).copy()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(plot_df["player_name"], plot_df["ranking_score"].fillna(0).tolist())
            ax.set_title(f"TOP {len(plot_df)} – Weighted score")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

    with tab_analysis:
        st.subheader("RAW vs WEIGHTED")
        pos = st.selectbox("Pozycja", ["OH", "MB", "OPP", "L", "S"], key="analysis_pos")
        compare_df = rankings[pos]["compare"].copy()
        if compare_df.empty:
            st.warning("Brak danych dla tej pozycji.")
        else:
            st.caption("Tabela pokazuje ranking RAW obok WEIGHTED oraz zmianę miejsca względem RAW.")
            table_df = compare_df[["player_name", "raw_rank", "weighted_rank", "delta_rank"]].copy()
            st.dataframe(rename_display_columns(make_display_dataframe(table_df)), use_container_width=True, hide_index=True)

            sort_mode = st.radio("Sortowanie zmian", ["Największe awanse", "Największe spadki", "Alfabetycznie"], horizontal=True)
            chart_df = compare_df.copy()
            if sort_mode == "Największe awanse":
                chart_df = chart_df.sort_values("delta_rank").head(15)
            elif sort_mode == "Największe spadki":
                chart_df = chart_df.sort_values("delta_rank", ascending=False).head(15)
            else:
                chart_df = chart_df.sort_values("player_name")

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(chart_df["player_name"], chart_df["delta_rank"])
            ax.axhline(0, linewidth=1)
            ax.set_title("Zmiana pozycji względem RAW")
            ax.set_ylabel("WEIGHTED - RAW")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
