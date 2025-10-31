# app.py
import io
import json
import re
from pathlib import Path
import tempfile
import urllib.request

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------- #
# Page config & helpers
# --------------------------- #
st.set_page_config(
    page_title="Mission • WoodMac Price & Capture Dashboard",
    layout="wide",
)

st.title("Mission • WoodMac Price & Capture Dashboard")

def _badge(text, color="#0e1117"):
    st.markdown(
        f"""<span style="display:inline-block;padding:.2rem .5rem;border-radius:.5rem;
            background:{color};font-size:.8rem;border:1px solid #213042">{text}</span>""",
        unsafe_allow_html=True,
    )

# --------------------------- #
# Cached loaders (resources)
# --------------------------- #
@st.cache_resource(show_spinner=False)
def load_duckdb_from_bytes(data: bytes):
    """Write uploaded duckdb bytes to a temp file and return a read-only connection."""
    tmpdir = Path(tempfile.gettempdir())
    path = tmpdir / "uploaded_data.duckdb"
    path.write_bytes(data)
    con = duckdb.connect(str(path), read_only=True)
    return con

@st.cache_resource(show_spinner=False)
def load_duckdb_from_url(url: str):
    with urllib.request.urlopen(url) as r:
        data = r.read()
    return load_duckdb_from_bytes(data)

# --------------------------- #
# Sidebar: data inputs
# --------------------------- #
st.sidebar.header("1) Load data.duckdb")
source = st.sidebar.radio("Source", ["Upload", "GitHub Release"], horizontal=True)

con = None
if source == "Upload":
    up = st.sidebar.file_uploader("Upload data.duckdb", type=["duckdb"])
    if up is not None:
        con = load_duckdb_from_bytes(up.read())
        st.sidebar.success("Loaded DuckDB from upload.")
else:
    url = st.sidebar.text_input("Paste public URL to a .duckdb (optional)")
    if url:
        with st.spinner("Loading from URL…"):
            con = load_duckdb_from_url(url)
            st.sidebar.success("Loaded DuckDB from URL.")

# Optional solar 8760 for capture
st.sidebar.header("2) Optional: Solar 8760 for capture")
solar_up = st.sidebar.file_uploader("Drag and drop file here", type=["csv"], key="solar8760")

# If no DB yet, stop early
if con is None:
    st.info("Load a .duckdb to begin.")
    st.stop()

# ---- Guard: make sure required tables exist ----
def table_exists(_con, name: str) -> bool:
    try:
        return name in _con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    except Exception:
        return False

required = ["annual_metrics", "monthly_metrics"]
missing = [t for t in required if not table_exists(con, t)]
if missing:
    st.error(f"This database is missing required tables: {', '.join(missing)}. Re-run the ETL to produce them.")
    st.stop()

# --------------------------- #
# Introspect lists from DB (dynamic)
# --------------------------- #
# ---- Fingerprint so cache invalidates when you upload a new DB ----
def db_fingerprint(_con) -> str:
    # Prefer run_info if present; else rowcount of annual_metrics
    try:
        f = _con.execute("SELECT COALESCE(MAX(run_date), '0') AS f FROM run_info").fetchdf()["f"][0]
        return str(f)
    except Exception:
        n = _con.execute("SELECT COUNT(*) AS n FROM annual_metrics").fetchdf()["n"][0]
        return f"rows:{n}"

@st.cache_data(show_spinner=False)
def get_lists_cached(_con, fp: str):
    mkts = _con.execute(
        "SELECT DISTINCT forecast_market_name FROM annual_metrics ORDER BY 1"
    ).fetchdf()["forecast_market_name"].tolist()
    yrs  = _con.execute(
        "SELECT DISTINCT year FROM annual_metrics ORDER BY 1"
    ).fetchdf()["year"].tolist()
    scs  = _con.execute(
        "SELECT DISTINCT scenario FROM annual_metrics ORDER BY 1"
    ).fetchdf()["scenario"].tolist()
    nds  = _con.execute(
        "SELECT DISTINCT price_node_name FROM annual_metrics ORDER BY 1"
    ).fetchdf()["price_node_name"].tolist()
    return mkts, yrs, scs, nds

fp = db_fingerprint(con)
markets_all, years_all, scenarios_all, nodes_all = get_lists_cached(con, fp)

# --------------------------- #
# Single, consolidated filters (market-scoped)
# --------------------------- #

# Helpers to scope lists by market
def list_years_for_market(_con, mkt):
    return _con.execute(
        "SELECT DISTINCT year FROM annual_metrics WHERE forecast_market_name = ? ORDER BY 1",
        [mkt],
    ).fetchdf()["year"].tolist()

def list_scenarios_for_market(_con, mkt):
    return _con.execute(
        "SELECT DISTINCT scenario FROM annual_metrics WHERE forecast_market_name = ? ORDER BY 1",
        [mkt],
    ).fetchdf()["scenario"].tolist()

def list_nodes_for_market(_con, mkt):
    return _con.execute(
        "SELECT DISTINCT price_node_name FROM annual_metrics WHERE forecast_market_name = ? ORDER BY 1",
        [mkt],
    ).fetchdf()["price_node_name"].tolist()

filters = st.container()
colA, colB = filters.columns([1, 1.2])

# 1) Market (global list)
with colA:
    market = st.selectbox("Market", markets_all, index=0 if markets_all else None)

# 2) Market-scoped lists
years     = list_years_for_market(con, market) if market else []
scenarios = list_scenarios_for_market(con, market) if market else []
nodes     = list_nodes_for_market(con, market) if market else []

with colA:
    year = st.selectbox("Year", years, index=0 if years else None)

with colB:
    scenario = st.selectbox("Scenario", scenarios, index=0 if scenarios else None)
    sel_nodes = st.multiselect(
        "Nodes",
        options=nodes,
        default=nodes[:min(10, len(nodes))] if nodes else [],
        help="Select one or more nodes to analyze",
    )

if not sel_nodes:
    st.warning("Select at least one node to continue.")
    st.stop()
# Register selection as a temp table on *this* connection (deduped/sanitized)
nodes_df = pd.DataFrame({"node": pd.unique([str(x) for x in sel_nodes])})
try:
    con.unregister("nodes_df")
except Exception:
    pass
con.register("nodes_df", nodes_df)

# --------------------------- #
# Queries (JOIN nodes_df)
# --------------------------- #
def q_monthly_slice(_con, mkt, yr, scen):
    return _con.execute(
        """
        SELECT m.*
        FROM monthly_metrics m
        JOIN nodes_df n ON m.price_node_name = n.node
        WHERE m.forecast_market_name = ?
          AND m.year = ?
          AND m.scenario = ?
        ORDER BY m.price_node_name, m.month
        """,
        [mkt, int(yr), scen],
    ).fetchdf()

def q_annual_for_nodes(_con, mkt):
    return _con.execute(
        """
        SELECT
            a.forecast_market_name AS market,
            a.price_node_name      AS node,
            a.year,
            a.scenario,
            a.avg_price_dmwh,
            a.avg_basis_dmwh,
            a.tb2_dmwh,
            a.tb4_dmwh
        FROM annual_metrics a
        JOIN nodes_df n ON a.price_node_name = n.node
        WHERE a.forecast_market_name = ?
        ORDER BY node, year, scenario
        """,
        [mkt],
    ).fetchdf()

def q_hourly_slice(_con, mkt, yr, scen):
    try:
        return _con.execute(
            """
            SELECT h.price_node_name AS node, h.forecast_market_name AS market,
                   h.year, h.month, h.day, h.hour, h.scenario, h.price_dmwh
            FROM hourly_prices h
            JOIN nodes_df n ON h.price_node_name = n.node
            WHERE h.forecast_market_name = ?
              AND h.year = ?
              AND h.scenario = ?
            ORDER BY node, month, day, hour
            """,
            [mkt, int(yr), scen],
        ).fetchdf()
    except Exception:
        return pd.DataFrame(columns=["node","market","year","month","day","hour","scenario","price_dmwh"])

monthly = q_monthly_slice(con, market, year, scenario)
annual  = q_annual_for_nodes(con, market)
hourly  = q_hourly_slice(con, market, year, scenario)

# --------------------------- #
# KPIs for selected year/scenario
# --------------------------- #
st.subheader("Key metrics")
k = annual[(annual["year"] == year) & (annual["scenario"] == scenario)]
k = k[["avg_price_dmwh","avg_basis_dmwh","tb2_dmwh","tb4_dmwh"]].mean(numeric_only=True)

kpi = st.columns(4)
kpi[0].metric("Avg price ($/MWh)", f"{k.get('avg_price_dmwh', float('nan')):,.2f}")
kpi[1].metric("Avg basis ($/MWh)", f"{k.get('avg_basis_dmwh', float('nan')):,.2f}")
kpi[2].metric("TB2 spread ($/MWh)", f"{k.get('tb2_dmwh', float('nan')):,.2f}")
kpi[3].metric("TB4 spread ($/MWh)", f"{k.get('tb4_dmwh', float('nan')):,.2f}")

# --------------------------- #
# Annual scenario comparison chart
# --------------------------- #
st.subheader("Scenario comparison (annual)")
annual_plot = annual.copy()
if not annual_plot.empty:
    fig = px.line(
        annual_plot,
        x="year",
        y="avg_price_dmwh",
        color="scenario",
        hover_data=["node"],
        markers=True,
        title=None,
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, b=10, t=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No annual data for current selection.")

# --------------------------- #
# Monthly profile for selected year + scenario
# --------------------------- #
st.subheader(f"Monthly prices · {market} · {year} · {scenario}")
if not monthly.empty:
    m_agg = (
        monthly.groupby(["month", "scenario"], as_index=False)["avg_price_dmwh"]
        .mean(numeric_only=True)
        .rename(columns={"avg_price_dmwh": "price"})
    )
    figm = px.bar(
        m_agg,
        x="month",
        y="price",
        color="scenario",
        title=None,
    )
    figm.update_layout(height=380, margin=dict(l=10, r=10, b=10, t=10))
    st.plotly_chart(figm, use_container_width=True)
else:
    st.info("No monthly rows for the selected filters.")

# --------------------------- #
# Capture price (optional 8760)
# --------------------------- #
st.subheader("Solar capture price (optional)")
cap_col = st.container()

def _try_parse_solar(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either (month, day, hour, mwh) or hour_of_year + mwh."""
    cols = {c.lower(): c for c in df.columns}
    df2 = df.copy()
    # normalize column names to lowercase for checks
    df2.columns = [c.lower() for c in df2.columns]

    if {"month","day","hour"}.issubset(df2.columns) and ("mwh" in df2.columns or "mw" in df2.columns):
        if "mw" in df2.columns and "mwh" not in df2.columns:
            df2["mwh"] = df2["mw"]  # accept MW as proxy if hourly
        out = df2[["month","day","hour","mwh"]].copy()
        out["mwh"] = out["mwh"].astype(float).clip(lower=0)
        return out

    if {"hour_of_year"}.issubset(df2.columns) and ("mwh" in df2.columns or "mw" in df2.columns):
        if "mw" in df2.columns and "mwh" not in df2.columns:
            df2["mwh"] = df2["mw"]
        # build month/day/hour from hour_of_year approx (1..8760)
        # For a quick, portable approach, we’ll only use hour_of_year and join on it:
        out = df2[["hour_of_year","mwh"]].copy()
        out["mwh"] = out["mwh"].astype(float).clip(lower=0)
        return out

    raise ValueError(
        "Solar CSV must include either (month, day, hour, mwh) OR (hour_of_year, mwh)."
    )

if solar_up is None:
    cap_col.info("Upload a Solar 8760 CSV to compute capture price.")
else:
    try:
        solar_raw = pd.read_csv(solar_up)
        solar = _try_parse_solar(solar_raw)

        if hourly.empty:
            cap_col.warning("Hourly price table is unavailable in this DB → capture price skipped.")
        else:
            # Build hour_of_year for price frame to support both join modes
            hp = hourly[["node","year","month","day","hour","price_dmwh"]].copy()
            # simple HOY index (non-leap, 30-day months not exact; but we only need relative weighting)
            # Use DuckDB to compute a deterministic HOY if desired; here we approximate
            # safer: compute via cumulative within-month/day ordering
            # We'll map month/day/hour to a synthetic HOY using duckdb quickly:
            idx = con.execute("""
                WITH src AS (
                  SELECT month, day, hour,
                         ROW_NUMBER() OVER (ORDER BY month, day, hour) AS hoy
                  FROM (SELECT DISTINCT month, day, hour FROM hourly_prices WHERE year = ?)
                )
                SELECT month, day, hour, hoy
                FROM src
                ORDER BY month, day, hour
            """, [int(year)]).fetchdf()
            hp = hp.merge(idx, on=["month","day","hour"], how="left")

            # Normalize weights and compute capture
            if {"month","day","hour"}.issubset(solar.columns):
                s = solar.rename(columns=str.lower).copy()
                # get HOY mapping as well
                s = s.merge(idx, on=["month","day","hour"], how="left")
            else:
                # hour_of_year mode already
                s = solar.rename(columns=str.lower).copy()
                s = s.rename(columns={"hour_of_year": "hoy"})

            s = s.dropna(subset=["mwh", "hoy"])
            s["mwh"] = s["mwh"].astype(float)
            s = s.groupby("hoy", as_index=False)["mwh"].sum()

            mix = hp.merge(s, on="hoy", how="inner")
            if mix.empty:
                cap_col.warning("No overlap between hourly prices and solar 8760 rows.")
            else:
                w = mix["mwh"].clip(lower=0)
                cp = (mix["price_dmwh"] * w).sum() / (w.sum() if w.sum() else 1.0)
                simple_avg = hp["price_dmwh"].mean()

                c1, c2, c3 = cap_col.columns(3)
                c1.metric("Capture price ($/MWh)", f"{cp:,.2f}")
                c2.metric("Simple hourly avg ($/MWh)", f"{simple_avg:,.2f}")
                c3.metric("Capture Δ ($/MWh)", f"{(cp - simple_avg):+.2f}")

                # Hour-of-year curve (averaged across nodes)
                curve = mix.groupby("hoy", as_index=False)["price_dmwh"].mean()
                figc = px.line(curve, x="hoy", y="price_dmwh", title=None)
                figc.update_layout(height=320, margin=dict(l=10, r=10, b=10, t=10))
                cap_col.plotly_chart(figc, use_container_width=True)

    except Exception as e:
        cap_col.error(f"Capture calc could not run: {e}")

# --------------------------- #
# Run info footer
# --------------------------- #
st.caption("Tip: all filter lists are pulled live from the uploaded DuckDB. Upload a new DB to refresh.")
