import tempfile, os
import streamlit as st
from pathlib import Path
import duckdb
import urllib.request

# Cache the CONNECTION as a resource (not data)
@st.cache_resource(show_spinner=False)
def load_duckdb_from_bytes(data: bytes):
    # write to a stable temp path for this session
    tmpdir = tempfile.gettempdir()
    path = Path(tmpdir) / "uploaded_data.duckdb"
    path.write_bytes(data)
    # open a read-only connection; cached as a resource
    con = duckdb.connect(str(path), read_only=True)
    return con

@st.cache_resource(show_spinner=False)
def load_duckdb_from_url(url: str):
    with urllib.request.urlopen(url) as r:
        data = r.read()
    return load_duckdb_from_bytes(data)

def kpi_card(label, value, suffix=""):
    st.metric(label, f"{value:,.2f}{suffix}")

def pick_latest_release_asset(owner_repo: str, pattern=r"\.duckdb$"):
    """Return (tag, asset_url) for latest release that has a .duckdb asset."""
    api = f"https://api.github.com/repos/{owner_repo}/releases/latest"
    req = urllib.request.Request(api, headers={"Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(req) as r:
        release = json.load(r)
    tag = release.get("tag_name")
    assets = release.get("assets", [])
    for a in assets:
        if re.search(pattern, a["name"]):
            return tag, a["browser_download_url"]
    return tag, None

st.title("Mission • WoodMac Price & Capture Dashboard")

with st.sidebar:
    st.subheader("1) Load data.duckdb")
    src = st.radio("Source", ["Upload", "GitHub Release"], horizontal=True)
    con = None

    if src == "Upload":
        up = st.file_uploader("Upload data.duckdb", type=["duckdb"])
        if up:
            con = load_duckdb_from_bytes(up.read())
            st.success("Loaded DuckDB from upload.")
    else:
        repo = st.text_input("owner/repo", value="grayson-powermarkets/mission-energy-woodmac-pipeline")
        if st.button("Load latest release"):
            tag, url = pick_latest_release_asset(repo)
            if not url:
                st.error(f"No .duckdb asset found on latest release {tag}")
            else:
                con = load_duckdb_from_url(url)
                st.success(f"Loaded {tag}")

    st.divider()
    st.subheader("2) Optional: Solar 8760 for capture")
    solar_up = st.file_uploader("Upload solar 8760 CSV", type=["csv"])

if con is None:
    st.info("Load a .duckdb to begin.")
    st.stop()

# Inspect what we have
tables = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
has_monthly = "monthly_metrics" in tables
has_annual  = "annual_metrics" in tables
has_hourly  = "hourly_prices" in tables

if not (has_monthly and has_annual):
    st.error("Expected monthly_metrics and annual_metrics tables.")
    st.stop()

# Filters (distinct from monthly—faster)
filters = con.execute("""
    SELECT DISTINCT forecast_market_name AS market,
           forecast_zone_name AS zone,
           price_node_name AS node,
           year, scenario
    FROM monthly_metrics
""").fetchdf()

col1, col2 = st.columns([2,3])
with col1:
    markets = sorted(filters["market"].unique())
    market = st.selectbox("Market", markets)
    years = sorted(filters["year"].unique())
    year = st.selectbox("Year", years)
    scenarios = sorted(filters["scenario"].unique())
    scenario = st.selectbox("Scenario", scenarios)

with col2:
    nodes = sorted(filters.loc[filters["market"]==market, "node"].unique())
    sel_nodes = st.multiselect("Nodes", nodes, default=nodes[: min(10, len(nodes))])

# Query monthly slice
q_monthly = con.execute("""
    SELECT * FROM monthly_metrics
    WHERE forecast_market_name = ? AND year = ? AND scenario = ? AND price_node_name IN ?
""", [market, year, scenario, sel_nodes]).fetchdf()

# KPIs
c1, c2, c3, c4 = st.columns(4)
if len(q_monthly):
    kpi_card("Avg price (annual)", q_monthly["avg_price_dmwh"].mean(), " $/MWh")
    kpi_card("Avg basis (annual)", q_monthly["avg_basis_dmwh"].mean(), " $/MWh")
    kpi_card("TB2 (avg mo)", q_monthly["tb2_dmwh"].mean(), " $/MWh")
    kpi_card("TB4 (avg mo)", q_monthly["tb4_dmwh"].mean(), " $/MWh")
else:
    st.warning("No monthly rows for current selection.")

# Charts
st.subheader("Scenario comparison (annual)")
annual = con.execute("""
    SELECT forecast_market_name AS market, price_node_name AS node,
           year, scenario, avg_price_dmwh, avg_basis_dmwh, tb2_dmwh, tb4_dmwh
    FROM annual_metrics
    WHERE forecast_market_name = ? AND price_node_name IN ?
""", [market, sel_nodes]).fetchdf()

if not annual.empty:
    g = px.line(annual, x="year", y="avg_price_dmwh", color="scenario",
                facet_row=None, markers=True, hover_data=["node"])
    g.update_layout(height=400, margin=dict(l=10,r=10,b=10,t=30))
    st.plotly_chart(g, use_container_width=True)

st.subheader("Monthly heatmap — avg price")
if not q_monthly.empty:
    heat = q_monthly.groupby(["price_node_name","month"], as_index=False)["avg_price_dmwh"].mean()
    fig = px.density_heatmap(heat, x="month", y="price_node_name",
                             z="avg_price_dmwh", color_continuous_scale="Turbo")
    fig.update_layout(height=500, margin=dict(l=10,r=10,b=10,t=30))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Solar capture price -----------------
st.subheader("Solar capture price (from 8760)")
if solar_up is None:
    st.caption("Upload a CSV with columns like: datetime (or date + hour) and generation (MWh).")
else:
    try:
        solar = pd.read_csv(solar_up)
        # Normalize columns: expect either 'datetime' or 'date'+'hour', and 'gen' (MWh)
        cols = {c.lower(): c for c in solar.columns}
        if "datetime" in cols:
            solar["datetime"] = pd.to_datetime(solar[cols["datetime"]], utc=True, errors="coerce")
        else:
            # Expect 'date' + 'hour'
            if not {"date","hour"}.issubset(cols):
                raise ValueError("Need datetime OR (date + hour) columns in 8760.")
            dt = pd.to_datetime(solar[cols["date"]], utc=True, errors="coerce")
            solar["datetime"] = dt + pd.to_timedelta(solar[cols["hour"]], unit="h")

        # generation column
        gen_col = None
        for cand in ["gen", "generation", "mwh", "mwhe", "output_mwh"]:
            if cand in cols:
                gen_col = cols[cand]; break
        if gen_col is None:
            raise ValueError("Could not find a generation (MWh) column (e.g. gen, generation, mwh).")

        solar = solar.dropna(subset=["datetime"])
        solar["year"] = solar["datetime"].dt.year

        if not has_hourly:
            st.error("This artifact does not include hourly_prices. Re-run ETL with keep_hourly: true.")
        else:
            # pull hourly node + scenario + year
            hourly = con.execute("""
                SELECT price_node_name AS node, forecast_market_name AS market,
                       year, month, day, hour, scenario, price_dmwh
                FROM hourly_prices
                WHERE forecast_market_name = ? AND year = ? AND scenario = ? AND price_node_name IN ?
            """, [market, year, scenario, sel_nodes]).fetchdf()

            # Build a UTC timestamp for hourly rows
            # Assumes data is in UTC; if not, adjust here.
            hourly["datetime"] = pd.to_datetime(dict(
                year=hourly["year"], month=hourly["month"], day=hourly["day"]
            )) + pd.to_timedelta(hourly["hour"], unit="h")

            # Limit solar to the selected year
            s = solar.loc[solar["year"] == year, ["datetime", gen_col]].rename(columns={gen_col:"gen_mwh"})

            # Join & compute capture price per node
            merged = hourly.merge(s, on="datetime", how="inner")
            cap = (merged.groupby("node", as_index=False)
                          .apply(lambda g: pd.Series({
                              "capture_dmwh": (g["price_dmwh"]*g["gen_mwh"]).sum() / max(g["gen_mwh"].sum(), 1e-9),
                              "gen_mwh_total": g["gen_mwh"].sum()
                          }), include_groups=False))
            st.dataframe(cap, use_container_width=True)

            # Comparison vs annual avg price
            ann = annual[(annual["year"]==year) & (annual["scenario"]==scenario)]
            comp = cap.merge(ann[["node","avg_price_dmwh"]], on="node", how="left")
            comp["capture_premium"] = comp["capture_dmwh"] - comp["avg_price_dmwh"]
            st.caption("Capture premium = capture price - annual average price")
            st.dataframe(comp, use_container_width=True)

            figc = px.bar(comp, x="node", y="capture_dmwh", hover_data=["avg_price_dmwh","capture_premium"])
            figc.update_layout(height=400, margin=dict(l=10,r=10,b=10,t=30))
            st.plotly_chart(figc, use_container_width=True)
    except Exception as e:
        st.error(f"8760 parsing error: {e}")

# ----------------- Exports -----------------
st.subheader("Export")
colx, coly = st.columns(2)
with colx:
    if not q_monthly.empty:
        st.download_button("Download monthly slice (CSV)",
                           q_monthly.to_csv(index=False).encode("utf-8"),
                           file_name=f"monthly_{market}_{year}_{scenario}.csv",
                           mime="text/csv")
with coly:
    if not annual.empty:
        st.download_button("Download annual slice (CSV)",
                           annual.to_csv(index=False).encode("utf-8"),
                           file_name=f"annual_{market}_{scenario}.csv",
                           mime="text/csv")

st.caption("Tip: change ETL filters in configs/default.yml (markets/years/nodes) and rerun to refresh the artifact.")
