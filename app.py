
import streamlit as st
import pandas as pd
from optimization import optimize
from visualization import plot_network, summary

# ────────────────── page config ─────────────────────────────
st.set_page_config(
    page_title="Warehouse Optimizer – Scenarios",
    page_icon="🏭",
    layout="wide",
)

st.title("Warehouse Optimizer — Scenario Workspace")
st.caption(
    "Create multiple scenarios, including optional two‑echelon networks (up to 3 RDC/SDC), "
    "then run the solver and export results."
)

# ────────────────── bootstrap session state ─────────────────
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}  # name ➞ dict

# ────────────────── helper to draw inputs in sidebar ────────
def render_inputs(name: str, scenario: dict):
    """
    Render all user‑controlled inputs for a given scenario
    into Streamlit’s sidebar.  We mutate the supplied `scenario`
    dict in‑place so the caller can immediately use updated values.
    """
    with st.sidebar:
        st.header(f"Inputs — {name}")

        # ── file upload ───────────────────────────────────────
        up = st.file_uploader("Store demand CSV", key=f"up_{name}")
        if up:
            scenario["upload"] = up
        if "upload" in scenario and st.checkbox(
            "Show preview", key=f"prev_{name}"
        ):
            st.dataframe(pd.read_csv(scenario["upload"]).head())

        # convenience helper for numeric inputs ----------------
        def n(key, label, default, fmt="%.10f", **k):
            scenario.setdefault(key, default)
            scenario[key] = st.number_input(
                label,
                value=scenario[key],
                format=fmt,
                key=f"{name}_{key}",
                **k,
            )

        # ── cost parameters ───────────────────────────────────
        n("rate_out_min", "Outbound $/lb‑min", 0.02)
        n("fixed_cost", "Fixed cost $/warehouse", 250000.0, "%.2f", step=50000.0)
        n("sqft_per_lb", "Sq ft per lb", 0.02)
        n("cost_sqft", "Variable $/sq ft / yr", 6.0, "%.2f")

        # ── real drive times ─────────────────────────────────
        scenario.setdefault("drive_times", False)
        scenario["drive_times"] = st.checkbox(
            "Use real drive times (OpenRouteService)", value=scenario["drive_times"], key=f"dt_{name}"
        )
        if scenario["drive_times"]:
            scenario.setdefault("ors_key", "")
            scenario["ors_key"] = st.text_input(
                "OpenRouteService API key", value=scenario["ors_key"], key=f"ors_{name}", type="password"
            )

        # ── number of warehouses (k) ──────────────────────────
        scenario.setdefault("auto_k", True)
        scenario["auto_k"] = st.checkbox(
            "Optimize # warehouses", value=scenario["auto_k"], key=f"auto_{name}"
        )
        if scenario["auto_k"]:
            scenario.setdefault("k_rng", (2, 5))
            scenario["k_rng"] = st.slider(
                "k range", 1, 10, scenario["k_rng"], key=f"k_rng_{name}"
            )
            k_vals_ui = range(int(scenario["k_rng"][0]), int(scenario["k_rng"][1]) + 1)
        else:
            n(
                "k_fixed",
                "# warehouses",
                3,
                "%.0f",
                step=1,
                min_value=1,
                max_value=10,
            )
            k_vals_ui = [int(scenario["k_fixed"])]

        # ── fixed warehouses ─────────────────────────────────
        st.subheader("Fixed Warehouses (up to 10)")
        scenario.setdefault("fixed", [[0.0, 0.0, False] for _ in range(10)])
        for i in range(10):
            with st.expander(f"Fixed Warehouse {i+1}", expanded=False):
                lat = st.number_input(
                    "Latitude",
                    value=scenario["fixed"][i][1],
                    key=f"{name}_fw_lat{i}",
                    format="%.6f",
                )
                lon = st.number_input(
                    "Longitude",
                    value=scenario["fixed"][i][0],
                    key=f"{name}_fw_lon{i}",
                    format="%.6f",
                )
                use = st.checkbox(
                    "Use this location",
                    value=scenario["fixed"][i][2],
                    key=f"{name}_fw_use{i}",
                )
                scenario["fixed"][i] = [lon, lat, use]
        fixed_centers = [[lon, lat] for lon, lat, use in scenario["fixed"] if use]

        # ── inbound supply points ─────────────────────────────
        scenario.setdefault("inbound_on", False)
        scenario["inbound_on"] = st.checkbox(
            "Factor inbound flow", value=scenario["inbound_on"], key=f"in_on_{name}"
        )
        inbound_rate = 0.0
        inbound_pts = []
        if scenario["inbound_on"]:
            n("in_rate", "Inbound $/lb‑min", 0.01)
            inbound_rate = scenario["in_rate"]
            scenario.setdefault(
                "sup", [[0.0, 0.0, 0.0, False] for _ in range(5)]
            )  # lon, lat, pct, use
            for j in range(5):
                with st.expander(f"Supply Point {j+1}", expanded=False):
                    slat = st.number_input(
                        "Latitude",
                        value=scenario["sup"][j][1],
                        key=f"{name}_sp_lat{j}",
                        format="%.6f",
                    )
                    slon = st.number_input(
                        "Longitude",
                        value=scenario["sup"][j][0],
                        key=f"{name}_sp_lon{j}",
                        format="%.6f",
                    )
                    pct = st.number_input(
                        "% inbound flow",
                        min_value=0.0,
                        max_value=100.0,
                        value=scenario["sup"][j][2],
                        key=f"{name}_sp_pct{j}",
                        format="%.2f",
                    )
                    use_sp = st.checkbox(
                        "Use this supply point",
                        value=scenario["sup"][j][3],
                        key=f"{name}_sp_use{j}",
                    )
                    scenario["sup"][j] = [slon, slat, pct, use_sp]
            inbound_pts = [
                [lon, lat, pct / 100]
                for lon, lat, pct, use in scenario["sup"]
                if use and pct > 0
            ]

        # ── RDC/SDC definitions ───────────────────────────────
        st.subheader("Redistribution / Service Distribution Centers (up to 3)")
        scenario.setdefault(
            "rdcs",
            [
                {"enabled": False, "lon": 0.0, "lat": 0.0, "type": "RDC"}
                for _ in range(3)
            ],
        )
        rdc_list = []
        for i in range(3):
            rd = scenario["rdcs"][i]
            with st.expander(f"Center {i+1}", expanded=False):
                rd["enabled"] = st.checkbox(
                    "Enable", value=rd["enabled"], key=f"{name}_rdc_enable{i}"
                )
                if rd["enabled"]:
                    rd["lat"] = st.number_input(
                        "Latitude",
                        value=rd["lat"],
                        key=f"{name}_rdc_lat{i}",
                        format="%.6f",
                    )
                    rd["lon"] = st.number_input(
                        "Longitude",
                        value=rd["lon"],
                        key=f"{name}_rdc_lon{i}",
                        format="%.6f",
                    )
                    rd["type"] = st.radio(
                        "Center type",
                        ["RDC (redistribute only)", "SDC (redistribute + serve customers)"],
                        index=0 if rd["type"] == "RDC" else 1,
                        key=f"{name}_rdc_type{i}",
                    )
                scenario["rdcs"][i] = rd
                if rd["enabled"]:
                    rdc_list.append(
                        {"coords": [rd["lon"], rd["lat"]], "is_sdc": rd["type"].startswith("SDC")}
                    )

        # ── transfer + RDC cost parameters ────────────────────
        n("trans_rate", "Transfer $/lb‑minnn (RDC ➜ WH)", 0.015)
        transfer_rate = scenario["trans_rate"]
        n(
            "rdc_sqft_per_lb",
            "RDC Sq ft per lb shipped",
            scenario.get("sqft_per_lb", 0.02),
        )
        n(
            "rdc_cost_sqft",
            "RDC variable $/sq ft / yr",
            scenario.get("cost_sqft", 6.0),
            "%.2f",
        )

        # ── RUN SOLVER ─────────────────────────────────────────
        if st.button("Run solver", key=f"run_{name}"):
            if "upload" not in scenario:
                st.warning("Upload a CSV first.")
            elif scenario["inbound_on"] and not inbound_pts:
                st.warning("Enable at least one supply point.")
            else:
                df = pd.read_csv(scenario["upload"])
                result = optimize(
                    df,
                    k_vals_ui,
                    scenario["rate_out_min"],
                    scenario["sqft_per_lb"],
                    scenario["cost_sqft"],
                    scenario["fixed_cost"],
                    consider_inbound=scenario["inbound_on"],
                    inbound_rate_min=inbound_rate,
                    inbound_pts=inbound_pts,
                    fixed_centers=fixed_centers,
                    rdc_list=rdc_list,
                    transfer_rate_min=transfer_rate,
                    rdc_sqft_per_lb=scenario.get("rdc_sqft_per_lb"),
                    rdc_cost_per_sqft=scenario.get("rdc_cost_sqft"),
                    use_drive_times=scenario.get("drive_times", False),
                    ors_api_key=scenario.get("ors_key", ""),
                )
                scenario["result"] = result
                st.success("Solver finished.")

# ────────────────── build tabs (per scenario) ─────────────────────
scenario_names = list(st.session_state["scenarios"])
tabs = scenario_names + ["➕  New scenario"]
tab_refs = st.tabs(tabs)

# ─── “new scenario” tab ───────────────────────────────────────────
with tab_refs[-1]:
    new_name = st.text_input("Scenario name")
    if st.button("Create") and new_name:
        if new_name in st.session_state["scenarios"]:
            st.warning("That name already exists.")
        else:
            st.session_state["scenarios"][new_name] = {}
            st.success("Scenario created.")
            st.experimental_rerun()

# ─── iterate existing scenarios ──────────────────────────────────
for idx, name in enumerate(scenario_names):
    scenario = st.session_state["scenarios"][name]

    # render tab content (outputs only) ---------------------------
    with tab_refs[idx]:
        st.header(f"Scenario: {name}")

        # call helper to build/handle sidebar inputs
        render_inputs(name, scenario)

        # visualise results if available -------------------------
        if "result" in scenario:
            r = scenario["result"]
            plot_network(r["assigned"], r["centers"])
            summary(
                r["assigned"],
                r["total_cost"],
                r["out_cost"],
                r["in_cost"],
                r["trans_cost"],
                r["wh_cost"],
                r["centers"],
                r["demand_per_wh"],
                scenario["sqft_per_lb"],
                rdc_enabled=len(r.get("rdc_only_idx", [])) > 0,
                rdc_idx=None,
                rdc_sqft_per_lb=scenario.get("rdc_sqft_per_lb"),
                consider_inbound=scenario["inbound_on"],
                show_transfer=(len(r.get("rdc_only_idx", [])) > 0 and r["trans_cost"] > 0),
            )

            csv = r["assigned"].to_csv(index=False).encode()
            st.download_button(
                "Download assignment CSV",
                csv,
                file_name=f"{name}_assignment.csv",
                key=f"dl_{name}",
            )
