from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
NOTEBOOKS_DIR = ROOT / "notebooks"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = OUTPUTS_DIR / "figures"


@dataclass(frozen=True)
class Config:
    random_state: int = 42
    analysis_end_date: str = "2026-03-31"
    supplier_count: int = 120
    po_count: int = 3900
    site_count: int = 12


CFG = Config()

SITE_TYPES = ["Open Pit", "Underground", "Processing Plant", "Smelter"]
PROVINCES = ["Limpopo", "North West", "Mpumalanga", "Northern Cape", "Free State"]
SUPPLIER_REGIONS = ["Gauteng", "North West", "Mpumalanga", "KwaZulu-Natal", "Western Cape", "Northern Cape"]
SUPPLIER_TIERS = ["strategic", "core", "transactional"]
CONTRACT_STATUS = ["active", "under_review", "probation"]
MATERIAL_CATEGORIES = [
    "Explosives",
    "Diesel",
    "Grinding Media",
    "Drill Consumables",
    "Tyres",
    "Safety PPE",
    "Reagents",
    "Electrical Spares",
    "Conveyor Parts",
    "General MRO",
]
INCIDENT_TYPES = [
    "Delayed critical spare",
    "Stockout shutdown",
    "Partial delivery disruption",
    "Quality-related rework",
    "Transport delay",
    "Documentation hold",
]
SUPPLIER_PREFIXES = [
    "Mhlabeni",
    "Oreline",
    "Rockridge",
    "Mavuno",
    "Tshipi",
    "North Crest",
    "Basalt",
    "Ironline",
    "Vantage",
    "Delta Shaft",
    "Kopano",
    "Metsi",
]
SUPPLIER_SUFFIXES = ["Supply", "Industrial", "Mining Services", "Procure", "Materials", "Resources"]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, DOCS_DIR, NOTEBOOKS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_suppliers(rng: np.random.Generator) -> pd.DataFrame:
    tier_probs = [0.2, 0.5, 0.3]
    status_probs = [0.76, 0.16, 0.08]
    categories = rng.choice(MATERIAL_CATEGORIES, size=CFG.supplier_count, replace=True)
    tiers = rng.choice(SUPPLIER_TIERS, size=CFG.supplier_count, replace=True, p=tier_probs)
    regions = rng.choice(SUPPLIER_REGIONS, size=CFG.supplier_count, replace=True)
    statuses = rng.choice(CONTRACT_STATUS, size=CFG.supplier_count, replace=True, p=status_probs)

    tier_reliability = {"strategic": 82, "core": 73, "transactional": 61}
    tier_lead = {"strategic": 12, "core": 18, "transactional": 24}
    tier_cost = {"strategic": 108, "core": 100, "transactional": 94}

    rows = []
    for i in range(CFG.supplier_count):
        tier = tiers[i]
        category = categories[i]
        reliability = tier_reliability[tier] + rng.normal(0, 8)
        if category in {"Tyres", "Electrical Spares", "Conveyor Parts"}:
            reliability -= rng.uniform(2, 7)
        if category in {"Safety PPE", "General MRO"}:
            reliability += rng.uniform(1, 4)
        lead_time = tier_lead[tier] + rng.integers(-4, 8)
        if category in {"Explosives", "Reagents", "Electrical Spares"}:
            lead_time += rng.integers(2, 8)
        cost_index = tier_cost[tier] + rng.normal(0, 7)
        if category in {"Tyres", "Explosives"}:
            cost_index += rng.uniform(3, 10)
        if category in {"General MRO", "Safety PPE"}:
            cost_index -= rng.uniform(2, 6)
        if statuses[i] == "under_review":
            reliability -= rng.uniform(5, 12)
        elif statuses[i] == "probation":
            reliability -= rng.uniform(10, 18)

        rows.append(
            {
                "supplier_id": f"S{str(i + 1).zfill(4)}",
                "supplier_name": f"{rng.choice(SUPPLIER_PREFIXES)} {rng.choice(SUPPLIER_SUFFIXES)}",
                "category": category,
                "region": regions[i],
                "supplier_tier": tier,
                "lead_time_days": int(np.clip(round(lead_time), 5, 45)),
                "contract_status": statuses[i],
                "unit_cost_index": round(float(np.clip(cost_index, 72, 135)), 2),
                "reliability_score": round(float(np.clip(reliability, 38, 96)), 2),
                "local_supplier_flag": int(regions[i] in {"Gauteng", "North West", "Mpumalanga", "Northern Cape"}),
            }
        )
    return pd.DataFrame(rows)


def build_sites(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(CFG.site_count):
        rows.append(
            {
                "site_id": f"SITE_{i + 1:02d}",
                "province": rng.choice(PROVINCES, p=[0.24, 0.22, 0.22, 0.2, 0.12]),
                "site_type": rng.choice(SITE_TYPES, p=[0.25, 0.35, 0.3, 0.1]),
            }
        )
    return pd.DataFrame(rows)


def build_purchase_orders(rng: np.random.Generator, suppliers: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    end = pd.Timestamp(CFG.analysis_end_date)
    start = end - pd.Timedelta(days=179)
    date_range = pd.date_range(start, end, freq="D")
    supplier_lookup = suppliers.set_index("supplier_id").to_dict("index")
    site_ids = sites["site_id"].tolist()
    supplier_ids = suppliers["supplier_id"].tolist()

    category_base_value = {
        "Explosives": 420000,
        "Diesel": 360000,
        "Grinding Media": 285000,
        "Drill Consumables": 160000,
        "Tyres": 310000,
        "Safety PPE": 70000,
        "Reagents": 240000,
        "Electrical Spares": 210000,
        "Conveyor Parts": 250000,
        "General MRO": 85000,
    }
    category_base_qty = {
        "Explosives": 8,
        "Diesel": 20,
        "Grinding Media": 12,
        "Drill Consumables": 55,
        "Tyres": 10,
        "Safety PPE": 220,
        "Reagents": 18,
        "Electrical Spares": 16,
        "Conveyor Parts": 9,
        "General MRO": 130,
    }

    rows = []
    for i in range(CFG.po_count):
        supplier_id = rng.choice(supplier_ids)
        s = supplier_lookup[supplier_id]
        category = s["category"]
        site_id = rng.choice(site_ids)
        order_date = pd.Timestamp(rng.choice(date_range))
        urgent = rng.random() < 0.18
        expected_lead = s["lead_time_days"] - rng.integers(1, 5) if urgent else s["lead_time_days"] + rng.integers(-2, 4)
        expected_lead = int(np.clip(expected_lead, 3, 45))
        expected_delivery = order_date + pd.Timedelta(days=expected_lead)

        delay_bias = (68 - s["reliability_score"]) / 38
        if category in {"Tyres", "Conveyor Parts", "Electrical Spares"}:
            delay_bias += 0.2
        if urgent:
            delay_bias -= 0.6
        actual_lead = expected_lead + int(round(rng.normal(delay_bias, 2.0)))
        actual_lead = int(np.clip(actual_lead, 1, 58))
        actual_delivery = order_date + pd.Timedelta(days=actual_lead)

        cycle_days = max(1, int(round(rng.normal(8.5 if urgent else 14.5, 3.8))))
        if s["supplier_tier"] == "transactional":
            cycle_days += rng.integers(1, 4)
        if category in {"Explosives", "Electrical Spares"}:
            cycle_days += rng.integers(1, 5)

        value = category_base_value[category] * (s["unit_cost_index"] / 100) * rng.uniform(0.72, 1.34)
        if urgent:
            value *= rng.uniform(1.08, 1.2)
        qty = category_base_qty[category] * rng.uniform(0.7, 1.35)

        po_status = "received"
        if actual_delivery > end:
            po_status = "in_transit"
        elif actual_delivery > expected_delivery:
            po_status = rng.choice(["received", "received", "partially_received", "delayed"])
        elif rng.random() < 0.04:
            po_status = "cancelled"

        rows.append(
            {
                "po_id": f"PO{str(i + 1).zfill(6)}",
                "supplier_id": supplier_id,
                "site_id": site_id,
                "order_date": order_date.date().isoformat(),
                "expected_delivery_date": expected_delivery.date().isoformat(),
                "actual_delivery_date": actual_delivery.date().isoformat(),
                "material_category": category,
                "order_value_zar": round(float(value), 2),
                "order_quantity": round(float(qty), 2),
                "urgent_order_flag": int(urgent),
                "po_status": po_status,
                "procurement_cycle_days": int(np.clip(cycle_days, 2, 35)),
            }
        )
    return pd.DataFrame(rows)


def build_deliveries(rng: np.random.Generator, purchase_orders: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    supplier_lookup = suppliers.set_index("supplier_id").to_dict("index")
    rows = []
    for i, po in purchase_orders.iterrows():
        s = supplier_lookup[po["supplier_id"]]
        expected = pd.Timestamp(po["expected_delivery_date"])
        actual = pd.Timestamp(po["actual_delivery_date"])
        on_time = int(actual <= expected)

        qty_fill = rng.normal(0.97, 0.05)
        if s["supplier_tier"] == "transactional":
            qty_fill -= rng.uniform(0.02, 0.08)
        if po["material_category"] in {"Electrical Spares", "Conveyor Parts", "Tyres"}:
            qty_fill -= rng.uniform(0.01, 0.06)
        if po["urgent_order_flag"] == 1:
            qty_fill -= rng.uniform(0.0, 0.03)

        quality_issue = int(rng.random() < (0.04 + max(0, (70 - s["reliability_score"]) / 180)))
        transport_issue = int(rng.random() < (0.05 + (0 if on_time else 0.1)))
        documentation_issue = int(rng.random() < (0.03 + (0.02 if s["contract_status"] != "active" else 0)))

        rows.append(
            {
                "delivery_id": f"D{str(i + 1).zfill(6)}",
                "po_id": po["po_id"],
                "supplier_id": po["supplier_id"],
                "site_id": po["site_id"],
                "delivery_date": actual.date().isoformat(),
                "on_time_flag": on_time,
                "quantity_fulfilled_pct": round(float(np.clip(qty_fill, 0.58, 1.0)), 3),
                "quality_issue_flag": quality_issue,
                "transport_issue_flag": transport_issue,
                "documentation_issue_flag": documentation_issue,
            }
        )
    return pd.DataFrame(rows)


def build_inventory(rng: np.random.Generator, deliveries: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    weeks = pd.date_range("2025-11-03", pd.Timestamp(CFG.analysis_end_date), freq="W-MON")
    delivery_dates = deliveries.copy()
    delivery_dates["week"] = pd.to_datetime(delivery_dates["delivery_date"]).dt.to_period("W").apply(lambda p: p.start_time.date().isoformat())
    delivery_summary = (
        delivery_dates.groupby(["site_id", "week"])
        .agg(on_time_flag=("on_time_flag", "mean"), quantity_fulfilled_pct=("quantity_fulfilled_pct", "mean"))
        .reset_index()
    )
    delivery_lookup = {(row.site_id, row.week): row for row in delivery_summary.itertuples(index=False)}

    rows = []
    for site in sites["site_id"]:
        site_pressure = rng.uniform(0.9, 1.2)
        for category in MATERIAL_CATEGORIES:
            opening_stock = rng.integers(110, 250)
            threshold = rng.integers(28, 62)
            for week in weeks:
                week_key = week.date().isoformat()
                d = delivery_lookup.get((site, week_key))
                on_time_mean = d.on_time_flag if d else 0.86
                fill_mean = d.quantity_fulfilled_pct if d else 0.95
                weekly_use = rng.normal(58, 15) * site_pressure
                if category in {"Diesel", "Explosives", "Grinding Media"}:
                    weekly_use *= rng.uniform(1.1, 1.35)

                replenishment = rng.normal(70, 16) * fill_mean
                if on_time_mean < 0.75:
                    replenishment *= rng.uniform(0.65, 0.9)
                if category in {"Electrical Spares", "Conveyor Parts", "Tyres"}:
                    replenishment *= rng.uniform(0.82, 1.02)

                closing_stock = opening_stock + replenishment - weekly_use
                days_below = 0
                emergency = 0
                if closing_stock < threshold:
                    gap = threshold - closing_stock
                    days_below = int(np.clip(round(gap / max(threshold, 1) * 14), 1, 7))
                    emergency = int(rng.random() < 0.45 + (0.25 if on_time_mean < 0.75 else 0))
                stockout = int(closing_stock <= 0 or (closing_stock < threshold * 0.22 and rng.random() < 0.24))
                if stockout:
                    closing_stock = max(closing_stock, 0)
                    days_below = max(days_below, rng.integers(2, 8))
                    emergency = 1

                rows.append(
                    {
                        "record_id": f"INV-{site}-{category[:3].upper()}-{week.strftime('%Y%m%d')}",
                        "site_id": site,
                        "material_category": category,
                        "week": week_key,
                        "opening_stock": round(float(opening_stock), 2),
                        "closing_stock": round(float(max(closing_stock, 0)), 2),
                        "stockout_flag": stockout,
                        "days_below_threshold": int(days_below),
                        "emergency_replenishment_flag": int(emergency),
                    }
                )
                opening_stock = max(closing_stock + rng.uniform(12, 38) * emergency, 15)
    return pd.DataFrame(rows)


def build_incidents(
    rng: np.random.Generator,
    purchase_orders: pd.DataFrame,
    deliveries: pd.DataFrame,
    inventory_status: pd.DataFrame,
) -> pd.DataFrame:
    delivery_map = deliveries.set_index("po_id").to_dict("index")
    risky_pos = purchase_orders.copy()
    risky_pos["late_days"] = (
        pd.to_datetime(risky_pos["actual_delivery_date"]) - pd.to_datetime(risky_pos["expected_delivery_date"])
    ).dt.days
    risky_pos["incident_weight"] = 0.015
    risky_pos.loc[risky_pos["late_days"] > 0, "incident_weight"] += 0.075
    risky_pos.loc[risky_pos["urgent_order_flag"] == 1, "incident_weight"] += 0.025
    risky_pos.loc[risky_pos["material_category"].isin(["Electrical Spares", "Conveyor Parts", "Tyres"]), "incident_weight"] += 0.045

    inventory_risk = inventory_status[inventory_status["stockout_flag"] == 1][["site_id", "material_category", "week"]]
    rows = []
    incident_id = 1

    for po in risky_pos.itertuples(index=False):
        if rng.random() >= min(po.incident_weight, 0.26):
            continue
        delivery = delivery_map.get(po.po_id, {})
        severity = rng.choice(["Low", "Medium", "High", "Critical"], p=[0.28, 0.38, 0.24, 0.10])
        if po.material_category in {"Electrical Spares", "Conveyor Parts"} and po.late_days > 3:
            severity = rng.choice(["Medium", "High", "Critical"], p=[0.25, 0.5, 0.25])

        if po.late_days > 3:
            incident_type = "Delayed critical spare"
        elif delivery.get("quality_issue_flag", 0) == 1:
            incident_type = "Quality-related rework"
        elif delivery.get("transport_issue_flag", 0) == 1:
            incident_type = "Transport delay"
        elif delivery.get("documentation_issue_flag", 0) == 1:
            incident_type = "Documentation hold"
        else:
            incident_type = "Partial delivery disruption"

        downtime = {"Low": rng.uniform(0.5, 3), "Medium": rng.uniform(2, 8), "High": rng.uniform(6, 18), "Critical": rng.uniform(16, 42)}[severity]
        cost = downtime * rng.uniform(25000, 85000)
        rows.append(
            {
                "incident_id": f"INC{incident_id:05d}",
                "site_id": po.site_id,
                "supplier_id": po.supplier_id,
                "incident_date": pd.Timestamp(po.actual_delivery_date).date().isoformat(),
                "incident_type": incident_type,
                "operational_impact_level": severity,
                "downtime_hours": round(float(downtime), 2),
                "cost_impact_zar": round(float(cost), 2),
            }
        )
        incident_id += 1

    for inv in inventory_risk.itertuples(index=False):
        if rng.random() < 0.18:
            severity = rng.choice(["Medium", "High", "Critical"], p=[0.42, 0.4, 0.18])
            downtime = {"Medium": rng.uniform(3, 10), "High": rng.uniform(8, 20), "Critical": rng.uniform(18, 48)}[severity]
            related_supplier = rng.choice(purchase_orders[purchase_orders["site_id"] == inv.site_id]["supplier_id"])
            rows.append(
                {
                    "incident_id": f"INC{incident_id:05d}",
                    "site_id": inv.site_id,
                    "supplier_id": related_supplier,
                    "incident_date": pd.Timestamp(inv.week).date().isoformat(),
                    "incident_type": "Stockout shutdown",
                    "operational_impact_level": severity,
                    "downtime_hours": round(float(downtime), 2),
                    "cost_impact_zar": round(float(downtime * rng.uniform(30000, 95000)), 2),
                }
            )
            incident_id += 1
    return pd.DataFrame(rows)


def build_site_summary(
    purchase_orders: pd.DataFrame,
    deliveries: pd.DataFrame,
    inventory_status: pd.DataFrame,
    incidents: pd.DataFrame,
    sites: pd.DataFrame,
) -> pd.DataFrame:
    po_summary = purchase_orders.groupby("site_id").agg(
        monthly_procurement_value=("order_value_zar", lambda s: s.sum() / 6.0),
        supplier_count=("supplier_id", "nunique"),
    )
    delay_summary = deliveries.groupby("site_id").agg(delay_rate=("on_time_flag", lambda s: 1 - s.mean()))
    stock_summary = inventory_status.groupby("site_id").agg(stockout_rate=("stockout_flag", "mean"))
    incident_summary = incidents.groupby("site_id").agg(incident_rate=("incident_id", "count"))
    out = sites.set_index("site_id").join(po_summary).join(delay_summary).join(stock_summary).join(incident_summary)
    out["incident_rate"] = out["incident_rate"].fillna(0) / 6.0
    out = out.reset_index()
    out["monthly_procurement_value"] = out["monthly_procurement_value"].round(2)
    out["delay_rate"] = out["delay_rate"].round(4)
    out["stockout_rate"] = out["stockout_rate"].round(4)
    out["incident_rate"] = out["incident_rate"].round(2)
    return out


def supplier_performance_summary(
    suppliers: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    deliveries: pd.DataFrame,
    incidents: pd.DataFrame,
) -> pd.DataFrame:
    po = purchase_orders.copy()
    po["late_days"] = (
        pd.to_datetime(po["actual_delivery_date"]) - pd.to_datetime(po["expected_delivery_date"])
    ).dt.days.clip(lower=0)
    supplier_po = po.groupby("supplier_id").agg(
        po_count=("po_id", "count"),
        urgent_order_share=("urgent_order_flag", "mean"),
        average_procurement_cycle_days=("procurement_cycle_days", "mean"),
        average_order_value_zar=("order_value_zar", "mean"),
    )
    supplier_del = deliveries.groupby("supplier_id").agg(
        on_time_delivery_rate=("on_time_flag", "mean"),
        average_fulfilment_pct=("quantity_fulfilled_pct", "mean"),
        quality_issue_rate=("quality_issue_flag", "mean"),
        transport_issue_rate=("transport_issue_flag", "mean"),
        documentation_issue_rate=("documentation_issue_flag", "mean"),
    )
    supplier_inc = incidents.groupby("supplier_id").agg(
        incident_count=("incident_id", "count"),
        downtime_hours=("downtime_hours", "sum"),
        cost_impact_zar=("cost_impact_zar", "sum"),
    )
    late = po.groupby("supplier_id").agg(
        average_late_days=("late_days", "mean"),
        delay_rate=("late_days", lambda s: (s > 0).mean()),
    )
    out = (
        suppliers.set_index("supplier_id")
        .join(supplier_po)
        .join(supplier_del)
        .join(supplier_inc)
        .join(late)
        .fillna({"incident_count": 0, "downtime_hours": 0, "cost_impact_zar": 0})
        .reset_index()
    )
    out["risk_score"] = (
        (1 - out["on_time_delivery_rate"]) * 35
        + (1 - out["average_fulfilment_pct"]) * 18
        + out["quality_issue_rate"] * 16
        + out["transport_issue_rate"] * 10
        + out["documentation_issue_rate"] * 8
        + np.minimum(out["cost_impact_zar"] / 150000, 12)
        + np.where(out["supplier_tier"] == "strategic", 8, np.where(out["supplier_tier"] == "core", 5, 2))
    ).round(2)
    return out.sort_values(["risk_score", "cost_impact_zar"], ascending=[False, False])


def procurement_cycle_summary(purchase_orders: pd.DataFrame) -> pd.DataFrame:
    po = purchase_orders.copy()
    po["delay_days"] = (
        pd.to_datetime(po["actual_delivery_date"]) - pd.to_datetime(po["expected_delivery_date"])
    ).dt.days
    return (
        po.groupby(["site_id", "material_category", "urgent_order_flag"], as_index=False)
        .agg(
            po_count=("po_id", "count"),
            avg_procurement_cycle_days=("procurement_cycle_days", "mean"),
            avg_delay_days=("delay_days", "mean"),
            avg_order_value_zar=("order_value_zar", "mean"),
        )
        .sort_values(["avg_delay_days", "avg_procurement_cycle_days"], ascending=[False, False])
    )


def stockout_risk_summary(inventory_status: pd.DataFrame, purchase_orders: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    po = purchase_orders.copy()
    po["week"] = pd.to_datetime(po["actual_delivery_date"]).dt.to_period("W").apply(lambda p: p.start_time.date().isoformat())
    merged = po.merge(deliveries[["po_id", "on_time_flag", "quantity_fulfilled_pct"]], on="po_id", how="left")
    delivery_summary = (
        merged.groupby(["site_id", "material_category", "week"], as_index=False)
        .agg(delay_rate=("on_time_flag", lambda s: 1 - s.mean()), avg_fulfilment_pct=("quantity_fulfilled_pct", "mean"))
    )
    inv = inventory_status.merge(delivery_summary, on=["site_id", "material_category", "week"], how="left")
    inv["delay_rate"] = inv["delay_rate"].fillna(0)
    inv["avg_fulfilment_pct"] = inv["avg_fulfilment_pct"].fillna(1.0)
    return (
        inv.groupby(["site_id", "material_category"], as_index=False)
        .agg(
            stockout_rate=("stockout_flag", "mean"),
            avg_days_below_threshold=("days_below_threshold", "mean"),
            emergency_replenishment_rate=("emergency_replenishment_flag", "mean"),
            linked_delay_rate=("delay_rate", "mean"),
            linked_fulfilment_pct=("avg_fulfilment_pct", "mean"),
        )
        .sort_values(["stockout_rate", "avg_days_below_threshold"], ascending=[False, False])
    )


def incident_impact_summary(incidents: pd.DataFrame) -> pd.DataFrame:
    return (
        incidents.groupby(["incident_type", "operational_impact_level"], as_index=False)
        .agg(
            incident_count=("incident_id", "count"),
            total_downtime_hours=("downtime_hours", "sum"),
            total_cost_impact_zar=("cost_impact_zar", "sum"),
            avg_downtime_hours=("downtime_hours", "mean"),
        )
        .sort_values(["total_cost_impact_zar", "incident_count"], ascending=[False, False])
    )


def site_risk_summary(site_summary: pd.DataFrame, stockout_risk: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    top_stock = (
        stockout_risk.sort_values(["stockout_rate", "avg_days_below_threshold"], ascending=[False, False])
        .groupby("site_id")
        .first()
        .reset_index()[["site_id", "material_category"]]
        .rename(columns={"material_category": "highest_stockout_category"})
    )
    incident_site = incidents.groupby("site_id", as_index=False).agg(downtime_hours=("downtime_hours", "sum"), cost_impact_zar=("cost_impact_zar", "sum"))
    out = site_summary.merge(top_stock, on="site_id", how="left").merge(incident_site, on="site_id", how="left")
    out[["downtime_hours", "cost_impact_zar"]] = out[["downtime_hours", "cost_impact_zar"]].fillna(0)
    out["site_risk_score"] = (
        out["delay_rate"] * 30 + out["stockout_rate"] * 30 + out["incident_rate"] * 6 + np.minimum(out["downtime_hours"] / 10, 20)
    ).round(2)
    return out.sort_values("site_risk_score", ascending=False)


def write_insights(
    supplier_summary: pd.DataFrame,
    procurement_summary: pd.DataFrame,
    stockout_summary: pd.DataFrame,
    incident_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> None:
    on_time = deliveries["on_time_flag"].mean() * 100
    top_supplier = supplier_summary.iloc[0]
    top_site = site_summary.iloc[0]
    top_incident = incident_summary.iloc[0]
    top_stock = stockout_summary.iloc[0]
    insights = f"""# Mining Supplier Performance & Procurement Efficiency Intelligence

## INSIGHT 1
**FINDING**  
On-time delivery sits at {on_time:.1f}% across the current run, which is workable but not comfortable for mining operations that depend on critical parts and consumables landing when planned.

**SO WHAT**  
The operating risk is not a total supplier collapse. It is repeated reliability slippage that quietly adds pressure into stock buffers, maintenance planning, and site responsiveness.

**RECOMMENDATION**  
Put supplier reviews behind on-time delivery, fulfilment quality, and incident exposure together rather than using unit cost alone.

## INSIGHT 2
**FINDING**  
The highest-risk supplier in the current run is {top_supplier['supplier_name']} ({top_supplier['category']}) with weak on-time performance and measurable downtime-linked impact.

**SO WHAT**  
This is the kind of supplier that can look manageable on a purchase-order report while still creating operational instability.

**RECOMMENDATION**  
Move this supplier into a formal intervention track: service review, root-cause analysis, and contingency sourcing for the most exposed materials.

## INSIGHT 3
**FINDING**  
Urgent orders do move faster on average, but they also sit in a more expensive part of the procurement mix.

**SO WHAT**  
Urgency is acting as a pressure valve. That helps sites recover in the short term, but it can hide planning weakness and push avoidable cost into the cycle.

**RECOMMENDATION**  
Track urgent-order share by site and category, then treat repeated urgent buying as a planning signal, not just a purchasing fact.

## INSIGHT 4
**FINDING**  
The strongest stockout pressure in the current run sits in {top_stock['material_category']} at {top_stock['site_id']}, where delays and weak fulfilment are feeding inventory risk.

**SO WHAT**  
Stockout exposure is not random. It is clustering where procurement delay and partial supply are already showing up.

**RECOMMENDATION**  
Raise control on the highest-risk category-site combinations first: tighter reorder points, earlier escalation, and more disciplined emergency replenishment review.

## INSIGHT 5
**FINDING**  
{top_incident['incident_type']} is the most expensive incident class in the current run, with the biggest combined downtime and cost impact.

**SO WHAT**  
Supplier and procurement issues are not just back-office process problems. They are showing up in downtime, production pressure, and direct financial loss.

**RECOMMENDATION**  
Use downtime-linked incident cost in supplier governance so the business can see the real operating price of unreliable supply.

## INSIGHT 6
**FINDING**  
{top_site['site_id']} carries the highest site risk score in the current output, combining delay exposure, stockout pressure, and incident load.

**SO WHAT**  
Site performance is uneven. A single group-wide policy response will miss where the pressure is actually concentrated.

**RECOMMENDATION**  
Prioritise a site-level recovery plan for the highest-risk locations before rolling out broader procurement changes.
"""
    (OUTPUTS_DIR / "insights.md").write_text(insights, encoding="utf-8")


def build_notebook() -> None:
    notebook_json = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Mining Supplier Performance & Procurement Efficiency Intelligence\n", "\n", "This notebook reviews supplier reliability, procurement efficiency, stockout exposure, incident impact, and site-level operational pressure from the frozen synthetic dataset.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "ROOT = Path('..').resolve()\n",
                "DATA = ROOT / 'data' / 'processed'\n",
                "OUTPUTS = ROOT / 'outputs'\n",
                "suppliers = pd.read_csv(DATA / 'suppliers.csv')\n",
                "purchase_orders = pd.read_csv(DATA / 'purchase_orders.csv', parse_dates=['order_date','expected_delivery_date','actual_delivery_date'])\n",
                "deliveries = pd.read_csv(DATA / 'deliveries.csv', parse_dates=['delivery_date'])\n",
                "inventory_status = pd.read_csv(DATA / 'inventory_status.csv', parse_dates=['week'])\n",
                "incidents = pd.read_csv(DATA / 'incidents.csv', parse_dates=['incident_date'])\n",
                "site_summary = pd.read_csv(DATA / 'site_summary.csv')\n",
                "supplier_summary = pd.read_csv(OUTPUTS / 'supplier_performance_summary.csv')\n",
                "procurement_summary = pd.read_csv(OUTPUTS / 'procurement_cycle_summary.csv')\n",
                "stockout_summary = pd.read_csv(OUTPUTS / 'stockout_risk_summary.csv')\n",
                "incident_summary = pd.read_csv(OUTPUTS / 'incident_impact_summary.csv')\n",
                "site_risk_summary = pd.read_csv(OUTPUTS / 'site_risk_summary.csv')\n",
            ]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Setup\nLoad the frozen processed data and summary outputs.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["suppliers.head()\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Data overview\nReview the structure and scale of each table.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["{name: df.shape for name, df in {'suppliers': suppliers, 'purchase_orders': purchase_orders, 'deliveries': deliveries, 'inventory_status': inventory_status, 'incidents': incidents, 'site_summary': site_summary}.items()}\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Supplier performance\nFocus on on-time delivery, fulfilment quality, and downtime-linked impact.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["supplier_summary[['supplier_id','supplier_name','category','supplier_tier','on_time_delivery_rate','average_fulfilment_pct','risk_score']].head(10)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Procurement efficiency\nLook at cycle time and delay patterns by site, category, and urgency.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["procurement_summary.head(12)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Inventory risk\nCheck which site-category combinations carry the strongest stockout pressure.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["stockout_summary.head(12)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Incident impact\nReview downtime and cost by incident type.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["incident_summary.head(12)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Site analysis\nCombine delay, stockout, and incident pressure at site level.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["site_risk_summary.head(12)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Insights\nRead the narrative summary exported in `outputs/insights.md` alongside the tables above.\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS_DIR / "mining_supplier_performance_intelligence.ipynb").write_text(json.dumps(notebook_json, indent=2), encoding="utf-8")


def build_docs_html(site_risk_summary: pd.DataFrame, supplier_summary: pd.DataFrame, stockout_summary: pd.DataFrame, incident_summary: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    on_time = deliveries["on_time_flag"].mean() * 100
    top_supplier = supplier_summary.iloc[0]
    top_site = site_risk_summary.iloc[0]
    top_stock = stockout_summary.iloc[0]
    top_incident = incident_summary.iloc[0]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mining Supplier Performance & Procurement Efficiency Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{ --bg:#101513; --panel:#171d1a; --panel2:#0f1412; --text:#f5f4ef; --muted:#c7c1b8; --soft:#8f968e; --line:#2a312d; --accent:#9dc08b; --serif:'DM Serif Display', Georgia, serif; --sans:'Instrument Sans', system-ui, sans-serif; --mono:'DM Mono', monospace; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font-family:var(--sans); line-height:1.7; }}
    .page {{ max-width:980px; margin:0 auto; padding:2rem 1.5rem 4rem; }} .eyebrow {{ font-family:var(--mono); letter-spacing:.12em; text-transform:uppercase; color:var(--accent); font-size:.75rem; }}
    h1,h2 {{ font-family:var(--serif); font-weight:400; letter-spacing:-.02em; }} h1 {{ font-size:3rem; line-height:1.05; margin:.6rem 0 1rem; }} h2 {{ font-size:2rem; margin:2.5rem 0 1rem; }}
    p, li {{ color:var(--muted); }} .grid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); margin:2rem 0; }} .card {{ background:var(--panel); border:1px solid var(--line); padding:1.25rem; border-radius:16px; }}
    .label {{ font-family:var(--mono); text-transform:uppercase; font-size:.72rem; color:var(--soft); }} .value {{ font-size:2rem; margin:.4rem 0; color:var(--text); }} .section {{ padding-top:1rem; border-top:1px solid var(--line); margin-top:2rem; }}
    .rec {{ padding:1rem 1.1rem; background:var(--panel2); border-left:3px solid var(--accent); margin:1rem 0; }} @media (max-width:700px) {{ h1 {{ font-size:2.25rem; }} .page {{ padding:1.25rem 1rem 3rem; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="eyebrow">Mining Operations Analytics</div>
    <h1>Supplier Performance & Procurement Efficiency Intelligence</h1>
    <p>This case study turns synthetic mining procurement data into a practical view of supplier reliability, inventory pressure, and operational risk. The focus is weekly decision support, not dashboard theatre.</p>
    <div class="grid">
      <div class="card"><div class="label">On-time delivery</div><div class="value">{on_time:.1f}%</div><p>Current delivery reliability across the frozen run.</p></div>
      <div class="card"><div class="label">Highest-risk supplier</div><div class="value">{top_supplier['supplier_id']}</div><p>{top_supplier['category']} supplier with the highest combined operational risk score.</p></div>
      <div class="card"><div class="label">Top stockout pressure</div><div class="value">{top_stock['material_category']}</div><p>Highest current stockout exposure in the summary outputs.</p></div>
      <div class="card"><div class="label">Highest-risk site</div><div class="value">{top_site['site_id']}</div><p>Site with the strongest combined delay, stockout, and incident pressure.</p></div>
    </div>
    <div class="section"><h2>Problem</h2><p>Mining sites can absorb supplier underperformance for a while before it becomes visible in procurement reporting. Delays, partial deliveries, stockouts, and supplier-related incidents often appear as separate issues when they are really part of the same operating problem.</p></div>
    <div class="section"><h2>Context</h2><p>The dataset models suppliers, purchase orders, deliveries, inventory status, incidents, and site summaries. The goal is to show how procurement and supplier data can be turned into operational supply reliability intelligence rather than a basic vendor scorecard.</p></div>
    <div class="section"><h2>Dataset</h2><p>The run uses a fixed snapshot date and random state so the outputs stay reproducible. Strategic suppliers generally perform better, urgent orders move faster but cost more, and stockout pressure links back to late or weakly fulfilled deliveries without becoming perfectly correlated.</p></div>
    <div class="section"><h2>Insights</h2><div class="rec"><strong>Supplier risk:</strong> {top_supplier['supplier_name']} sits at the top of the current risk ranking. The issue is not price alone. It is the combination of delay, fulfilment quality, and operational exposure.</div><div class="rec"><strong>Inventory pressure:</strong> {top_stock['site_id']} shows the strongest stockout pressure in {top_stock['material_category']}, which points to a category-site control problem rather than a broad inventory issue.</div><div class="rec"><strong>Incident impact:</strong> {top_incident['incident_type']} carries the largest combined downtime and cost impact in the current run, which makes supply reliability an operating KPI, not only a procurement KPI.</div></div>
    <div class="section"><h2>Visuals</h2><p>The visuals stay intentionally simple and operational.</p><p><a href="../outputs/figures/supplier-risk-by-category.png">Supplier risk by category</a><br><a href="../outputs/figures/urgent-vs-standard-cycle.png">Urgent vs standard cycle time</a><br><a href="../outputs/figures/site-risk-overview.png">Site risk overview</a></p></div>
    <div class="section"><h2>Recommendations</h2><ul><li>Escalate intervention on the highest-risk suppliers in critical categories before re-running sourcing decisions on cost alone.</li><li>Track urgent-order share by site and category as a planning pressure signal rather than a normal operating state.</li><li>Strengthen category-site controls where stockouts and delay exposure repeatedly overlap.</li><li>Bring downtime and incident cost into supplier governance so reliability is judged on operating impact, not just PO performance.</li></ul></div>
    <div class="section"><h2>Tools</h2><p>Python, pandas, Jupyter, CSV exports, and GitHub-based portfolio publishing.</p></div>
    <div class="section"><h2>Navigation</h2><p><a href="../outputs/insights.md" style="color:#9dc08b;">Read insights markdown</a></p></div>
  </div>
</body>
</html>"""
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")


def build_figures(supplier_summary: pd.DataFrame, procurement_summary: pd.DataFrame, site_risk: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    cat_risk = supplier_summary.groupby("category", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(cat_risk["category"], cat_risk["risk_score"], color="#35524a")
    ax.invert_yaxis()
    ax.set_title("Average supplier risk score by category")
    ax.set_xlabel("Risk score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "supplier-risk-by-category.png", dpi=160)
    plt.close(fig)

    urgent = procurement_summary.groupby("urgent_order_flag", as_index=False).agg(
        avg_cycle=("avg_procurement_cycle_days", "mean"),
        avg_delay=("avg_delay_days", "mean"),
    )
    urgent["order_type"] = urgent["urgent_order_flag"].map({0: "Standard", 1: "Urgent"})
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(urgent))
    width = 0.36
    ax.bar(x - width / 2, urgent["avg_cycle"], width, label="Cycle days", color="#8aa67a")
    ax.bar(x + width / 2, urgent["avg_delay"], width, label="Delay days", color="#c07b5a")
    ax.set_xticks(x)
    ax.set_xticklabels(urgent["order_type"])
    ax.set_title("Urgent vs standard procurement timing")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "urgent-vs-standard-cycle.png", dpi=160)
    plt.close(fig)

    top_sites = site_risk.head(8).copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(top_sites["site_id"], top_sites["site_risk_score"], color="#5c7a67")
    ax.set_title("Highest-risk sites")
    ax.set_ylabel("Site risk score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "site-risk-overview.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(CFG.random_state)
    suppliers = build_suppliers(rng)
    sites = build_sites(rng)
    purchase_orders = build_purchase_orders(rng, suppliers, sites)
    deliveries = build_deliveries(rng, purchase_orders, suppliers)
    inventory_status = build_inventory(rng, deliveries, sites)
    incidents = build_incidents(rng, purchase_orders, deliveries, inventory_status)
    site_summary = build_site_summary(purchase_orders, deliveries, inventory_status, incidents, sites)

    supplier_summary = supplier_performance_summary(suppliers, purchase_orders, deliveries, incidents)
    procurement_summary = procurement_cycle_summary(purchase_orders)
    stockout_summary = stockout_risk_summary(inventory_status, purchase_orders, deliveries)
    incident_summary = incident_impact_summary(incidents)
    site_risk = site_risk_summary(site_summary, stockout_summary, incidents)

    suppliers.to_csv(PROCESSED_DIR / "suppliers.csv", index=False)
    purchase_orders.to_csv(PROCESSED_DIR / "purchase_orders.csv", index=False)
    deliveries.to_csv(PROCESSED_DIR / "deliveries.csv", index=False)
    inventory_status.to_csv(PROCESSED_DIR / "inventory_status.csv", index=False)
    incidents.to_csv(PROCESSED_DIR / "incidents.csv", index=False)
    site_summary.to_csv(PROCESSED_DIR / "site_summary.csv", index=False)

    supplier_summary.to_csv(OUTPUTS_DIR / "supplier_performance_summary.csv", index=False)
    procurement_summary.to_csv(OUTPUTS_DIR / "procurement_cycle_summary.csv", index=False)
    stockout_summary.to_csv(OUTPUTS_DIR / "stockout_risk_summary.csv", index=False)
    incident_summary.to_csv(OUTPUTS_DIR / "incident_impact_summary.csv", index=False)
    site_risk.to_csv(OUTPUTS_DIR / "site_risk_summary.csv", index=False)

    write_insights(supplier_summary, procurement_summary, stockout_summary, incident_summary, site_risk, deliveries)
    build_figures(supplier_summary, procurement_summary, site_risk)
    build_notebook()
    build_docs_html(site_risk, supplier_summary, stockout_summary, incident_summary, deliveries)

    print("Project built successfully")
    print("On-time delivery rate:", round(deliveries["on_time_flag"].mean() * 100, 2))
    print("Highest-risk supplier category:", supplier_summary.iloc[0]["category"])
    print("Strongest stockout driver:", stockout_summary.iloc[0]["material_category"])
    print("Biggest operational improvement opportunity: supplier intervention on critical categories and site-level stock control")


if __name__ == "__main__":
    main()
