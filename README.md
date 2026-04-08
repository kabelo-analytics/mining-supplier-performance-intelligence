# Mining Supplier Performance & Procurement Efficiency Intelligence

This project is a mining operations case study focused on supply reliability, procurement friction, and downstream operational risk.

The analysis asks a simple operating question:

Where are supplier and procurement issues creating avoidable delay, stock pressure, downtime, or cost impact, and what should operations teams tighten first?

## Why this case matters

Mining sites do not feel procurement problems evenly.

A late diesel order, a weak fill rate on grinding media, or repeat delays on critical spares can turn into emergency buying, stock pressure, maintenance disruption, and direct operating loss. This project uses realistic synthetic data to show how a procurement and supply team could monitor those patterns before they become routine fire-fighting.

## Mining operations context

The case is set up around six months of synthetic procurement activity across multiple sites. The operating environment includes:

- critical consumables and spares
- supplier tiers with different cost and reliability profiles
- urgent orders that move faster but cost more
- site-level inventory pressure
- incidents linked to delays, partial deliveries, and stockouts

The data is synthetic, but the operating logic is intentionally imperfect. Strategic suppliers tend to perform better, but not perfectly. Some lower-cost suppliers create more instability. Delays cluster in more difficult categories. Sites do not behave the same way.

## Workflow

1. Generate a frozen synthetic snapshot using `random_state=42` and a fixed analysis end date.
2. Build purchase-order, delivery, inventory, incident, and site-level views.
3. Score supplier and site risk using delivery reliability, fulfilment quality, stockout pressure, and incident exposure.
4. Export summary tables, a notebook, figures, and a written insight pack.

## KPI tree

North star:

- Operational Supply Reliability

Supporting KPIs:

- on-time delivery rate
- procurement cycle days
- quantity fulfilment rate
- stockout rate
- incident rate
- downtime hours
- cost impact
- supplier concentration risk

## Dataset overview

Files generated in `data/processed`:

- `suppliers.csv`
- `purchase_orders.csv`
- `deliveries.csv`
- `inventory_status.csv`
- `incidents.csv`
- `site_summary.csv`

Exported analytical outputs in `outputs`:

- `supplier_performance_summary.csv`
- `procurement_cycle_summary.csv`
- `stockout_risk_summary.csv`
- `incident_impact_summary.csv`
- `site_risk_summary.csv`
- `insights.md`

## Workflow logic by table

- `suppliers.csv`: supplier master data with category, tier, regional footprint, lead time, cost index, and baseline reliability
- `purchase_orders.csv`: order timing, expected vs actual delivery, urgency, order values, and procurement cycle days
- `deliveries.csv`: on-time flag, fulfilment quality, and operational issue flags
- `inventory_status.csv`: weekly stock position, stockout exposure, and emergency replenishment pressure
- `incidents.csv`: downtime and cost impact linked to supply disruption
- `site_summary.csv`: site-level rollup of procurement value, supplier mix, delay rate, stockout rate, and incident rate

## Current snapshot

The latest frozen run in this repo shows:

- on-time delivery rate: `58.21%`
- highest-risk supplier category: `Electrical Spares`
- strongest stockout driver: `Grinding Media`
- biggest improvement opportunity: supplier intervention on critical categories and tighter site-level stock control

These values come from the generated outputs in this repo and should remain stable across reruns unless the generation logic changes.

## Key insights

1. Delivery reliability is under pressure without being uniformly broken. That matters because repeated slippage creates operational drag long before it becomes a full crisis.
2. Electrical Spares stands out as the highest-risk supplier category in the current run, which is exactly the kind of category that can disrupt maintenance and plant continuity.
3. Urgent orders are helping sites recover faster, but the pattern also points to planning weakness and avoidable cost escalation.
4. Stockout risk is clustering in a small set of category-site combinations rather than being evenly spread.
5. Incident cost makes supplier underperformance easier to see in operational terms than purchase-order reporting alone.
6. Site risk is uneven, which means intervention should start where pressure is concentrated rather than with a generic group-wide response.

## Business recommendations

- Move the highest-risk suppliers in critical categories into a structured performance-review cadence tied to delivery, fill rate, and incident exposure.
- Track urgent-order share by site and material category so repeat emergency buying becomes a planning signal.
- Tighten reorder discipline and escalation triggers for the highest-risk category-site combinations first.
- Use downtime-linked incident cost in supplier governance so procurement decisions reflect operating consequence, not just purchase price.
- Prioritise site recovery plans for the highest-risk sites before rolling out broader sourcing changes.

## Tools

- Python
- pandas
- numpy
- matplotlib
- Jupyter Notebook

## Reproduce locally

From the project root:

```bash
python src/generate_and_analyze.py
```

This writes the processed tables, summary outputs, notebook, figures, and `docs/index.html`.

## Notes on realism and limitations

- This is a synthetic case study designed to show an operations analytics workflow.
- The project demonstrates a decision-support approach, not a live mining system.
- Risk labels and summary outputs are descriptive and triage-oriented. They are meant to guide operational attention, not replace supplier audits or production planning.
