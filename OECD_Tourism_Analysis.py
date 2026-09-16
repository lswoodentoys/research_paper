#!/usr/bin/env python
"""
OECD tourism panel analysis (2021–2025)
Run:
    pip install pandas numpy scipy statsmodels linearmodels openpyxl
    python OECD_Tourism_Analysis.py --input OECD_Tour_dataset.csv --output analysis_results

The script:
- audits columns, panel keys, missingness, and plausible percentage ranges
- produces descriptive and annual summary tables
- computes pooled and within-country Pearson/Spearman correlations
- estimates pooled OLS, country FE, and two-way FE models with country-clustered SEs
- runs balanced-panel and excluding-2021 sensitivity checks when feasible
- exports CSV/Excel tables and a text report

IMPORTANT:
Check the exact column names and indicator definitions against the source metadata.
This is observational analysis; coefficients are associations, not causal effects.
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf

EXPECTED_ALIASES = {
    "country": ["country", "economy", "location", "ref_area", "country_name", "geo"],
    "year": ["year", "time", "period", "time_period"],
    "GDP_SH": ["gdp_sh", "gdp share", "tourism_gdp_share"],
    "GVA_SH": ["gva_sh", "gva share", "tourism_gva_share"],
    "EMP_SH": ["emp_sh", "employment share", "tourism_employment_share"],
}

def normalize(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def resolve_columns(df):
    normalized = {normalize(c): c for c in df.columns}
    found = {}
    for target, aliases in EXPECTED_ALIASES.items():
        for alias in aliases:
            key = normalize(alias)
            if key in normalized:
                found[target] = normalized[key]
                break
    missing = [x for x in EXPECTED_ALIASES if x not in found]
    if missing:
        raise ValueError(
            "Could not identify required columns: " + ", ".join(missing) +
            "\nAvailable columns: " + ", ".join(map(str, df.columns)) +
            "\nEdit EXPECTED_ALIASES at the top of this script to match your CSV."
        )
    return found

def safe_corr(x, y, method="pearson"):
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3 or z.iloc[:,0].nunique() < 2 or z.iloc[:,1].nunique() < 2:
        return {"n": len(z), "correlation": np.nan, "p_value": np.nan}
    if method == "pearson":
        r, p = pearsonr(z.iloc[:,0], z.iloc[:,1])
    else:
        r, p = spearmanr(z.iloc[:,0], z.iloc[:,1])
    return {"n": len(z), "correlation": r, "p_value": p}

def model_result(model, name, n_countries, n_obs):
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append({
            "model": name,
            "term": term,
            "coefficient": model.params.get(term, np.nan),
            "std_error": model.bse.get(term, np.nan),
            "p_value": model.pvalues.get(term, np.nan),
            "ci_lower": ci.loc[term, 0] if term in ci.index else np.nan,
            "ci_upper": ci.loc[term, 1] if term in ci.index else np.nan,
            "n_obs": int(n_obs),
            "n_countries": int(n_countries),
            "r_squared": getattr(model, "rsquared", np.nan),
            "r_squared_adj": getattr(model, "rsquared_adj", np.nan),
        })
    return rows

def fit_ols(data, y, x, country, year, model_name, country_fe=False, year_fe=False):
    d = data[[y, x, country, year]].dropna().copy()
    if d.empty:
        return [], None
    formula = f"{y} ~ {x}"
    if country_fe:
        formula += f" + C({country})"
    if year_fe:
        formula += f" + C({year})"
    model = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d[country]}
    )
    return model_result(model, model_name, d[country].nunique(), len(d)), model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="OECD_Tour_dataset.csv")
    parser.add_argument("--output", default="analysis_results")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input)
    cols = resolve_columns(df)
    country, year = cols["country"], cols["year"]
    gdp, gva, emp = cols["GDP_SH"], cols["GVA_SH"], cols["EMP_SH"]

    # Standardize analysis names while preserving original columns.
    data = df.rename(columns={
        country: "country_id", year: "year_id",
        gdp: "GDP_SH", gva: "GVA_SH", emp: "EMP_SH"
    }).copy()
    data["country_id"] = data["country_id"].astype(str).str.strip()
    data["year_id"] = pd.to_numeric(data["year_id"], errors="coerce")
    for c in ["GDP_SH", "GVA_SH", "EMP_SH"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    report = []
    report.append(f"Input file: {args.input}")
    report.append(f"Rows: {len(data)}; columns: {len(data.columns)}")
    report.append(f"Countries/economies: {data['country_id'].nunique(dropna=True)}")
    report.append(f"Years observed: {sorted(data['year_id'].dropna().unique().tolist())}")
    report.append("")

    # Duplicate country-year audit
    duplicates = data[data.duplicated(["country_id", "year_id"], keep=False)].sort_values(
        ["country_id", "year_id"]
    )
    duplicates.to_csv(os.path.join(args.output, "duplicate_country_years.csv"), index=False)
    report.append(f"Duplicate country-year rows: {len(duplicates)}")

    # Missingness
    missing = data[["GDP_SH", "GVA_SH", "EMP_SH"]].isna().sum().rename("missing_n").to_frame()
    missing["missing_pct"] = 100 * missing["missing_n"] / len(data)
    missing.to_csv(os.path.join(args.output, "missingness.csv"))

    # Range checks: shares should generally be 0–100 if stored as percentage points.
    range_rows = []
    for v in ["GDP_SH", "GVA_SH", "EMP_SH"]:
        vals = data[v].dropna()
        range_rows.append({
            "variable": v, "min": vals.min() if len(vals) else np.nan,
            "max": vals.max() if len(vals) else np.nan,
            "below_0_n": int((vals < 0).sum()),
            "above_100_n": int((vals > 100).sum()),
            "valid_n": len(vals)
        })
    pd.DataFrame(range_rows).to_csv(os.path.join(args.output, "range_checks.csv"), index=False)

    # Descriptive statistics
    desc = data[["GDP_SH", "GVA_SH", "EMP_SH"]].describe(
        percentiles=[.25, .5, .75]
    ).T.rename(columns={"25%":"q1", "50%":"median", "75%":"q3"})
    desc.to_csv(os.path.join(args.output, "descriptive_statistics.csv"))

    # Annual summary
    annual = data.groupby("year_id")[["GDP_SH", "GVA_SH", "EMP_SH"]].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    annual.columns = ["_".join(map(str, c)) for c in annual.columns]
    annual.reset_index().to_csv(os.path.join(args.output, "annual_summary.csv"), index=False)

    # Correlations: pooled and country-demeaned (within)
    corr_rows = []
    for x in ["GDP_SH", "GVA_SH"]:
        for method in ["pearson", "spearman"]:
            r = safe_corr(data[x], data["EMP_SH"], method)
            corr_rows.append({"scope":"pooled", "x":x, "y":"EMP_SH", "method":method, **r})
            within = data[[ "country_id", x, "EMP_SH" ]].dropna().copy()
            within[x+"_within"] = within[x] - within.groupby("country_id")[x].transform("mean")
            within["EMP_within"] = within["EMP_SH"] - within.groupby("country_id")["EMP_SH"].transform("mean")
            r2 = safe_corr(within[x+"_within"], within["EMP_within"], method)
            corr_rows.append({"scope":"within_country_demeaned", "x":x, "y":"EMP_SH", "method":method, **r2})
    pd.DataFrame(corr_rows).to_csv(os.path.join(args.output, "correlations.csv"), index=False)

    # Regression models
    all_rows = []
    models = {}
    for x, label in [("GDP_SH", "GDP"), ("GVA_SH", "GVA")]:
        for cfe, yfe, name in [
            (False, False, f"{label}_pooled_OLS"),
            (True, False, f"{label}_country_FE"),
            (True, True, f"{label}_two_way_FE"),
        ]:
            rows, model = fit_ols(data, "EMP_SH", x, "country_id", "year_id",
                                  name, country_fe=cfe, year_fe=yfe)
            all_rows.extend(rows)
            models[name] = model

    # Sensitivity: exclude 2021
    no_2021 = data[data["year_id"] != 2021].copy()
    for x, label in [("GDP_SH", "GDP"), ("GVA_SH", "GVA")]:
        rows, _ = fit_ols(no_2021, "EMP_SH", x, "country_id", "year_id",
                          f"{label}_two_way_FE_excluding_2021",
                          country_fe=True, year_fe=True)
        all_rows.extend(rows)

    # Balanced panel subset: only countries with all years 2021–2025 and complete variables.
    expected_years = {2021, 2022, 2023, 2024, 2025}
    complete = data.dropna(subset=["GDP_SH", "GVA_SH", "EMP_SH", "year_id"])
    year_sets = complete.groupby("country_id")["year_id"].apply(lambda s: set(s.astype(int)))
    balanced_ids = year_sets[year_sets.apply(lambda s: expected_years.issubset(s))].index
    balanced = complete[complete["country_id"].isin(balanced_ids)].copy()
    report.append(f"Countries with complete observations for all three indicators in 2021–2025: {len(balanced_ids)}")
    for x, label in [("GDP_SH", "GDP"), ("GVA_SH", "GVA")]:
        rows, _ = fit_ols(balanced, "EMP_SH", x, "country_id", "year_id",
                          f"{label}_two_way_FE_balanced_panel",
                          country_fe=True, year_fe=True)
        all_rows.extend(rows)

    reg = pd.DataFrame(all_rows)
    reg.to_csv(os.path.join(args.output, "regression_results.csv"), index=False)

    # Plain-text audit report
    report.append(f"Rows with duplicate country-year keys are saved to duplicate_country_years.csv.")
    report.append("Check duplicate keys before interpreting panel regression results.")
    report.append("All regression coefficients are associational, not causal.")
    with open(os.path.join(args.output, "audit_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # Consolidated Excel workbook
    xlsx_path = os.path.join(args.output, "analysis_tables.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        desc.to_excel(writer, sheet_name="Descriptive")
        annual.reset_index().to_excel(writer, sheet_name="Annual", index=False)
        pd.DataFrame(corr_rows).to_excel(writer, sheet_name="Correlations", index=False)
        reg.to_excel(writer, sheet_name="Regression", index=False)
        missing.to_excel(writer, sheet_name="Missingness")
        pd.DataFrame(range_rows).to_excel(writer, sheet_name="Range checks", index=False)

    print("Analysis complete.")
    print(f"Results folder: {os.path.abspath(args.output)}")
    print("Review audit_report.txt and duplicate_country_years.csv before interpreting results.")

if __name__ == "__main__":
    main()
