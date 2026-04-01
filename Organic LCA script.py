import olca_ipc as ipc
import olca_schema as schema
import pandas as pd
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
file_id = random.randint(1000, 9999)
PORT          = 8080
SYSTEM_NAME   = "Organic ML"
IMPACT_METHOD = "ReCiPe 2016 Midpoint (H)"
PROCESS_NAME  = "Organic ML"
RUNS_PER_BLOCK = 200
OUTPUT_CSV = f"ML_Dataset_Organic_v2_{file_id}.csv"

# Corrected realistic ranges for Indian organic rice farming
MANURE_MIN,  MANURE_MAX  = 5000,  15000   # kg/ha (5t - 15t)
COMPOST_MIN, COMPOST_MAX = 1000,  2000    # kg/ha (1t - 2t)

MANURE_MID  = (MANURE_MIN  + MANURE_MAX)  / 2   # 10000
COMPOST_MID = (COMPOST_MIN + COMPOST_MAX) / 2   # 1500

TARGET_CATEGORIES = [
    "global warming",
    "freshwater eutrophication",
    "terrestrial acidification",
    "terrestrial ecotoxicity"
]

# ── CONNECT ───────────────────────────────────────────────────────────────────
client = ipc.Client(PORT)

system_ref  = client.find(schema.ProductSystem, SYSTEM_NAME)
method_ref  = client.find(schema.ImpactMethod,  IMPACT_METHOD)
process_ref = client.find(schema.Process,       PROCESS_NAME)

print(f"✅ System  : {system_ref.name}")
print(f"✅ Method  : {method_ref.name}")
print(f"✅ Process : {process_ref.name}")
print(f"\n📐 Ranges  — Manure: {MANURE_MIN}–{MANURE_MAX} kg/ha | Compost: {COMPOST_MIN}–{COMPOST_MAX} kg/ha")
print(f"📐 Midpoints — Manure={MANURE_MID} kg/ha | Compost={COMPOST_MID} kg/ha")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def make_redef(name, value):
    r                  = schema.ParameterRedef()
    r.name             = name
    r.value            = value
    r.context          = schema.Ref()
    r.context.id       = process_ref.id
    r.context.ref_type = schema.RefType.Process
    return r

def build_setup(manure, compost):
    setup                  = schema.CalculationSetup()
    setup.target           = schema.Ref()
    setup.target.id        = system_ref.id
    setup.target.ref_type  = schema.RefType.ProductSystem
    setup.impact_method    = schema.Ref()
    setup.impact_method.id = method_ref.id
    setup.parameters       = [
        make_redef("Manure_rate",  manure),
        make_redef("Compost_rate", compost),
    ]
    return setup

# ── AUTO-DETECT ImpactValue attrs ─────────────────────────────────────────────
print("\n🔎 Detecting ImpactValue structure...")
result = client.calculate(build_setup(MANURE_MID, COMPOST_MID))
result.wait_until_ready()
all_impacts = result.get_total_impacts()
first       = all_impacts[0]

SCORE_ATTR = "amount" if hasattr(first, "amount") else "value"
CAT_ATTR   = "name"   if hasattr(first.impact_category, "name") else "ref_id"
print(f"   Score attr    : '{SCORE_ATTR}'")
print(f"   Category attr : '{CAT_ATTR}'")

# ── MAP CATEGORY NAMES ────────────────────────────────────────────────────────
cat_map = {}
for iv in all_impacts:
    cat_name = getattr(iv.impact_category, CAT_ATTR)
    for target in TARGET_CATEGORIES:
        if target.lower() in cat_name.lower():
            cat_map[target] = cat_name
result.dispose()

print(f"   Matched       : {cat_map}")
if len(cat_map) != len(TARGET_CATEGORIES):
    unmatched = [t for t in TARGET_CATEGORIES if t not in cat_map]
    raise ValueError(f"❌ Could not match: {unmatched}")

# ── CORE CALCULATION FUNCTION ─────────────────────────────────────────────────
def run_calc(manure, compost, sample_type):
    result = client.calculate(build_setup(manure, compost))
    result.wait_until_ready()
    impacts = result.get_total_impacts()

    row = {
        "sample_type"  : sample_type,
        "Manure_rate"  : round(manure,  2),
        "Compost_rate" : round(compost, 2),
    }
    for iv in impacts:
        cat_name = getattr(iv.impact_category, CAT_ATTR)
        score    = getattr(iv, SCORE_ATTR)
        for target, exact_name in cat_map.items():
            if cat_name == exact_name:
                row[target.replace(" ", "_")] = score

    result.dispose()
    return row

# ── SAMPLING BLOCKS ───────────────────────────────────────────────────────────
total   = RUNS_PER_BLOCK * 3
records = []
failed  = 0
run_idx = 0

blocks = [
    ("OAT_Manure",  lambda: random.uniform(MANURE_MIN,  MANURE_MAX),  lambda: COMPOST_MID),
    ("OAT_Compost", lambda: MANURE_MID,                                lambda: random.uniform(COMPOST_MIN, COMPOST_MAX)),
    ("RANDOM",      lambda: random.uniform(MANURE_MIN,  MANURE_MAX),  lambda: random.uniform(COMPOST_MIN, COMPOST_MAX)),
]

print(f"\n🚀 Starting {total} runs across 3 blocks ({RUNS_PER_BLOCK} each)...\n")

for label, manure_fn, compost_fn in blocks:
    print(f"── Block: {label} ──────────────────────────────────────────")
    block_failed = 0

    for i in range(RUNS_PER_BLOCK):
        run_idx    += 1
        manure_val  = round(manure_fn(),  2)
        compost_val = round(compost_fn(), 2)

        try:
            row = run_calc(manure_val, compost_val, label)
            records.append(row)

            if (i + 1) % 50 == 0:
                print(f"   [{i+1:>3}/{RUNS_PER_BLOCK}] "
                      f"Manure={manure_val:.0f} Compost={compost_val:.0f} | "
                      f"GW={row.get('global_warming', 'N/A'):.2f} | "
                      f"FE={row.get('freshwater_eutrophication', 'N/A'):.6f} | "
                      f"TA={row.get('terrestrial_acidification', 'N/A'):.4f} | "
                      f"TE={row.get('terrestrial_ecotoxicity', 'N/A'):.2f}")

        except Exception as e:
            failed      += 1
            block_failed += 1
            print(f"   ⚠️  Run {run_idx} failed (Manure={manure_val} Compost={compost_val}): {e}")
            continue

    print(f"   ✅ Block done — {RUNS_PER_BLOCK - block_failed} successful\n")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

col_order = [
    "sample_type", "Manure_rate", "Compost_rate",
    "global_warming", "freshwater_eutrophication",
    "terrestrial_acidification", "terrestrial_ecotoxicity"
]
df = df[[c for c in col_order if c in df.columns]]

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Complete — {len(df)}/{total} runs saved to '{OUTPUT_CSV}' ({failed} failed)")
print(f"\n📊 Counts per block:\n{df['sample_type'].value_counts().to_string()}")
print(f"\n📊 Preview:\n{df.head(6).to_string()}")
print(f"\n📈 Stats:\n{df.describe().to_string()}")