import olca_ipc as ipc
import olca_schema as schema
import pandas as pd
import random

# ── CONFIG ────────────────────────────────────────────────────────────────────
file_id = random.randint(1000, 9999)
PORT          = 8080
SYSTEM_NAME   = "ML Rice"
IMPACT_METHOD = "ReCiPe 2016 Midpoint (H)"
PROCESS_NAME  = "ML Rice"
RUNS_PER_BLOCK = 200
OUTPUT_CSV = f"ML_Dataset_v2_{file_id}.csv"

N_MIN, N_MAX   = 120, 150
P_MIN, P_MAX   =  40,  60
K_MIN, K_MAX   =  30,  40
Zn_MIN, Zn_MAX =  10,  30

N_MID  = (N_MIN  + N_MAX)  / 2   # 135.0
P_MID  = (P_MIN  + P_MAX)  / 2   # 50.0
K_MID  = (K_MIN  + K_MAX)  / 2   # 35.0
Zn_MID = (Zn_MIN + Zn_MAX) / 2   # 20.0

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
print(f"\n📐 Midpoints — N={N_MID} P={P_MID} K={K_MID} Zn={Zn_MID}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def make_redef(name, value):
    r                  = schema.ParameterRedef()
    r.name             = name
    r.value            = value
    r.context          = schema.Ref()
    r.context.id       = process_ref.id
    r.context.ref_type = schema.RefType.Process
    return r

def build_setup(n, p, k, zn):
    setup                  = schema.CalculationSetup()
    setup.target           = schema.Ref()
    setup.target.id        = system_ref.id
    setup.target.ref_type  = schema.RefType.ProductSystem
    setup.impact_method    = schema.Ref()
    setup.impact_method.id = method_ref.id
    setup.parameters       = [
        make_redef("N_rate",       n),
        make_redef("P_rate",       p),
        make_redef("K_rate",       k),
        make_redef("Zn_rate",      zn),
        make_redef("n2o_emission", round(n * 0.003 * 1.571, 6)),
        make_redef("no3_emission", round(n * 0.009 * 4.429, 6)),
        make_redef("nh3_emission", round(n * 0.30  * 1.214, 6)),
        make_redef("po4_emission", round(p * 0.01  * 3.065, 6)),
    ]
    return setup

# ── AUTO-DETECT ImpactValue attrs ─────────────────────────────────────────────
print("\n🔎 Detecting ImpactValue structure...")
result = client.calculate(build_setup(N_MID, P_MID, K_MID, Zn_MID))
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
def run_calc(n, p, k, zn, sample_type):
    result = client.calculate(build_setup(n, p, k, zn))
    result.wait_until_ready()
    impacts = result.get_total_impacts()

    row = {
        "sample_type" : sample_type,
        "N_rate"      : round(n,  4),
        "P_rate"      : round(p,  4),
        "K_rate"      : round(k,  4),
        "Zn_rate"     : round(zn, 4),
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
total   = RUNS_PER_BLOCK * 5
records = []
failed  = 0
run_idx = 0

blocks = [
    ("OAT_N",  lambda: random.uniform(N_MIN,  N_MAX),  lambda: P_MID,                        lambda: K_MID,                        lambda: Zn_MID),
    ("OAT_P",  lambda: N_MID,                          lambda: random.uniform(P_MIN, P_MAX),  lambda: K_MID,                        lambda: Zn_MID),
    ("OAT_K",  lambda: N_MID,                          lambda: P_MID,                         lambda: random.uniform(K_MIN, K_MAX),  lambda: Zn_MID),
    ("OAT_Zn", lambda: N_MID,                          lambda: P_MID,                         lambda: K_MID,                         lambda: random.uniform(Zn_MIN, Zn_MAX)),
    ("RANDOM", lambda: random.uniform(N_MIN,  N_MAX),  lambda: random.uniform(P_MIN, P_MAX),  lambda: random.uniform(K_MIN, K_MAX),  lambda: random.uniform(Zn_MIN, Zn_MAX)),
]

print(f"\n🚀 Starting {total} runs across 5 blocks ({RUNS_PER_BLOCK} each)...\n")

for label, n_fn, p_fn, k_fn, zn_fn in blocks:
    print(f"── Block: {label} ──────────────────────────────────────────")
    block_failed = 0

    for i in range(RUNS_PER_BLOCK):
        run_idx += 1
        n_val  = round(n_fn(),  4)
        p_val  = round(p_fn(),  4)
        k_val  = round(k_fn(),  4)
        zn_val = round(zn_fn(), 4)

        try:
            row = run_calc(n_val, p_val, k_val, zn_val, label)
            records.append(row)

            if (i + 1) % 50 == 0:
                print(f"   [{i+1:>3}/{RUNS_PER_BLOCK}] "
                      f"N={n_val:.2f} P={p_val:.2f} K={k_val:.2f} Zn={zn_val:.2f} | "
                      f"GW={row.get('global_warming', 'N/A'):.4f} | "
                      f"FE={row.get('freshwater_eutrophication', 'N/A'):.6f} | "
                      f"TA={row.get('terrestrial_acidification', 'N/A'):.4f} | "
                      f"TE={row.get('terrestrial_ecotoxicity', 'N/A'):.4f}")

        except Exception as e:
            failed      += 1
            block_failed += 1
            print(f"   ⚠️  Run {run_idx} failed (N={n_val} P={p_val} K={k_val} Zn={zn_val}): {e}")
            continue

    print(f"   ✅ Block done — {RUNS_PER_BLOCK - block_failed} successful\n")

# ── SAVE ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)

col_order = [
    "sample_type", "N_rate", "P_rate", "K_rate", "Zn_rate",
    "global_warming", "freshwater_eutrophication",
    "terrestrial_acidification", "terrestrial_ecotoxicity"
]
df = df[[c for c in col_order if c in df.columns]]

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Complete — {len(df)}/{total} runs saved to '{OUTPUT_CSV}' ({failed} failed)")
print(f"\n📊 Counts per block:\n{df['sample_type'].value_counts().to_string()}")
print(f"\n📊 Preview:\n{df.head(8).to_string()}")
print(f"\n📈 Stats:\n{df.describe().to_string()}")