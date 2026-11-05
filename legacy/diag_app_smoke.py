"""Headless smoke test for R8_koumyou_v3.py (legacy, superseded by iridis.py) using
Streamlit's AppTest -- runs the actual script and simulates widget interactions without
needing a real browser, to catch Python-level exceptions in the S4 backend integration path.

Run from the repo root (data/nk is resolved relative to the process cwd, not this file):
    cd koumyou_sim && python3 legacy/diag_app_smoke.py
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root, for `from RCWA2D import ...`
APP_PATH = os.path.join(_HERE, "R8_koumyou_v3.py")
from streamlit.testing.v1 import AppTest

def check(at, label):
    if at.exception:
        print(f"[{label}] EXCEPTION:")
        for e in at.exception:
            print(f"  {e}")
        return False
    print(f"[{label}] OK (no exception)")
    return True

ok = True

at = AppTest.from_file(APP_PATH, default_timeout=60)
at.run()
ok &= check(at, "initial load (0 columns)")

# Add two columns via the button
for _ in range(2):
    at.button[0].click().run()
ok &= check(at, "after adding 2 columns")

# Add a per-column pair film layer to column 1 (cid=1)
add_pair_btn = None
for b in at.button:
    if b.key == 'ladd_p_1':
        add_pair_btn = b
        break
if add_pair_btn is not None:
    add_pair_btn.click().run()
    ok &= check(at, "add per-column pair film layer")
else:
    print("[add per-column pair] button not found -- skipped")

# Switch solver backend to S4 and confirm the spectrum section (which triggers calc_rcwa1d)
# runs through S4 without exception.
backend_radio = None
for r in at.radio:
    if r.label == 'RCWA backend':
        backend_radio = r
        break
if backend_radio is not None:
    backend_radio.set_value('S4').run()
    ok &= check(at, "switch solver backend to S4 (spectrum recompute)")
else:
    print("[S4 backend radio] not found -- HAVE_S4 may be False -- skipped")


# Switch to Angle sweep (3D) mode -- exercises the 0/30/60 swatches and the
# Reflectance vs Angle / vs Wavelength charts (calc_rcwa1d called once per
# swept angle plus 3x more for the fixed swatch angles).
angle_mode_radio = None
for r in at.radio:
    if r.label == 'Angle mode':
        angle_mode_radio = r
        break
if angle_mode_radio is not None:
    angle_mode_radio.set_value('Angle sweep (3D)').run()
    ok &= check(at, "switch to Angle sweep (3D) mode")
else:
    print("[Angle mode radio] not found -- skipped")

print()
print("ALL OK" if ok else "FAILURES DETECTED")
