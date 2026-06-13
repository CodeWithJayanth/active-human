# active-human

ProMP-based predictor for human arm motion. Predicts 4-DOF right-arm joint
angles and velocities (shoulder yaw, pitch, roll, elbow) at 0.5s and 1.0s
look-ahead horizons.

## Setup

```bash
git clone https://github.com/CodeWithJayanth/active-human.git
cd active-human
pip install -r requirements.txt
```

## Run the accuracy report

Run from inside the repo root (important — running from elsewhere triggers a
cache rebuild):

```bash
python -m arm_predictor.run_tests --group all
```

Prints, per test case, the per-joint position error (deg) and velocity error
(deg/s) at the 0.5s and 1.0s horizons. Use `--group A` (or B/C/D) to run a
single group.

## Rebuild the cache (optional)

The predictor ships with a prebuilt `promp_cache.pkl`. To regenerate it:

```bash
python build_cache.py --verify    # check the existing cache
python build_cache.py --rebuild   # rebuild from scratch (slow)
```

## Layout

- `arm_predictor/` — the predictor package (predictor, motion model, config)
- `promp_cache.pkl` — prebuilt motion prior (required)
- `build_cache.py` — regenerates the cache
- `requirements.txt` — dependencies
