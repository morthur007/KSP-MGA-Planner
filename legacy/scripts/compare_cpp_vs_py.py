import csv
import math

def load_states(filepath):
    states = {}
    epochs = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["body"]
            epochs[name] = float(row["et_seconds"])
            states[name] = {
                "r": [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                "v": [float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["vz_m_s"])],
            }
    return epochs, states

def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))

def sub(a, b):
    return [a[i] - b[i] for i in range(3)]

def norm(a):
    return math.sqrt(dot(a, a))

file_cpp = "data/jnsq_gate0/principia_sample_future_exact.csv"
file_py  = "data/jnsq_gate0/principia_target.csv"

epochs_cpp, states_cpp = load_states(file_cpp)
epochs_py, states_py = load_states(file_py)

print(f"{'Corpo':<16} | {'pos m':>14} | {'dt epoch s':>14} | {'dt aparente s':>14}")
print("-" * 70)

errors = []
for name in sorted(states_cpp):
    if name not in states_py:
        continue

    r_cpp = states_cpp[name]["r"]
    r_py = states_py[name]["r"]
    v_cpp = states_cpp[name]["v"]

    dr = sub(r_cpp, r_py)
    pos_err = norm(dr)

    vv = max(dot(v_cpp, v_cpp), 1e-30)
    apparent_dt = dot(dr, v_cpp) / vv

    epoch_dt = epochs_cpp[name] - epochs_py[name]

    errors.append(pos_err)

    print(f"{name:<16} | {pos_err:14.6f} | {epoch_dt:14.9f} | {apparent_dt:14.6f}")

rms = math.sqrt(sum(e * e for e in errors) / len(errors))
print("-" * 70)
print(f"RMS FINAL: {rms:.6f} m")