import matplotlib.pyplot as plt

deltas = [0.04, 0.1, 0.2, 0.5, 1.0]
highs = [0.9314, 0.9524, 0.9601, 0.9652, 0.9698]
modes = [25.0, 25.0, 24.9, 24.5, 24.0]

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(deltas, highs, '-^', lw=3, markersize=10)
ax.set_title("Percentage of High quality", fontsize=24)
ax.set_xlabel(r"$\delta$", fontsize=20)
ax.tick_params(labelsize=16)
fig.set_tight_layout(True)
fig.savefig("grid-high-quality.png", dpi=100)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(deltas, modes, '-^', lw=3, markersize=10)
ax.set_title("Number of Covered Modes", fontsize=24)
ax.set_xlabel(r"$\delta$", fontsize=20)
ax.tick_params(labelsize=16)
fig.set_tight_layout(True)
fig.savefig("grid-mode.png", dpi=100)