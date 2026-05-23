import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from shapely.geometry import LinearRing

# -------------------------
# Config
# -------------------------
path_in = "barreira_interna.txt"

path_smooth_out = "barreira_suavizada_interna.txt"
path_outer_out  = "barreira_suavizada_externa.txt"

s_smooth = 10       # suavização (aumenta => mais suave)
n_points = 2000        # pontos no caminho suavizado
offset = 3      # distância do contorno exterior (m)
join_style = 1         # 1=round, 2=mitre, 3=bevel

# -------------------------
# 1) Ler original
# -------------------------
pts = np.loadtxt(path_in)
x, y = pts[:, 0], pts[:, 1]

# Remove duplicação se já estiver fechado
if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-9:
    x = x[:-1]
    y = y[:-1]

# -------------------------
# 2) Suavizar + fechar (periódico)
# -------------------------
tck, u = splprep([x, y], s=s_smooth, per=True)
u_new = np.linspace(0, 1, n_points, endpoint=False)
xs, ys = splev(u_new, tck)

smooth = np.column_stack([xs, ys])
smooth_closed = np.vstack([smooth, smooth[0]])  # fecha

np.savetxt(path_smooth_out, smooth_closed, fmt="%.6f")
print(f"Guardado: {path_smooth_out}")

# -------------------------
# 3) Contorno exterior (buffer)
# -------------------------
ring = LinearRing(smooth_closed)
corridor = ring.buffer(offset, join_style=join_style).buffer(0)

if corridor.geom_type == "MultiPolygon":
    corridor = max(corridor.geoms, key=lambda g: g.area)

outer = np.array(corridor.exterior.coords)  # já vem fechado
np.savetxt(path_outer_out, outer, fmt="%.6f")
print(f"Guardado: {path_outer_out}")

# -------------------------
# 4) Plot
# -------------------------
plt.figure(figsize=(9, 7))

# original (opcional)
plt.plot(x, y, "o", markersize=2, label="Original (ficheiro)")

# suavizado
plt.plot(smooth_closed[:, 0], smooth_closed[:, 1], "-", linewidth=2, label="Suavizado (fechado)")

# contorno exterior
plt.plot(outer[:, 0], outer[:, 1], "-", linewidth=2, label=f"Contorno exterior (offset={offset})")

plt.gca().set_aspect("equal", "box")
plt.grid(True)
plt.legend()
plt.title("Original vs Suavizado vs Contorno Exterior")
plt.show()