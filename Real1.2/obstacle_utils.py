import math


def obstacle_center(obstacle):
    return float(obstacle["x"]), float(obstacle["y"])


def obstacle_axes(obstacle):
    fallback = float(obstacle.get("r", 0.0))
    if "a" in obstacle or "b" in obstacle:
        a = float(obstacle.get("a", fallback))
        b = float(obstacle.get("b", fallback if fallback > 0.0 else a))
    else:
        a = fallback
        b = fallback

    return max(abs(a), 1e-9), max(abs(b), 1e-9)


def obstacle_angle(obstacle):
    return float(obstacle.get("theta", obstacle.get("angle", 0.0)))


def obstacle_is_ellipse(obstacle):
    a, b = obstacle_axes(obstacle)
    return (
        "a" in obstacle
        or "b" in obstacle
        or abs(a - b) > 1e-9
        or abs(obstacle_angle(obstacle)) > 1e-9
    )


def ellipse_radius_in_direction(a, b, ux, uy):
    denom = (ux / max(1e-9, a)) ** 2 + (uy / max(1e-9, b)) ** 2
    return 1.0 / math.sqrt(max(1e-12, denom))


def obstacle_radius_in_direction(obstacle, ux, uy):
    a, b = obstacle_axes(obstacle)
    theta = obstacle_angle(obstacle)
    c = math.cos(theta)
    s = math.sin(theta)

    ux_local = c * ux + s * uy
    uy_local = -s * ux + c * uy
    return ellipse_radius_in_direction(a, b, ux_local, uy_local)


def obstacle_label(obstacle):
    a, b = obstacle_axes(obstacle)
    if obstacle_is_ellipse(obstacle):
        return f"OBS a={a:.2f} b={b:.2f}m"
    return f"OBS r={a:.2f}m"
