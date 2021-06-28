def rk4(f,t0, y0, h):
    k1 = f(t0, y0)
    k2 = f(t0 + 0.5 * h, y0 + 0.5 * k1 * h)
    k3 = f(t0 + 0.5 * h, y0 + 0.5 * k2 * h)
    k4 = f(t0 + h, y0 + k3 * h)
    y1 = y0 + (k1 + k2 + k2 + k3 + k3 + k4) * h / 6
    return y1