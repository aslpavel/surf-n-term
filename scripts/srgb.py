#!/usr/bin/env python3
import numpy as np
import numpy.linalg as la
import argparse
import inspect
import math


def true_l2s(value):
    """Convert linear component to SRGB gamma corrected"""
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def true_s2l(value):
    """Convert SRGB gamma corrected component to linear"""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def l2s_sqrt_fit():
    def l2s_xs(x0):
        x1 = math.sqrt(x0)  # x ^ 1/2
        x2 = math.sqrt(x1)  # x ^ 1/4
        x3 = math.sqrt(x2)  # x ^ 1/8
        return [x0, x1, x2, x3]

    def l2s(cs, x):
        if x <= 0.0031308:
            return x * 12.92
        return cs.dot(l2s_xs(x))

    xs = []
    ys = []
    for x in np.linspace(0.0031308, 1.0, 65536):
        xs.append(l2s_xs(x))
        ys.append(true_l2s(x))
    xs = np.array(xs)
    ys = np.array(ys)

    cs = la.lstsq(xs, ys, rcond=-1)[0]
    print("coefficient:", cs)

    # evaluate maximum error
    errors = [
        (abs(true_l2s(x) - l2s(cs, x)), x)
        for x in np.linspace(true_s2l(1 / 255), 1.0, 100000)
    ]
    print("maximum error {} at {}".format(*max(*errors)))

    errors_roundtrip = []
    for x in range(256):
        r = int(round(l2s(cs, true_s2l(x / 255.0)) * 255.0))
        if x != r:
            errors_roundtrip.append("{} {}".format(hex(x), hex(r)))
    print(
        "round trip errors {} at:\n{}".format(
            len(errors_roundtrip), "\n".join(errors_roundtrip)
        )
    )

    return lambda x: l2s(cs, x)


def main():
    """Generate polynomial interpolation of SRGB <-> Linear RGB convertion"""
    # parser = argparse.ArgumentParser(description=inspect.getdoc(main))
    l2s_sqrt_fit()


if __name__ == "__main__":
    main()
