#!/usr/bin/env python
"""Write the shipped ET2026a PSDs (bilby_xG/data/noise_curves/ET2026a_*.txt)
from a checkout of https://gitlab.et-gw.eu/pub/et-sensitivity-curves.

ET2026a is the 2026 snapshot of the ET noise budget (pygwinc, 90 degree
arms), 10 km and 15 km, HF and LF detectors. Each ``total.txt`` has the
frequency, the displacement ASD, the pyGWINC strain ASD with its
high-frequency correction for a signal 15 degrees off zenith, and the same
for 0 degrees. Written here, for L = 10, 15 km:

- ``ET2026a_<L>_HF_psd_pub.txt``: the HF detector's pyGWINC (15 degree) ASD,
  squared;
- ``ET2026a_<L>_HF_psd_pub_nofsr.txt``: (displacement ASD / L)^2, without the
  high-frequency correction (which a finite-size response has itself, with
  its sky dependence);
- ``ET2026a_<L>_HFLF_psd.txt``: the xylophone, 1 / (1 / S_HF + 1 / S_LF), of
  the HF and LF (displacement ASD / L)^2.

usage: scripts/make_et2026a_psds.py ET_CURVES_CHECKOUT
"""
import os
import subprocess
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "bilby_xG", "data", "noise_curves")


def main(repo):
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    source = f"https://gitlab.et-gw.eu/pub/et-sensitivity-curves @ {commit}"
    for length in (10, 15):
        hf = np.loadtxt(os.path.join(repo, f"{length}_km", "HF", "data", "total.txt"))
        lf = np.loadtxt(os.path.join(repo, f"{length}_km", "LF", "data", "total.txt"))
        if not np.array_equal(hf[:, 0], lf[:, 0]):
            raise ValueError(f"{length} km: HF and LF on different frequencies")
        s_hf = (hf[:, 1] / (length * 1e3)) ** 2
        s_lf = (lf[:, 1] / (length * 1e3)) ** 2
        curves = {
            "HF_psd_pub": (hf[:, 2] ** 2, "HF, pyGWINC strain ASD (15 deg off zenith) squared"),
            "HF_psd_pub_nofsr": (s_hf, "HF, (displacement ASD / L)^2"),
            "HFLF_psd": (1 / (1 / s_hf + 1 / s_lf),
                         "HFLF, 1 / (1 / S_HF + 1 / S_LF) of (displacement ASD / L)^2"),
        }
        for name, (psd, what) in curves.items():
            path = os.path.join(OUT, f"ET2026a_{length}_{name}.txt")
            np.savetxt(path, np.column_stack([hf[:, 0], psd]),
                       header=f"ET2026a {length} km {what} [1/Hz]\n"
                              f"from {source}, scripts/make_et2026a_psds.py\n"
                              "frequency [Hz]  PSD [1/Hz]")
            print("->", path)


if __name__ == "__main__":
    main(sys.argv[1])
