#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler
from coffea import hist, util
from coffea.processor import accumulate


def get_yield_in_slice(output, sl):
    for proc in ['TT', 'QCD_bEnriched_HT', 'WJetsToLNu_Njet', 'ZJetsToNuNu_HT', 'other', '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']:
        print (proc, output['MT_vs_sdmass'][proc].integrate('mt', sl).integrate('mass', slice(100,150)).sum('dataset').values())


if __name__ == '__main__':

    output_scaled = accumulate(
        [
            util.load("outputs/output_signal_scaled_run20220125_150734.coffea"),
            util.load("outputs/output_other_scaled_run20220125_163014.coffea"),
        ]
    )

    get_yield_in_slice(output_scaled, slice(1000,1400))
