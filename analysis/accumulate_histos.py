#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler
from coffea import hist, util
from coffea.processor import accumulate


def get_yield_in_slice(output, sl):
    for proc in ['TT', 'QCD_bEnriched_HT', 'WJetsToLNu_Njet', 'ZJetsToNuNu_HT', 'other', '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']:
        print (proc, output['MT_vs_sdmass'][proc].integrate('mt', sl).integrate('mass', slice(100,150)).sum('dataset').values())

def get_old_analysis_yields(output):
    for sl in [slice(300,350), slice(350,500), slice(500,1000)]:
        for proc in ['TT', 'QCD_bEnriched_HT', 'WJetsToLNu_Njet', 'ZJetsToNuNu_HT', 'other', '2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']:
            print (proc, output['met_pt'][proc].integrate('pt', sl).sum('dataset').values()[()]*36/3000.)

if __name__ == '__main__':

    output_scaled = accumulate(
        [
            util.load("outputs/output_signal_scaled_run20220125_150734.coffea"),
            util.load("outputs/output_QCD_scaled_run20220125_171824.coffea"),
            util.load("outputs/output_TT_scaled_run20220125_171025.coffea"),
            util.load("outputs/output_W_scaled_run20220125_170224.coffea"),
            util.load("outputs/output_Z_scaled_run20220125_183109.coffea"),
            util.load("outputs/output_other_scaled_run20220125_163014.coffea"),
        ]
    )

    get_yield_in_slice(output_scaled, slice(1000,1400))
