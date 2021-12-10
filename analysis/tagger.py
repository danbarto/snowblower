#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import awkward as ak
import uproot
import numpy as np
import glob
import os

from functools import partial

from tools.helpers import get_four_vec_fromPtEtaPhiM, match, delta_r, delta_r2, yahist_2D_lookup

from coffea import processor, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from yahist import Hist2D

import numpy as np

def match_count(first, second, deltaRCut=0.4):
    drCut2 = deltaRCut**2
    combs = ak.cartesian([first, second], nested=True)
    return ak.sum((delta_r2(combs['0'], combs['1'])<drCut2), axis=2)


effs = {}
for x in ['0b', '1b', '2b', '1h']:
    effs[x] = Hist2D.from_json(os.path.expandvars("../data/htag/h_eff_tthbb_%s.json"%x))


def get_weight(effs, pt, eta):
    # NOTE need to load the efficiencies. need 2d yahist lookup.
    # needs to be done individually for each class of jets
    return yahist_2D_lookup(
        effs,
        pt,
        abs(eta),
        )


dataset_axis            = hist.Cat("dataset",       "Primary dataset")
pt_axis                 = hist.Bin("pt",            r"$p_{T}$ (GeV)",   50, 0, 1000)  # 5 GeV is fine enough
mass_axis               = hist.Bin("mass",          r"$M$ (GeV)",   50, 0, 500)  # 5 GeV is fine enough
eta_axis                = hist.Bin("eta",           r"$\eta$",          48, 0, 2.4)
phi_axis                = hist.Bin("phi",           r"$\phi$",          16, -3.2, 3.2) # reduced from 64
multiplicity_axis       = hist.Bin("multiplicity",  r"N",               3, -0.5, 2.5)

desired_output = {
    #"PV_npvs" :         hist.Hist("PV_npvs", dataset_axis, ext_multiplicity_axis),
    #"PV_npvsGood" :     hist.Hist("PV_npvsGood", dataset_axis, ext_multiplicity_axis),
    "0b":           hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "1b":           hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "2b":           hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "0b_tagged":    hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "1b_tagged":    hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "2b_tagged":    hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "0h":           hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "0h_tagged":    hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "1h":           hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "1h_tagged":    hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),
    "tagged":       hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),  # actually tagged, independent of gen match
    "inclusive":    hist.Hist("Counts", dataset_axis, pt_axis, eta_axis, phi_axis, mass_axis),  # all AK8 jets, reweighted according to efficiency
    "NH_true":      hist.Hist("Counts", dataset_axis, multiplicity_axis),
    "NH_weight":    hist.Hist("Counts", dataset_axis, multiplicity_axis),
    }


class measure_eff(processor.ProcessorABC):
    def __init__(self, accumulator={}):

        #self.leptonSF = LeptonSF(year=year)

        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):

        output = self.accumulator.identity()

        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=0

        ev = events[presel]
        dataset = ev.metadata['dataset']

        fat = ev.FatJet[
            (ev.FatJet.pt>200) &\
            (abs(ev.FatJet.eta)<2.4) &\
            (ev.FatJet.jetId>0)
        ]

        higgs = ev.GenPart[((abs(ev.GenPart.pdgId)==25)&(ev.GenPart.status==62))]

        bquark = ev.GenPart[((abs(ev.GenPart.pdgId)==5)&(ev.GenPart.status==71))]

        selection = PackedSelection()
        selection.add('dilep',         ak.num(fat)>0 )
        selection.add('filter',        ev.MET.pt > 200 )

        bl_reqs = ['dilep', 'filter']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)

        nb_in_fat = match_count(fat, bquark, deltaRCut=0.8)

        nhiggs_in_fat = match_count(fat, higgs, deltaRCut=0.8)
        zerohiggs = (nhiggs_in_fat==0)
        onehiggs = (nhiggs_in_fat==1)

        zerob = ((nb_in_fat==0) & (zerohiggs))  # verified to work!
        oneb  = ((nb_in_fat==1) & (zerohiggs))  # verified to work!
        twob  = ((nb_in_fat>=2) & (zerohiggs))  # verified to work!
        tagged = (fat.deepTagMD_HbbvsQCD > 0.80)

        output['0h'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['0h_tagged'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs & tagged)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs & tagged)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs & tagged)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & zerohiggs & tagged)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['1h'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['1h_tagged'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs & tagged)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs & tagged)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs & tagged)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & onehiggs & tagged)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['0b'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & zerob)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & zerob)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & zerob)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & zerob)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['0b_tagged'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & zerob & tagged)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & zerob & tagged)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & zerob & tagged)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & zerob & tagged)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['1b'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & oneb)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & oneb)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & oneb)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & oneb)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['1b_tagged'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & oneb & tagged)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & oneb & tagged)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & oneb & tagged)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & oneb & tagged)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['2b'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & twob)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & twob)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & twob)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & twob)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['2b_tagged'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & twob & tagged)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & twob & tagged)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & twob & tagged)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & twob & tagged)].mass)),
            #weight = weight.weight()[baseline]
        )

        return output

    def postprocess(self, accumulator):
        return accumulator# This will need a NanoAOD processor


class apply_eff(processor.ProcessorABC):
    def __init__(self, accumulator={}, effs={}):

        self.effs = effs  # NOTE need efficiencies for every sample!

        self._accumulator = processor.dict_accumulator( accumulator )

    @property
    def accumulator(self):
        return self._accumulator

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):

        output = self.accumulator.identity()

        # we can use a very loose preselection to filter the events. nothing is done with this presel, though
        presel = ak.num(events.Jet)>=0

        ev = events[presel]
        dataset = ev.metadata['dataset']

        fat = ev.FatJet[
            (ev.FatJet.pt>200) &\
            (abs(ev.FatJet.eta)<2.4) &\
            (ev.FatJet.jetId>0)
        ]

        higgs = ev.GenPart[((abs(ev.GenPart.pdgId)==25)&(ev.GenPart.status==62))]

        bquark = ev.GenPart[((abs(ev.GenPart.pdgId)==5)&(ev.GenPart.status==71))]

        selection = PackedSelection()
        selection.add('dilep',         ak.num(fat)>0 )
        selection.add('filter',        ev.MET.pt > 200 )

        bl_reqs = ['dilep', 'filter']

        bl_reqs_d = { sel: True for sel in bl_reqs }
        baseline = selection.require(**bl_reqs_d)

        nb_in_fat = match_count(fat, bquark, deltaRCut=0.8)

        nhiggs_in_fat = match_count(fat, higgs, deltaRCut=0.8)
        zerohiggs = (nhiggs_in_fat==0)
        onehiggs = (nhiggs_in_fat==1)

        zerob = ((nb_in_fat==0) & (zerohiggs))  # verified to work!
        oneb  = ((nb_in_fat==1) & (zerohiggs))  # verified to work!
        twob  = ((nb_in_fat>=2) & (zerohiggs))  # verified to work!
        tagged = (fat.deepTagMD_HbbvsQCD > 0.80)

        w_0b = get_weight(self.effs[dataset]['0b'], fat.pt, fat.eta)
        w_1b = get_weight(self.effs[dataset]['1b'], fat.pt, fat.eta)
        w_2b = get_weight(self.effs[dataset]['2b'], fat.pt, fat.eta)
        w_1h = get_weight(self.effs[dataset]['1h'], fat.pt, fat.eta)

        w_all = w_0b * zerob + w_1b * oneb + w_2b * twob # + w_1h * onehiggs  # this should work
        if not np.isnan(sum(sum(self.effs[dataset]['1h'].counts))):
#        np.isnan(sum(ak.flatten(w_1h * onehiggs))):
            w_all = w_all + w_1h * onehiggs

        #if np.isnan(sum(ak.flatten(w_all))):
        #    print ("Weight is NaN for dataset:", dataset)

        output['tagged'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[(baseline & tagged)].pt)),
            eta = ak.to_numpy(ak.flatten(fat[(baseline & tagged)].eta)),
            phi = ak.to_numpy(ak.flatten(fat[(baseline & tagged)].phi)),
            mass = ak.to_numpy(ak.flatten(fat[(baseline & tagged)].msoftdrop)),
            #weight = weight.weight()[baseline]
        )

        output['inclusive'].fill(
            dataset = dataset,
            pt  = ak.to_numpy(ak.flatten(fat[baseline].pt)),
            eta = ak.to_numpy(ak.flatten(fat[baseline].eta)),
            phi = ak.to_numpy(ak.flatten(fat[baseline].phi)),
            mass = ak.to_numpy(ak.flatten(fat[baseline].msoftdrop)),
            weight = np.nan_to_num(ak.to_numpy(ak.flatten(w_all[baseline])), 0),  # NOTE: added the replacement here, hopefully faster.
        )

        fat_baseline = fat[baseline]
        fat_tagged = fat_baseline[(fat_baseline.deepTagMD_HbbvsQCD > 0.80)]

        output['NH_true'].fill(
            dataset=dataset,
            multiplicity = ak.num(fat_tagged, axis=1),
        )

        # NOTE: this is particularly difficult because every event contributes to every bin.
        output['NH_weight'].fill(
            dataset=dataset,
            multiplicity = np.zeros_like(ak.num(fat[baseline], axis=1)),
            weight = np.nan_to_num(ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output['NH_weight'].fill(
            # This already includes the overflow, so everything >0.
            # In the end this is all we care about, we don't differenciate N_H=2 from N_H=1
            dataset=dataset,
            multiplicity = np.ones_like(ak.num(fat[baseline], axis=1)),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )

        # NOTE: bottom line is: if you want to "apply" an NH>0 requirement you just reweight *all* your
        # events with np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0)
        # (1-w_all) is the probability of *not* tagging a fat jet, so Prod_{i}(1-w_all_{i}) is the
        # probability of having NH=0, P(NH=0). 1-P(NH=0) = P(NH>0), which is exactly what we want.

        return output

    def postprocess(self, accumulator):
        return accumulator# This will need a NanoAOD processor


if __name__ == '__main__':

    from yaml import Loader, Dumper
    import yaml


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--run_delphes', action='store_true', default=None, help="Run on delphes samples")
    argParser.add_argument('--run_flat', action='store_true', default=None, help="Run on flat ntuples")
    argParser.add_argument('--run_flat_remote', action='store_true', default=None, help="Run on flat ntuples")
    argParser.add_argument('--dask', action='store_true', default=None, help="Run on DASK cluster")
    args = argParser.parse_args()


    with open('../data/samples.yaml', 'r') as f:
        samples = yaml.load(f, Loader = Loader)

    #f_in = "root://cmseos.fnal.gov//store/user/snowmass/Snowmass2021//Delphes/DY0Jets_MLL-50_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PUDY0Jets_MLL-50_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_127_3.root"
    f_in = "root://cmseos.fnal.gov//store/user/snowmass/Snowmass2021/Delphes/ttHTobb_M125_TuneCUETP8M2_14TeV-powheg-pythia8_200PU/ttHTobb_M125_TuneCUETP8M2_14TeV-powheg-pythia8_97_1.root"

    # https://coffeateam.github.io/coffea/api/coffea.nanoevents.NanoEventsFactory.html#coffea.nanoevents.NanoEventsFactory.from_root
    ev = NanoEventsFactory.from_root(
        f_in,
        treepath="/Delphes",
        schemaclass=DelphesSchema
    ).events()

    '''
    ev.PuppiMissingET.MET  # this is the most absurd name.

    '''


    ev_flat = NanoEventsFactory.from_root(
        #'/nfs-7/userdata/dspitzba/ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU//ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU_1.root',
        '/hadoop/cms/store/user/ewallace/ProjectMetis/2HDMa_bb_1500_750_10_HLLHC_GEN_v4/delphes/delphes_ntuple_99.root',
        treepath='myana/mytree',
        schemaclass=BaseSchema,
    ).events()

    import time


    fileset = {
        'ttHbb': [
            f_in,
        ],
        'QCD': samples['QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['delphes'],
    }

    if args.run_delphes:

        tstart = time.time()

        output_delphes = processor.run_uproot_job(
            {'Znunu': samples['ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU']['delphes']},
            treename='Delphes',
            processor_instance = DelphesProcessor(),
            executor=processor.futures_executor,
            executor_args={"schema": DelphesSchema, "workers": 4},
            chunksize=100000,
            maxchunks=None,
        )
        elapsed = time.time() - tstart

        print ("Running on delphes on the grid: %.2f"%elapsed)

    if args.run_flat_remote:

        if args.dask:
            from tools.helpers import get_scheduler_address
            from dask.distributed import Client, progress

            scheduler_address = get_scheduler_address()
            c = Client(scheduler_address)

            exe_args = {
                'client': c,
                #'function_args': {'flatten': False},
                "schema": BaseSchema,
                #"tailtimeout": 300,
                "retries": 3,
                "skipbadfiles": True
            }
            exe = processor.dask_executor

        else:
            exe = processor.futures_executor
            exe_args = {"schema": BaseSchema, "workers": 20}


        tstart = time.time()

        output_flat_remote = processor.run_uproot_job(
            {
                'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU': samples['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']['ntuples'],
                'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU']['ntuples'],
                'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU']['ntuples'],
                'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU']['ntuples'],
                'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU']['ntuples'],
                'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU']['ntuples'],
                'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
                'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
                'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
                'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            },
            treename='myana/mytree',
            processor_instance = FlatProcessor(),
            executor = exe,
            executor_args = exe_args,
            chunksize=1000000,
            maxchunks=None,
        )
        elapsed = time.time() - tstart

        print ("Done.\n")
        print ("Running on flat tuples from grid: %.2f"%elapsed)


    if args.run_flat:

        tstart = time.time()

        output_flat = processor.run_uproot_job(
            {'Znunu': glob.glob('/nfs-7/userdata/dspitzba/ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU/*.root')},
            treename='myana/mytree',
            processor_instance = FlatProcessor(),
            executor=processor.futures_executor,
            executor_args={"schema": BaseSchema, "workers": 20},
            chunksize=1000000,
            maxchunks=None,
        )
        elapsed = time.time() - tstart

        print ("Running on flat tuples from nfs: %.2f"%elapsed)

    #print(output)
