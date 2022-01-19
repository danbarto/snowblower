#!/usr/bin/env python3
import awkward as ak
import uproot
import numpy as np
import glob
import os
from coffea.nanoevents import NanoEventsFactory, DelphesSchema, BaseSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea import hist, processor
# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

from functools import partial

from plots.helpers import makePlot2, scale_and_merge_histos
from tools.helpers import choose, get_four_vec_fromPtEtaPhiM, get_weight, match, match_count, mt, cross
from tools.jes import JES
from tools.basic_objects import getJets, getFatjets

import warnings
warnings.filterwarnings("ignore")

import shutil

jes_corrector = JES(verbose=False)

N_bins = hist.Bin('multiplicity', r'$N$', 4, 0.5, 4.5)
N_bins2 = hist.Bin('multiplicity', r'$N$', 7, 0.5, 7.5)
N_H_bins = hist.Bin('multiplicity', r'$N$', 1, -0.5, 0.5)
mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 40, 0, 400)
mass_bins2 = hist.Bin('mass', r'$M\ (GeV)$', 3, 0, 150)
ht_bins = hist.Bin('pt', r'$H_{T}\ (GeV)$', 60, 0, 3000)
mt_bins = hist.Bin('mt', r'$M_{T}\ (GeV)$', 9, 600, 2400)
pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 40, 200, 1200)
pt_bins2 = hist.Bin('pt', r'$p_{T}\ (GeV)$', 50, 0, 500)
met_bins = hist.Bin('pt', r'$MET_{pt}\ (GeV)$', 18, 100, 1000)
eta_bins = hist.Bin("eta", "$\eta$", 33, -4, 4)
phi_bins = hist.Bin("phi", "$\phi$", 33, -4, 4)
phi_bins2 = hist.Bin("phi", "$\phi$", 16, 0, 4)
tau1_bins = hist.Bin("tau", "$\tau_1$", 10, 0, 0.7)
tau2_bins = hist.Bin("tau", "$\tau_2$", 10, 0, 0.5)
tau3_bins = hist.Bin("tau", "$\tau_3$", 10, 0, 0.4)
tau4_bins = hist.Bin("tau", "$\tau_4$", 10, 0, 0.3)
tau21_bins = hist.Bin("tau", "$\tau_4$", 50, 0, 1.0)

def n_minus_one(selection, requirements, minus_one):
    reqs_d = { sel: True for sel in requirements if not sel in minus_one }
    return selection.require(**reqs_d)


class FlatProcessor(processor.ProcessorABC):
    def __init__(self, accumulator={}, effs={}):
        self._accumulator = processor.dict_accumulator({
            "MT_vs_sdmass_central": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_0b_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_0b_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_1b_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_1b_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_2b_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_2b_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_1h_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_central_1h_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
        })

        #add accumulators as needed or create list of general accumulators 
        
        self.accumulator.update(processor.dict_accumulator( accumulator ))
        
        self.effs = effs  # NOTE need efficiencies for every sample!

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']
        
        #meta_accumulator stuff
        
        sumw = np.sum(events['genweight'])
        sumw2 = np.sum(events['genweight']**2)

        output[events.metadata['filename']]['sumWeight'] += sumw  # naming for consistency...
        output[events.metadata['filename']]['sumWeight2'] += sumw2  # naming for consistency...
        output[events.metadata['filename']]['nChunk'] += 1

        output[dataset]['sumWeight'] += sumw
        output[dataset]['sumWeight2'] += sumw2
        output[dataset]['nChunk'] += 1
        
        #define objects
        
        #electrons
                
        electron = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.elec_pt,
            eta = events.elec_eta,
            phi = events.elec_phi,
            M = events.elec_mass,
            copy = False,
        )
        electron['id'] = events.elec_idpass  # > 0 should be loose
        electron['iso'] = events.elec_isopass
        #electron['charge'] = events.elec_charge

        ele_l = electron[((electron['id']>0)&(electron['iso']>0)&(electron.pt>10)&(np.abs(electron.eta)<3))]
        
        #muons
        
        muon = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.muon_pt,
            eta = events.muon_eta,
            phi = events.muon_phi,
            M = events.muon_mass,
            copy = False,
        )
        muon['id'] = events.muon_idpass  # > 0 should be loose
        muon['iso'] = events.muon_isopass
        #muon['charge'] = events.muon_charge

        muon_l = muon[((muon['id']>0)&(muon['iso']>0)&(muon.pt>4)&(np.abs(muon.eta)<2.8))]
        
        #taus
        
        tau = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.tau_pt,
            eta = events.tau_eta,
            phi = events.tau_phi,
            M = events.tau_mass,
            copy = False,
        )
        tau['iso'] = events.tau_isopass   # > 0 should be loose
        #tau['charge'] = events.tau_charge

        tau_l = tau[((tau['iso']>0)&(tau.pt>30)&(np.abs(tau.eta)<3))]
        
        #photons
        
        #gamma = get_four_vec_fromPtEtaPhiM(
        #    None,
        #    pt = events.gamma_pt,
        #    eta = events.gamma_eta,
        #    phi = events.gamma_phi,
        #    M = events.gamma_mass,
        #    copy = False,
        #)
        #gamma['id'] = events.gamma_idpass  # > 0 should be loose
        #gamma['iso'] = events.gamma_isopass

        #gamma_l = gamma[((gamma['id']>0)&(gamma['iso']>0)&(gamma.pt>20)&(np.abs(gamma.eta)<3))]
        
        #gen
        
        gen = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.genpart_pt,
            eta = events.genpart_eta,
            phi = events.genpart_phi,
            M = events.genpart_mass,
            copy = False,
        )
        gen['pdgId'] = events.genpart_pid
        gen['status'] = events.genpart_status

        #higgs = gen[((abs(gen.pdgId)==25)&(gen.status==62))]
        higgs = gen[(abs(gen.pdgId)==25)][:,-1:]  # just get the last Higgs. Delphes is not keeping all the higgses.

        bquark = gen[((abs(gen.pdgId)==5)&(gen.status==71))]  # I suspect that Delphes does not keep b's with pt less than 20?
        # so in rare occasions you'll only have one b with status 71
        
        variations = ['central', 'up', 'down']
        for var in variations:
        
            #jets

            old_jet = getJets(events, jes_corrector, 'central')

            jet_px_old = old_jet.pt*np.cos(old_jet.phi)
            jet_py_old = old_jet.pt*np.sin(old_jet.phi)

            jet = getJets(events, jes_corrector, var)

            jet_px = jet.pt*np.cos(jet.phi)
            jet_py = jet.pt*np.sin(jet.phi)

            #follow Delphes recommendations
            jet = jet[jet.pt > 30]
            jet = jet[jet.id > 0]
            jet = jet[np.abs(jet.eta) < 3] #eta within tracker range
            jet = jet[~match(jet, ele_l, deltaRCut=0.4)] #remove electron overlap
            jet = jet[~match(jet, muon_l, deltaRCut=0.4)] #remove muon overlap

            btag = jet[jet.btag>0] #loose wp for now

            #fatjets

            # Need FatJets and GenParts
            # FatJets start at pt>200 and go all the way to eta 3.x
            # This should be fine?
            # Objects are defined here: https://twiki.cern.ch/twiki/bin/view/CMS/DelphesInstructions
            # restrict abs(eta) to 2.8 (whatever the tracker acceptance of PhaseII CMS is)

            fatjet = getFatjets(events, jes_corrector, var)

            fatjet = fatjet[np.abs(fatjet.eta) < 3] #eta within tracker range        
            fatjet = fatjet[~match(fatjet, ele_l, deltaRCut=0.8)] #remove electron overlap
            fatjet = fatjet[~match(fatjet, muon_l, deltaRCut=0.8)] #remove muon overlap

            extrajet  = jet[~match(jet, fatjet, deltaRCut=1.2)] # remove AK4 jets that overlap with AK8 jets
            extrabtag = extrajet[extrajet.btag>0] #loose wp for now

            tau21 = np.divide(fatjet.tau2, fatjet.tau1)

            fatjet_on_h = fatjet[np.abs(fatjet.mass-125)<25]
            on_h = (ak.num(fatjet_on_h) > 0)

            lead_fatjet = fatjet[:,0:1]

            difatjet = choose(fatjet, 2)
            dijet = choose(jet[:,:4], 2)  # only take the 4 leading jets
            #di_AK8_AK4 = cross(extrajet, fatjet)

            dphi_difatjet = np.arccos(np.cos(difatjet['0'].phi-difatjet['1'].phi))
            dphi_dijet = np.arccos(np.cos(dijet['0'].phi-dijet['1'].phi))
            #dphi_AK8_AK4 = np.arccos(np.cos(di_AK8_AK4['0'].phi-di_AK8_AK4['1'].phi)) # not back-to-back
            AK8_QCD_veto = ak.all(dphi_difatjet<3.0, axis=1)  # veto any event with a back-to-back dijet system. No implicit cut on N_AK8 (ak.all!)
            AK4_QCD_veto = ak.all(dphi_dijet<3.0, axis=1)  # veto any event with a back-to-back dijet system. No implicit cut on N_AK4 (ak.all!)
            #min_dphi_AK8_AK4 = ak.to_numpy(ak.min(dphi_AK8_AK4, axis=1))
        
            #MET
        
            met_pt = ak.flatten(events.metpuppi_pt)
            met_phi = ak.flatten(events.metpuppi_phi)

            met_px = met_pt*np.cos(met_phi)
            met_py = met_pt*np.sin(met_phi)
            met_px_new = met_px - ak.sum(jet_px-jet_px_old, axis=1)
            met_py_new = met_py - ak.sum(jet_py-jet_py_old, axis=1)
            met_pt = np.sqrt(met_px_new**2+met_py_new**2)

            mt_AK8_MET = mt(fatjet.pt, fatjet.phi, met_pt, met_phi)
            min_mt_AK8_MET = ak.to_numpy(ak.min(mt_AK8_MET, axis=1))
            min_dphi_AK8_MET = ak.to_numpy(ak.min(np.arccos(np.cos(fatjet.phi-met_phi)), axis=1))
            min_dphi_AK4_MET = ak.to_numpy(ak.min(np.arccos(np.cos(jet.phi-met_phi)), axis=1))
            #min_dphi_AK4clean_MET = ak.to_numpy(ak.min(np.arccos(np.cos(extrajet.phi-met_phi)), axis=1))
            
            nb_in_fat = match_count(fatjet, bquark, deltaRCut=0.8)
            nhiggs_in_fat = match_count(fatjet, higgs, deltaRCut=0.8)
            zerohiggs = (nhiggs_in_fat==0)
            onehiggs = (nhiggs_in_fat==1)

            zerob = ((nb_in_fat==0) & (zerohiggs))  # verified to work!
            oneb  = ((nb_in_fat==1) & (zerohiggs))  # verified to work!
            twob  = ((nb_in_fat>=2) & (zerohiggs))  # verified to work!

            w_0b = get_weight(self.effs[dataset]['0b'], fatjet.pt, fatjet.eta)
            w_1b = get_weight(self.effs[dataset]['1b'], fatjet.pt, fatjet.eta)
            w_2b = get_weight(self.effs[dataset]['2b'], fatjet.pt, fatjet.eta)
            w_1h = get_weight(self.effs[dataset]['1h'], fatjet.pt, fatjet.eta)
            
            if var == 'central':
                for tagger in ['', '_0b_up', '_0b_down', '_1b_up', '_1b_down', '_2b_up', '_2b_down', '_1h_up', '_1h_down']:
                    if tagger == '_0b_up':
                         w_0b = get_weight(self.effs[dataset]['0b'], fatjet.pt, fatjet.eta)*1.05
                    if tagger == '_0b_down':
                         w_0b = get_weight(self.effs[dataset]['0b'], fatjet.pt, fatjet.eta)*0.95
                    if tagger == '_1b_up':
                         w_1b = get_weight(self.effs[dataset]['1b'], fatjet.pt, fatjet.eta)*1.05
                    if tagger == '_1b_down':
                         w_1b = get_weight(self.effs[dataset]['1b'], fatjet.pt, fatjet.eta)*0.95
                    if tagger == '_2b_up':
                         w_2b = get_weight(self.effs[dataset]['2b'], fatjet.pt, fatjet.eta)*1.05
                    if tagger == '_2b_down':
                         w_2b = get_weight(self.effs[dataset]['2b'], fatjet.pt, fatjet.eta)*0.95
                    if tagger == '_1h_up':
                         w_1h = get_weight(self.effs[dataset]['1h'], fatjet.pt, fatjet.eta)*1.05
                    if tagger == '_1h_down':
                         w_1h = get_weight(self.effs[dataset]['1h'], fatjet.pt, fatjet.eta)*0.95

                    w_all = w_0b * zerob + w_1b * oneb + w_2b * twob # + w_1h * onehiggs  # this should work
                    if not np.isnan(sum(sum(self.effs[dataset]['1h'].counts))):
                        w_all = w_all + w_1h * onehiggs

                    #selections
                    selection = PackedSelection()

                    selection.add('ele_veto', ak.num(ele_l, axis=1)==0)
                    selection.add('mu_veto',  ak.num(muon_l, axis=1)==0)
                    selection.add('tau_veto', ak.num(tau_l, axis=1)==0)
                    selection.add('met',      met_pt>300)
                    selection.add('nAK4',     ak.num(jet, axis=1)>1)
                    selection.add('nAK8',     ak.num(fatjet, axis=1)>0)
                    selection.add('min_AK8_pt', ak.min(fatjet.pt, axis=1)>300)
                    selection.add('dphi_AK8_MET>1', min_dphi_AK8_MET>1.0)
                    selection.add('dphi_AK4_MET<3', min_dphi_AK4_MET<3.0)
                    selection.add('dphi_AK4_MET>1', min_dphi_AK4_MET>1.0)
                    selection.add('AK4_QCD_veto', AK4_QCD_veto)
                    selection.add('AK8_QCD_veto', AK8_QCD_veto)
                    selection.add('on_H',     on_h)
                    selection.add('MT>600',   min_mt_AK8_MET>600)

                    #weights

                    weight = Weights(len(events))
                    weight.add("NH>0", np.nan_to_num(1-ak.prod(1-w_all, axis=1), 0))

                    #outputs

                    baseline = [
                        'ele_veto',
                        'mu_veto',
                        'tau_veto',
                        'met',
                        'nAK8',
                    ]

                    tight = [
                        'ele_veto',
                        'mu_veto',
                        'tau_veto',
                        'met',
                        'nAK8',
                        'nAK4',
                        'min_AK8_pt',
                        'dphi_AK8_MET>1',
                        'dphi_AK4_MET<3',
                        'dphi_AK4_MET>1',
                        'AK4_QCD_veto',
                        'AK8_QCD_veto',
                        'on_H',
                        'MT>600',
                    ]

                    tmp_sel = n_minus_one(selection, tight, ['on_H'])
                    output["MT_vs_sdmass"+'_'+var+tagger].fill(
                        dataset=dataset,
                        mt=min_mt_AK8_MET[tmp_sel],
                        mass=ak.flatten(lead_fatjet.mass[tmp_sel]),
                        weight = weight.weight()[tmp_sel]
                    )
                    
            elif var == 'up' or var == 'down':
                w_0b = get_weight(self.effs[dataset]['0b'], fatjet.pt, fatjet.eta)
                w_1b = get_weight(self.effs[dataset]['1b'], fatjet.pt, fatjet.eta)
                w_2b = get_weight(self.effs[dataset]['2b'], fatjet.pt, fatjet.eta)
                w_1h = get_weight(self.effs[dataset]['1h'], fatjet.pt, fatjet.eta)
                
                w_all = w_0b * zerob + w_1b * oneb + w_2b * twob # + w_1h * onehiggs  # this should work
                if not np.isnan(sum(sum(self.effs[dataset]['1h'].counts))):
                    w_all = w_all + w_1h * onehiggs

                #selections
                selection = PackedSelection()

                selection.add('ele_veto', ak.num(ele_l, axis=1)==0)
                selection.add('mu_veto',  ak.num(muon_l, axis=1)==0)
                selection.add('tau_veto', ak.num(tau_l, axis=1)==0)
                selection.add('met',      met_pt>300)
                selection.add('nAK4',     ak.num(jet, axis=1)>1)
                selection.add('nAK8',     ak.num(fatjet, axis=1)>0)
                selection.add('min_AK8_pt', ak.min(fatjet.pt, axis=1)>300)
                selection.add('dphi_AK8_MET>1', min_dphi_AK8_MET>1.0)
                selection.add('dphi_AK4_MET<3', min_dphi_AK4_MET<3.0)
                selection.add('dphi_AK4_MET>1', min_dphi_AK4_MET>1.0)
                selection.add('AK4_QCD_veto', AK4_QCD_veto)
                selection.add('AK8_QCD_veto', AK8_QCD_veto)
                selection.add('on_H',     on_h)
                selection.add('MT>600',   min_mt_AK8_MET>600)

                #weights

                weight = Weights(len(events))
                weight.add("NH>0", np.nan_to_num(1-ak.prod(1-w_all, axis=1), 0))

                #outputs

                baseline = [
                        'ele_veto',
                        'mu_veto',
                        'tau_veto',
                        'met',
                        'nAK8',
                ]

                tight = [
                        'ele_veto',
                        'mu_veto',
                        'tau_veto',
                        'met',
                        'nAK8',
                        'nAK4',
                        'min_AK8_pt',
                        'dphi_AK8_MET>1',
                        'dphi_AK4_MET<3',
                        'dphi_AK4_MET>1',
                        'AK4_QCD_veto',
                        'AK8_QCD_veto',
                        'on_H',
                        'MT>600',
                ]

                tmp_sel = n_minus_one(selection, tight, ['on_H'])
                output["MT_vs_sdmass"+'_'+var].fill(
                    dataset=dataset,
                    mt=min_mt_AK8_MET[tmp_sel],
                    mass=ak.flatten(lead_fatjet.mass[tmp_sel]),
                    weight = weight.weight()[tmp_sel]
                )
                

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from yaml import Loader, Dumper
    import yaml
    from yahist import Hist1D, Hist2D
    import json


    import argparse

    argParser = argparse.ArgumentParser(description = "Argument parser")
    argParser.add_argument('--run_flat', action='store_true', default=None, help="Run on flat ntuples")
    argParser.add_argument('--dask', action='store_true', default=None, help="Run on DASK cluster")
    args = argParser.parse_args()

    with open('../data/samples.yaml', 'r') as f:
        samples = yaml.load(f, Loader = Loader)

    if args.run_flat:

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
        
        fileset = {
            'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU': samples['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']['skim'],
            'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU']['skim'],
            'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']['ntuples'],
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']['ntuples'],
            #'2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500']['ntuples'],
            #'2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750']['ntuples'],
            #'2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000']['ntuples'],
            #'2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500']['ntuples'],
            #'2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750']['ntuples'],
            #'2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000']['ntuples'],
        }

        meta_accumulator = {}
        for sample in fileset:
            if sample not in meta_accumulator:
                meta_accumulator.update({sample: processor.defaultdict_accumulator(int)})
            for f in fileset[sample]:
                meta_accumulator.update({f: processor.defaultdict_accumulator(int)})

        run2_to_delphes = {
            'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU': 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',
            'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
            'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8',
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8', 
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT200to300_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT300to500_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT500to700_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT700to1000_TuneCP5_13TeV-madgraph-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
        }
        
        effs = {}
        for s in fileset.keys():
            effs[s] = {}
            for b in ['0b', '1b', '2b', '1h']:
                effs[s][b] = Hist2D.from_json(os.path.expandvars("../data/htag/eff_%s_%s.json"%(run2_to_delphes[s],b)))
        
        output_flat = processor.run_uproot_job(
            fileset,
            treename='mytree',
            processor_instance = FlatProcessor(accumulator=meta_accumulator, effs=effs),
            executor = exe,
            executor_args = exe_args,
            chunksize=1000000,
            maxchunks=None,
        )
        
        import time
        tstart = time.time()
        
        nevents = {}

        
        signal = [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000'
        ]
        
        for sample in fileset:
            nevents[sample] = 0
            if sample in signal:
                nevents[sample] = samples[sample]['nevents']
            else:    
                for file in fileset[sample]:
                    with uproot.open(file+':nevents') as counts:
                        nevents[file] = counts.counts()[0] 
                    nevents[sample] += nevents[file]

        elapsed = time.time() - tstart

        print ("Done.\n")
        print ("Getting nevents from root files: %.2f"%elapsed)
        
        import cloudpickle
        import gzip
        outname = 'output_flat'
        os.system("mkdir -p histos/")
        print('Saving output in %s...'%("histos/" + outname + ".pkl.gz"))
        with gzip.open("histos/" + outname + ".pkl.gz", "wb") as fout:
            cloudpickle.dump(output_flat, fout)
        print('Done!')
        
        meta = {}

        for sample in fileset:
            meta[sample] = output_flat[sample]
            meta[sample]['xsec'] = samples[sample]['xsec']
            meta[sample]['nevents'] = nevents[sample]

        scaled_output = {}
        
        for key in output_flat.keys():
            if type(output_flat[key]) is not type(output_flat['/nfs-7/userdata/ewallace/Hbb_MET/merge_ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU_v15/merge_1.root']):
                scaled_output[key] = scale_and_merge_histos(output_flat[key], meta, fileset, lumi=36)
                
        import cloudpickle
        import gzip
        outname = 'limits_36'
        os.system("mkdir -p histos/")
        print('Saving output in %s...'%("histos/" + outname + ".pkl.gz"))
        with gzip.open("histos/" + outname + ".pkl.gz", "wb") as fout:
            cloudpickle.dump(scaled_output, fout)
        print('Done!')