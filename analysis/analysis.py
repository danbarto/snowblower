#!/usr/bin/env python3
import awkward as ak
import uproot
import numpy as np
import glob
import os
from coffea.nanoevents import NanoEventsFactory, DelphesSchema, BaseSchema
from coffea.analysis_tools import Weights, PackedSelection
from coffea import hist, processor, util
# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

from functools import partial

from plots.helpers import makePlot2, scale_and_merge_histos
from tools.helpers import choose, delta_r, get_four_vec_fromPtEtaPhiM, get_weight, match, match_count, mt, cross, pad_and_flatten
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
met_bins_ext = hist.Bin('pt', r'$MET_{pt}\ (GeV)$', 20, 0, 1000)
eta_bins = hist.Bin("eta", "$\eta$", 33, -4, 4)
phi_bins = hist.Bin("phi", "$\phi$", 33, -4, 4)
phi_bins2 = hist.Bin("phi", "$\phi$", 20, 0, 4)
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
            'cutflow': processor.defaultdict_accumulator(
                # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int)
            ),
            "Mtt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                ht_bins,
            ),
            "Mtt_inclusive": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                ht_bins,
            ),
            "met_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                met_bins,
            ),
            "genmet_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                met_bins_ext,
            ),
            "genmet_pt_inclusive": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                met_bins_ext,
            ),
            #"met_pt_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    met_bins,
            #),
            "dphi_AK4_MET": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                phi_bins2,
            ),
            #"dphi_AK4_MET_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    phi_bins2,
            #),
            "dphi_AK8_MET": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                phi_bins2,
            ),
            #"dphi_AK8_MET_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    phi_bins2,
            #),
            "AK4_QCD_veto": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                phi_bins2,
            ),
            #"AK4_QCD_veto_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    phi_bins2,
            #),
            "AK8_QCD_veto": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                phi_bins2,
            ),
            #"AK8_QCD_veto_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    phi_bins2,
            #),
            "MT_vs_sdmass_BL": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_0b_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_0b_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_1b_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_1b_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_2b_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_2b_down": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_1h_up": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            "MT_vs_sdmass_1h_down": hist.Hist(
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
            "MT_vs_sdmass_jmr": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
                mass_bins2,
            ),
            #"AK8_sdmass_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    mass_bins,
            #),
            "MT": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
            ),
            "MT_single_lep": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mt_bins,
            ),
            "AK8_sdmass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                mass_bins,
            ),
            #"n_AK4_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    N_bins2,
            #),
            "n_AK4": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                N_bins2,
            ),
            #"min_AK8_pt_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    pt_bins,
            #),
            "min_AK8_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                pt_bins,
            ),
            #"NH_weight_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    N_H_bins,
            #),
            "NH_weight": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                N_H_bins,
            ),
            #"b_DeltaR_vs_H_pt_BL": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    pt_bins,
            #    phi_bins2,
            #),
            "b_DeltaR_vs_H_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                pt_bins,
                phi_bins2,
            ),
            "MT_vs_sdmass_LHE": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Cat("variation", "Variation"),
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
        
        #presel
        if dataset in ['WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU', 'WJetsToLNu_GenMET-100_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']:
            events = events[events.lheweight_size!=0]
        #define objects
        
        #electrons
        ele_sel = (events.elec_pt>10)
        electron = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.elec_pt[ele_sel],
            eta = events.elec_eta[ele_sel],
            phi = events.elec_phi[ele_sel],
            M = events.elec_mass[ele_sel],
            copy = False,
        )
        electron['id'] = events.elec_idpass[ele_sel]  # > 0 should be loose
        electron['iso'] = events.elec_isopass[ele_sel]

        ele_l = electron[((electron['id']>0)&(electron['iso']>0)&(electron.pt>10)&(np.abs(electron.eta)<3))]


        #muons
        mu_sel = (events.muon_pt>4)
        muon = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.muon_pt[mu_sel],
            eta = events.muon_eta[mu_sel],
            phi = events.muon_phi[mu_sel],
            M = events.muon_mass[mu_sel],
            copy = False,
        )
        muon['id'] = events.muon_idpass[mu_sel]  # > 0 should be loose
        muon['iso'] = events.muon_isopass[mu_sel]

        muon_l = muon[((muon['id']>0)&(muon['iso']>0)&(muon.pt>4)&(np.abs(muon.eta)<2.8))]
        
        #taus
        tau_sel = (events.tau_pt>30)
        tau = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.tau_pt[tau_sel],
            eta = events.tau_eta[tau_sel],
            phi = events.tau_phi[tau_sel],
            M = events.tau_mass[tau_sel],
            copy = False,
        )
        tau['iso'] = events.tau_isopass[tau_sel]   # > 0 should be loose

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

        gen_sel = ((abs(events.genpart_pid)==6) | (abs(events.genpart_pid)==5) | (abs(events.genpart_pid)==25))  # NOTE: attempt to speed up reading gigantic gen particle branches

        gen = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.genpart_pt[gen_sel],
            eta = events.genpart_eta[gen_sel],
            phi = events.genpart_phi[gen_sel],
            M = events.genpart_mass[gen_sel],
            copy = False,
        )
        gen['pdgId'] = events.genpart_pid[gen_sel]
        gen['status'] = events.genpart_status[gen_sel]

        #higgs = gen[((abs(gen.pdgId)==25)&(gen.status==62))]
        higgs = gen[(abs(gen.pdgId)==25)][:,-1:]  # just get the last Higgs. Delphes is not keeping all the higgses.

        bquark = gen[((abs(gen.pdgId)==5)&(gen.status==71))]  # I suspect that Delphes does not keep b's with pt less than 20?
        #bottom = gen[((gen.pdgId==5)&(gen.status==71))]  # I suspect that Delphes does not keep b's with pt less than 20?
        #abottom = gen[((gen.pdgId==-5)&(gen.status==71))]  # I suspect that Delphes does not keep b's with pt less than 20?
        # so in rare occasions you'll only have one b with status 71

        if dataset.count('TT_'):
            top = gen[gen.pdgId==6][:,-2:-1]
            atop = gen[gen.pdgId==-6][:,-2:-1]
            ttbar = ak.flatten(cross(top, atop))

        dibquark = choose(bquark, 2)
        b_DeltaR = delta_r(dibquark['0'], dibquark['1'])

        #find some easy way to list all the datasets that are in signal?
        lheweights = events.lheweight_val
        lheweight_ratio = {}
        #for i in range(363,463):
        #    lheweight_ratio[i] = lheweights[:,i]/lheweights[:,0]
        #for i in range(150,251):
        #    lheweight_ratio[i] = lheweights[:,i]/lheweights[:,0]
        #for i in [463,464]:
        #    lheweight_ratio[i] = lheweights[:,i]/lheweights[:,0]
        for i in range(0,113):
            lheweight_ratio[i] = lheweights[:,i]/lheweights[:,0]
        
        variations = ['', '_up', '_down']
        for var in variations:
            #jets
            old_jet = getJets(events, jes_corrector, '')

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

            resolutions = ['']
            if var == '':
                resolutions = ['', [0.05,0.1]]
            for res in resolutions:                
                fatjet = getFatjets(events, jes_corrector, var, res)

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
                genmet_pt = ak.flatten(events.genmet_pt)
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


                taggers = ['']
                if (var == '') and (res == ''):
                    taggers = ['', '_0b_up', '_0b_down', '_1b_up', '_1b_down', '_2b_up', '_2b_down', '_1h_up', '_1h_down']
                for tagger in taggers:
                    w_0b = get_weight(self.effs[dataset]['0b'], fatjet.pt, fatjet.eta)
                    w_1b = get_weight(self.effs[dataset]['1b'], fatjet.pt, fatjet.eta)
                    w_2b = get_weight(self.effs[dataset]['2b'], fatjet.pt, fatjet.eta)
                    w_1h = get_weight(self.effs[dataset]['1h'], fatjet.pt, fatjet.eta)

                    if tagger == '_0b_up':
                        w_0b = get_weight(self.effs[dataset]['0b'], fatjet.pt, fatjet.eta)*1.1
                    if tagger == '_0b_down':
                        w_0b = get_weight(self.effs[dataset]['0b'], fatjet.pt, fatjet.eta)*0.9
                    if tagger == '_1b_up':
                        w_1b = get_weight(self.effs[dataset]['1b'], fatjet.pt, fatjet.eta)*1.1
                    if tagger == '_1b_down':
                        w_1b = get_weight(self.effs[dataset]['1b'], fatjet.pt, fatjet.eta)*0.9
                    if tagger == '_2b_up':
                        w_2b = get_weight(self.effs[dataset]['2b'], fatjet.pt, fatjet.eta)*1.1
                    if tagger == '_2b_down':
                        w_2b = get_weight(self.effs[dataset]['2b'], fatjet.pt, fatjet.eta)*0.9
                    if tagger == '_1h_up':
                        w_1h = get_weight(self.effs[dataset]['1h'], fatjet.pt, fatjet.eta)*1.1
                    if tagger == '_1h_down':
                        w_1h = get_weight(self.effs[dataset]['1h'], fatjet.pt, fatjet.eta)*0.9

                    w_all = w_0b * zerob + w_1b * oneb + w_2b * twob # + w_1h * onehiggs  # this should work
                    if not np.isnan(sum(sum(self.effs[dataset]['1h'].counts))):
                        w_all = w_all + w_1h * onehiggs

                    #selections
                    selection = PackedSelection()

                    if dataset == 'TT_Mtt1000toInf_TuneCUETP8M1_14TeV-powheg-pythia8_200PU':
                        selection.add('overlap', (ttbar.mass>1000))
                    elif dataset == 'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU':
                        selection.add('overlap', (ttbar.mass<1000))
                    elif dataset == 'WJetsToLNu_GenMET-100_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU':
                        selection.add('overlap', genmet_pt>100)
                    elif dataset == 'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU':
                        selection.add('overlap', genmet_pt<100)
                    else:
                        selection.add('overlap', met_pt>0) # NOTE: this is a dummy selection that should always evaluate to true

                    selection.add('single_lep', ((ak.num(ele_l, axis=1)>0) | (ak.num(muon_l, axis=1)>0) | (ak.num(tau_l, axis=1)>0)))
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
                    selection.add('MT>1200',   min_mt_AK8_MET>1200)


                    #weights

                    weight = Weights(len(events))
                    weight.add("NH>0", np.nan_to_num(1-ak.prod(1-w_all, axis=1), 0))

                    #outputs

                    baseline = [
                        'overlap',
                        'ele_veto',
                        'mu_veto',
                        'tau_veto',
                        'met',
                        'nAK8',
                    ]

                    tight = [
                        'overlap',
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
                        'MT>1200',
                    ]

                    single_lep = [
                        'overlap',
                        'single_lep',
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
                        'MT>1200',
                    ]

                    base_sel = n_minus_one(selection, baseline, [])
                    tight_sel = n_minus_one(selection, tight, [])
                    
                    if res == '':
                        tmp_sel = n_minus_one(selection, tight, ['on_H', 'MT>1200', 'MT>600'])
                        output["MT_vs_sdmass"+var+tagger].fill(
                            dataset=dataset,
                            mt=min_mt_AK8_MET[tmp_sel],
                            mass=ak.flatten(lead_fatjet.mass[tmp_sel]),
                            weight = weight.weight()[tmp_sel]
                        )
                    
                    if (tagger == '') and (res != ''):            
                            tmp_sel = n_minus_one(selection, tight, ['on_H', 'MT>1200', 'MT>600'])
                            output["MT_vs_sdmass_jmr"].fill(
                                dataset=dataset,
                                mt=min_mt_AK8_MET[tmp_sel],
                                mass=ak.flatten(lead_fatjet.mass[tmp_sel]),
                                weight = weight.weight()[tmp_sel]
                            )

                    if (var == '') and (tagger == '') and (res == ''):
                        output['cutflow'][dataset]['total'] += len(events)
                        output['cutflow'][dataset]['lepton_veto'] += len(events[n_minus_one(selection, baseline, ['met', 'nAK8'])])
                        output['cutflow'][dataset]['MET>300'] += len(events[n_minus_one(selection, baseline, ['nAK8'])])
                        output['cutflow'][dataset]['N_AK8>0'] += len(events[base_sel])
                        output['cutflow'][dataset]['N_AK4>1'] += len(events[n_minus_one(selection, tight, ['min_AK8_pt','dphi_AK8_MET>1','dphi_AK4_MET<3','dphi_AK4_MET>1','AK4_QCD_veto','AK8_QCD_veto','on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['min_AK8_pt'] += len(events[n_minus_one(selection, tight, ['dphi_AK8_MET>1','dphi_AK4_MET<3','dphi_AK4_MET>1','AK4_QCD_veto','AK8_QCD_veto','on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['dphi_AK8_MET>1'] += len(events[n_minus_one(selection, tight, ['dphi_AK4_MET<3','dphi_AK4_MET>1','AK4_QCD_veto','AK8_QCD_veto','on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['1<dphi_AK4_MET<3'] += len(events[n_minus_one(selection, tight, ['AK4_QCD_veto','AK8_QCD_veto','on_H','MT>600','MT>1200',])])
                        output['cutflow'][dataset]['AK4_QCD_veto'] += len(events[n_minus_one(selection, tight, ['AK8_QCD_veto','on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['AK8_QCD_veto'] += len(events[n_minus_one(selection, tight, ['on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['N_H>0'] += sum(weight.weight()[n_minus_one(selection, tight, ['on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['on_H'] += sum(weight.weight()[n_minus_one(selection, tight, ['MT>600','MT>1200'])])
                        output['cutflow'][dataset]['MT>600'] += sum(weight.weight()[n_minus_one(selection, tight, ['MT>1200'])])
                        output['cutflow'][dataset]['MT>1200'] += sum(weight.weight()[tight_sel])

                        output['cutflow'][dataset]['total_w2'] += len(events)
                        output['cutflow'][dataset]['lepton_veto_w2'] += len(events[n_minus_one(selection, baseline, ['met', 'nAK8'])])
                        output['cutflow'][dataset]['MET>300_w2'] += len(events[n_minus_one(selection, baseline, ['nAK8'])])
                        output['cutflow'][dataset]['N_AK8>0_w2'] += len(events[base_sel])
                        output['cutflow'][dataset]['N_AK4>1_w2'] += len(events[n_minus_one(selection, tight, ['min_AK8_pt', 'dphi_AK8_MET>1', 'dphi_AK4_MET<3','dphi_AK4_MET>1', 'AK4_QCD_veto', 'AK8_QCD_veto', 'on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['min_AK8_pt_w2'] += len(events[n_minus_one(selection, tight, ['dphi_AK8_MET>1', 'dphi_AK4_MET<3','dphi_AK4_MET>1', 'AK4_QCD_veto_w2', 'AK8_QCD_veto', 'on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['dphi_AK8_MET>1_w2'] += len(events[n_minus_one(selection, tight, ['dphi_AK4_MET<3','dphi_AK4_MET>1', 'AK4_QCD_veto_w2', 'AK8_QCD_veto', 'on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['1<dphi_AK4_MET<3_w2'] += len(events[n_minus_one(selection, tight, ['AK4_QCD_veto', 'AK8_QCD_veto', 'on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['AK4_QCD_veto_w2'] += len(events[n_minus_one(selection, tight, ['AK8_QCD_veto', 'on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['AK8_QCD_veto_w2'] += len(events[n_minus_one(selection, tight, ['on_H','MT>600','MT>1200'])])
                        output['cutflow'][dataset]['N_H>0_w2'] += sum(weight.weight()[n_minus_one(selection, tight, ['on_H','MT>600', 'MT>1200'])]**2)
                        output['cutflow'][dataset]['on_H_w2'] += sum(weight.weight()[n_minus_one(selection, tight, ['MT>600','MT>1200'])]**2)
                        output['cutflow'][dataset]['MT>600_w2'] += sum(weight.weight()[n_minus_one(selection, tight, ['MT>1200'])]**2)
                        output['cutflow'][dataset]['MT>1200_w2'] += sum(weight.weight()[tight_sel]**2)

                        tmp_base_sel = n_minus_one(selection, baseline, ['met'])
                        tmp_sel = n_minus_one(selection, tight, ['met', 'MT>1200'])
                        output["met_pt"].fill(
                            dataset=dataset,
                            pt=met_pt[tmp_sel],
                            weight = weight.weight()[tmp_sel]
                        )

                        tmp_sel = n_minus_one(selection, tight, ['met', 'MT>1200'])
                        output["genmet_pt"].fill(
                            dataset=dataset,
                            pt=genmet_pt[tmp_sel],
                            weight = weight.weight()[tmp_sel]
                        )

                        tmp_sel = n_minus_one(selection, baseline, ['met', 'nAK8', 'overlap'])
                        output["genmet_pt_inclusive"].fill(
                            dataset=dataset,
                            pt=genmet_pt,
                            weight = weight.weight()
                        )

                        if dataset.count('TT_'):
                            tmp_sel = n_minus_one(selection, tight, ['met', 'MT>1200'])
                            output["Mtt"].fill(
                                dataset=dataset,
                                pt=ttbar.mass[tmp_sel],
                                weight = weight.weight()[tmp_sel]
                            )
                            
                            tmp_sel = n_minus_one(selection, baseline, ['overlap'])
                            output["Mtt_inclusive"].fill(
                                dataset=dataset,
                                pt=ttbar.mass[tmp_sel],
                                weight = weight.weight()[tmp_sel]
                            )

                        tmp_sel = n_minus_one(selection, tight, ['dphi_AK8_MET>1', 'MT>1200'])
                        output["dphi_AK8_MET"].fill(
                            dataset=dataset,
                            phi=min_dphi_AK8_MET[tmp_sel],
                            weight = weight.weight()[tmp_sel]
                        )

                        tmp_sel = n_minus_one(selection, tight, ['dphi_AK4_MET<3', 'dphi_AK4_MET>1', 'MT>1200'])
                        output["dphi_AK4_MET"].fill(
                            dataset=dataset,
                            phi=min_dphi_AK4_MET[tmp_sel],
                            weight = weight.weight()[tmp_sel]
                        )

                        tmp_sel = n_minus_one(selection, tight, ['AK4_QCD_veto', 'MT>1200'])
                        output["AK4_QCD_veto"].fill(
                            dataset=dataset,
                            phi=ak.flatten(dphi_dijet[tmp_sel & (ak.num(jet)>1)][:,0:1]),
                            weight = weight.weight()[tmp_sel & (ak.num(jet)>1)]
                        )

                        tmp_sel = n_minus_one(selection, tight, ['AK8_QCD_veto', 'MT>1200'])
                        output["AK8_QCD_veto"].fill(
                            dataset=dataset,
                            phi=ak.flatten(dphi_difatjet[tmp_sel & (ak.num(fatjet)>1)][:,0:1]),
                            weight = weight.weight()[tmp_sel & (ak.num(fatjet)>1)]
                        )

                        tmp_sel = n_minus_one(selection, tight, ['on_H', 'MT>1200'])
                        output["AK8_sdmass"].fill(
                            dataset=dataset,
                            mass=ak.flatten(lead_fatjet.mass[tmp_sel]),
                            weight = weight.weight()[tmp_sel]
                        )

                        output["MT_vs_sdmass_BL"].fill(
                            dataset=dataset,
                            mt=min_mt_AK8_MET[base_sel],
                            mass=ak.flatten(lead_fatjet.mass[base_sel]),
                            weight = weight.weight()[base_sel]
                        )
                            
                        tmp_sel = n_minus_one(selection, tight, ['nAK4', 'MT>1200'])
                        output["n_AK4"].fill(
                            dataset=dataset,
                            multiplicity=ak.num(jet[tmp_sel]),
                            weight = weight.weight()[tmp_sel]
                        )

                        tmp_sel = n_minus_one(selection, tight, ['min_AK8_pt', 'MT>1200'])
                        output["min_AK8_pt"].fill(
                            dataset=dataset,
                            pt=ak.min(fatjet.pt[tmp_sel], axis=1),
                            weight = weight.weight()[tmp_sel]
                        )

                        #output['NH_weight_BL'].fill(
                        #    dataset=dataset,
                        #    multiplicity = np.zeros_like(ak.num(fatjet[base_sel], axis=1)),
                        #    weight = np.nan_to_num(ak.prod(1-w_all[base_sel], axis=1), 0),
                        #)
                        #output['NH_weight_BL'].fill(
                            # This already includes the overflow, so everything >0.
                            # In the end this is all we care about, we don't differenciate N_H=2 from N_H=1
                        #    dataset=dataset,
                        #    multiplicity = np.ones_like(ak.num(fatjet[base_sel], axis=1)),
                        #    weight = np.nan_to_num(1-ak.prod(1-w_all[base_sel], axis=1), 0),
                        #)

                        tmp_sel = n_minus_one(selection, tight, ['on_H', 'MT>600', 'MT>1200'])
                        output['MT'].fill(
                            dataset=dataset,
                            mt = min_mt_AK8_MET[tmp_sel],
                            weight = weight.weight()[tmp_sel],
                        )
                       
                        tmp_sel = n_minus_one(selection, single_lep, ['on_H', 'MT>600', 'MT>1200'])
                        output['MT_single_lep'].fill(
                            dataset=dataset,
                            mt = min_mt_AK8_MET[tmp_sel],
                            weight = weight.weight()[tmp_sel],
                        )

                        tmp_sel = n_minus_one(selection, tight, ['MT>1200'])
                        output['NH_weight'].fill(
                            dataset=dataset,
                            multiplicity = np.zeros_like(ak.num(fatjet[tmp_sel], axis=1)),
                            weight = np.nan_to_num(ak.prod(1-w_all[tmp_sel], axis=1), 0),
                        )
                        output['NH_weight'].fill(
                            # This already includes the overflow, so everything >0.
                            # In the end this is all we care about, we don't differenciate N_H=2 from N_H=1
                            dataset=dataset,
                            multiplicity = np.ones_like(ak.num(fatjet[tmp_sel], axis=1)),
                            weight = np.nan_to_num(1-ak.prod(1-w_all[tmp_sel], axis=1), 0),
                        )

                        #output['b_DeltaR_vs_H_pt_BL'].fill(
                        #    dataset=dataset,
                        #    pt = pad_and_flatten(higgs.pt[(ak.num(bquark)==2)]),
                        #    phi = ak.flatten(b_DeltaR[(ak.num(bquark)==2)]),
                        #    #weight = weight.weight()[base_sel&(ak.num(bquark)==2)]
                        #)
                        output['b_DeltaR_vs_H_pt'].fill(
                            dataset=dataset,
                            pt = pad_and_flatten(higgs.pt[(ak.num(bquark)==2)]),
                            phi = ak.flatten(b_DeltaR[(ak.num(bquark)==2)]),
                            #weight = weight.weight()[tmp_sel&(ak.num(bquark)==2)]
                        )

                        for i in lheweight_ratio.keys():
                            weight = Weights(len(events))
                            weight.add("NH>0", np.nan_to_num(1-ak.prod(1-w_all, axis=1), 0))
                            weight.add("LHE_weights", lheweight_ratio[i])

                            tmp_sel = n_minus_one(selection, tight, ['on_H', 'MT>1200', 'MT>600'])

                            output["MT_vs_sdmass_LHE"].fill(
                                dataset=dataset,
                                variation=str(i),
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
    argParser.add_argument('--run', action='store', default='all', choices=['all','Z', 'W', 'TT', 'QCD', 'other', 'signal'])
    argParser.add_argument('--variation', action='store', default='central', choices=['central', 'jup', 'jdown'])
    argParser.add_argument('--workers', action='store', type=int, default=10)
    argParser.add_argument('--outdir', action='store', default='./outputs/')
    args = argParser.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

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
            exe_args = {"schema": BaseSchema, "workers": args.workers}

        fileset_all = {
            'TT': {
                'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU': samples['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']['skim'],
                'TT_Mtt1000toInf_TuneCUETP8M1_14TeV-powheg-pythia8_200PU': samples['TT_Mtt1000toInf_TuneCUETP8M1_14TeV-powheg-pythia8_200PU']['skim'],
                'ST_tch_14TeV_top_incl-powheg-pythia8-madspin_200PU': samples['ST_tch_14TeV_top_incl-powheg-pythia8-madspin_200PU']['skim'],
                'ST_tch_14TeV_antitop_incl-powheg-pythia8-madspin_200PU': samples['ST_tch_14TeV_antitop_incl-powheg-pythia8-madspin_200PU']['skim'],
                'ST_tW_top_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU': samples['ST_tW_top_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU']['skim'],
                'ST_tW_antitop_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU': samples['ST_tW_antitop_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU']['skim'],
            },
            'Z': {
                'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU']['skim'],
                'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU']['skim'],
                'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU']['skim'],
                'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU']['skim'],
                'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU']['skim'],
                'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU']['skim'],
                'ZJetsToNuNu_HT2500toInf_HLLHC': samples['ZJetsToNuNu_HT2500toInf_HLLHC']['skim'],
            },
            'W': {
                'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
                'WJetsToLNu_GenMET-100_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['WJetsToLNu_GenMET-100_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            },
            'QCD': {
                'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
                'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
                'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
                'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
                'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
                'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
                'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],

            },
            'other': {
                'ZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_200PU': samples['ZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_200PU']['skim'],
                'WminusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU': samples['WminusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU']['skim'],
                'WplusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU': samples['WplusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU']['skim'],
            },
            'signal': {
                #m_a=150
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']['ntuples'],
                #m_a=750
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250']['ntuples'],
                #m_a=250
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250']['ntuples'],
                #m_a=500
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000']['ntuples'],
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000']['ntuples'],
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250']['ntuples'],
            },
        }

        if args.run == 'all':
            fileset={}
            for name in fileset_all.keys():
                fileset.update(fileset_all[name])
        else:
            fileset = fileset_all[args.run]
            
        meta_accumulator = {}
        for sample in fileset:
            if sample not in meta_accumulator:
                meta_accumulator.update({sample: processor.defaultdict_accumulator(int)})
            for f in fileset[sample]:
                meta_accumulator.update({f: processor.defaultdict_accumulator(int)})

        run2_to_delphes = {
            'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU': 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',
            'TT_Mtt1000toInf_TuneCUETP8M1_14TeV-powheg-pythia8_200PU': 'TT_Mtt-1000toInf_TuneCP5_13TeV-powheg-pythia8',
            'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT2500toInf_HLLHC': 'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
            'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8',
            'WJetsToLNu_GenMET-100_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8',
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8', 
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT200to300_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT300to500_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT500to700_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT700to1000_TuneCP5_13TeV-madgraph-pythia8',
            # Some of the rare samples don't exist in UL18 (which is bad just by itself)
            # but what can we do.
            # W+jets should be as close as it gets for the extra radiation for diboson
            # signal sample for the SM Higgs samples
            'ZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_200PU': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            'WminusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            'WplusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            'VVTo2L2Nu_14TeV_amcatnloFXFX_madspin_pythia8_200PU': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            'ST_tch_14TeV_top_incl-powheg-pythia8-madspin_200PU': 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',  #'ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',
            'ST_tch_14TeV_antitop_incl-powheg-pythia8-madspin_200PU': 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',  #'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',
            'ST_tW_top_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU': 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',  #'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
            'ST_tW_antitop_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU': 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',  #'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
           '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',    
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',    
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250': 'ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8',
        }
        
        effs = {}
        for s in fileset.keys() :
            effs[s] = {}
            for b in ['0b', '1b', '2b', '1h']:
                effs[s][b] = Hist2D.from_json(os.path.expandvars("../data/htag/eff_%s_%s.json"%(run2_to_delphes[s],b)))
        
        output_flat = processor.run_uproot_job(
            fileset,
            treename='mytree',
            processor_instance = FlatProcessor(accumulator=meta_accumulator, effs=effs),
            executor = exe,
            executor_args = exe_args,
            chunksize=100000,
            maxchunks=None,
        )

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(args.outdir, f"output_{args.run}_run{timestamp}.coffea")
        util.save(output_flat, outfile)

        signal = [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_150_MH2_1000_MHC_1000',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_750_MH2_1250_MHC_1250',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_750_MH2_1600_MHC_1600',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_750_MH2_1900_MHC_1900',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_750_MH2_2250_MHC_2250',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_750_MH4_250_MH2_750_MHC_750',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_250_MH2_1000_MHC_1000',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_250_MH2_1250_MHC_1250',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_250_MH2_1600_MHC_1600',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_250_MH2_1750_MHC_1750',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_250_MH2_1900_MHC_1900',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_250_MH2_2000_MHC_2000',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_250_MH2_2250_MHC_2250',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000',
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1000_MH4_500_MH2_1000_MHC_1000',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1250_MH4_500_MH2_1250_MHC_1250',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1600_MH4_500_MH2_1600_MHC_1600',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_500_MH2_1750_MHC_1750',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1900_MH4_500_MH2_1900_MHC_1900',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_500_MH2_2000_MHC_2000',
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2250_MH4_500_MH2_2250_MHC_2250',
        ]
        
        nevents = {}
        
        for sample in fileset:
            nevents[sample] = 0
            if sample in signal:
                nevents[sample] = samples[sample]['nevents']
            else:    
                for file in fileset[sample]:
                    with uproot.open(file+':nevents') as counts:
                        nevents[file] = counts.counts()[0] 
                    nevents[sample] += nevents[file]
    
        meta = {}

        for sample in fileset:
            meta[sample] = output_flat[sample]
            meta[sample]['xsec'] = samples[sample]['xsec']
            meta[sample]['nevents'] = nevents[sample]

        scaled_output = {}
        
        for key in output_flat.keys():
            #if type(output_flat[key]) is not type(output_flat['cutflow']):
            if type(output_flat[key]) is type(output_flat['b_DeltaR_vs_H_pt']):
                scaled_output[key] = scale_and_merge_histos(output_flat[key], meta, fileset, lumi=3000)

        outfile = os.path.join(args.outdir, f"output_{args.run}_scaled_run{timestamp}.coffea")
        util.save(scaled_output, outfile)

        raise NotImplementedError

        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

        labels ={
            ('QCD_bEnriched_HT',): r'$QCD\ b-enriched$',
            ('ZJetsToNuNu_HT',): r'$ZJets\to\nu\nu$',
            ('WJetsToLNu_Njet',): r'$WJets\to L\nu$',
            ('TT',): r'$t\bar{t}$',
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',): '2HDMa_1500_750_10',
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',): '2HDMa_1750_750_10',
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000',): '2HDMa_2000_750_10',
        }

        colors ={
            ('QCD_bEnriched_HT',): '#D23FFE',
            ('ZJetsToNuNu_HT',): '#6BFE3F',
            ('WJetsToLNu_Njet',): '#FED23F',
            ('TT',): '#FE3F6B',
        }
        
        signals = [
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',),
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',),
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',), 
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000',),
            
        ]

        plot_dir = '/home/users/$USER/public_html/HbbMET/plots/Jan24/'
        
        #makePlot2(scaled_output, 'met_pt', 'pt', met_bins, r'$MET_{pt}\ (GeV)$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'met_pt_BL', 'pt', met_bins, r'$MET_{pt}\ (GeV)$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'dphi_AK4_MET', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'dphi_AK4_MET_BL', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'dphi_AK8_MET', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'dphi_AK8_MET_BL', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'AK4_QCD_veto', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'AK4_QCD_veto_BL', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'AK8_QCD_veto', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'AK8_QCD_veto_BL', 'phi', phi_bins2, r'$\Delta\varphi$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'AK8_sdmass', 'mass', mass_bins, r'$softdrop\ mass\ (GeV)$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'AK8_sdmass_BL', 'mass', mass_bins, r'$softdrop\ mass\ (GeV)$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'n_AK4', 'multiplicity', N_bins2, r'$N_{AK4}$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'n_AK4_BL', 'multiplicity', N_bins2, r'$N_{AK4}$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'min_AK8_pt', 'pt', pt_bins, r'$p_{T}\ (GeV)$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'min_AK8_pt_BL', 'pt', pt_bins, r'$p_{T}\ (GeV)$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'NH_weight', 'multiplicity', N_H_bins, r'$N_{H-tagged}$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'NH_weight_BL', 'multiplicity', N_H_bins, r'$N_{H-tagged}$', labels, colors, signals, plot_dir)
        #makePlot2(scaled_output, 'MT', 'mt', mt_bins, r'$M_{T}$', labels, colors, signals, plot_dir)
