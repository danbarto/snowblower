#!/usr/bin/env python3
import awkward as ak
import uproot
import numpy as np
import glob
import os
from coffea.nanoevents import NanoEventsFactory, DelphesSchema, BaseSchema
from coffea.analysis_tools import Weights
from coffea import hist, processor
# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

from functools import partial

from plots.helpers import makePlot2, scale_and_merge_histos
from tools.helpers import choose, delta_phi_alt_paf, get_four_vec_fromPtEtaPhiM, get_weight, match, match_count, mt, cross

import warnings
warnings.filterwarnings("ignore")

import shutil

class FlatProcessor(processor.ProcessorABC):
    def __init__(self, accumulator={}, effs={}):
        self._accumulator = processor.dict_accumulator({
            "met_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$MET_{pt}$ [GeV]", 100, 0, 1000),
            ),
            "ht": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$H_{T}$ [GeV]", 60, 0, 3000),
            ),
            "min_mt_fj_met": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mt", "$M_{T}$ [GeV]", 60, 0, 3000),
            ),
            "lead_fatjet_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
            ),
            "lead_fatjet_eta": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("eta", "$\eta$", 33, -4, 4),
            ),
            "lead_fatjet_phi": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 33, -4, 4),
            ),
            "lead_fatjet_sdmass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
            "lead_fatjet_tau1": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_1$", 10, 0, 0.7),
            ),
            "lead_fatjet_tau2": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_2$", 10, 0, 0.5),
            ),
            "lead_fatjet_tau3": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_3$", 10, 0, 0.4),
            ),
            "lead_fatjet_tau4": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_4$", 10, 0, 0.3),
            ),
            "lead_fatjet_tau21": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_{2}/\tau{1}$", 100, 0, 2.0)
            ),
            "nfatjet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{fatjet}$", 5, -0.5, 4.5),
            ),
            "lead_extrajet_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
            ),
            "lead_extrajet_eta": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("eta", "$\eta$", 33, -4, 4),
            ),
            "lead_extrajet_phi": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 33, -4, 4),
            ),
            "lead_extrajet_mass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
            "nextrajet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "lead_extrabtag_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
            ),
            "lead_extrabtag_eta": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("eta", "$\eta$", 33, -4, 4),
            ),
            "lead_extrabtag_phi": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 33, -4, 4),
            ),
            "lead_extrabtag_mass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
            "nextrabtag": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "njet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "n_b_in_AK8": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "n_H_in_AK8": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "n_H_gen": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{H}$", 7, -0.5, 6.5),
            ),
            "dphiDiFatJet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            "dphileadextrajet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            "min_dphiFatJetMet4": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            "min_dphiJetMetAll": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            'cutflow': processor.defaultdict_accumulator(
                # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int)
            ),
            "lead_fatjet_pt_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
            ),
            "lead_fatjet_eta_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("eta", "$\eta$", 33, -4, 4),
            ),
            "lead_fatjet_phi_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 33, -4, 4),
            ),
            "lead_fatjet_sdmass_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
            "lead_fatjet_tau1_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_1$", 10, 0, 0.7),
            ),
            "lead_fatjet_tau2_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_2$", 10, 0, 0.5),
            ),
            "lead_fatjet_tau3_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_3$", 10, 0, 0.4),
            ),
            "lead_fatjet_tau4_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_4$", 10, 0, 0.3),
            ),
            "lead_fatjet_tau21_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_{2}/\tau{1}$", 100, 0, 2.0)
            ),
            "nfatjet_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{fatjet}$", 5, -0.5, 4.5),
            ),
            "NH_weight": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{fatjet}$", 5, -0.5, 4.5),
            ),
            "met_pt_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$MET_{pt}$ [GeV]", 100, 0, 1000),
            ),
            "ht_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$H_{T}$ [GeV]", 60, 0, 3000),
            ),
            "min_mt_fj_met_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mt", "$M_{T}$ [GeV]", 60, 0, 3000),
            ),
            "lead_extrajet_pt_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
            ),
            "lead_extrajet_eta_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("eta", "$\eta$", 33, -4, 4),
            ),
            "lead_extrajet_phi_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 33, -4, 4),
            ),
            "lead_extrajet_mass_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
            "nextrajet_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "lead_extrabtag_pt_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
            ),
            "lead_extrabtag_eta_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("eta", "$\eta$", 33, -4, 4),
            ),
            "lead_extrabtag_phi_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 33, -4, 4),
            ),
            "lead_extrabtag_mass_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
            "nextrabtag_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "njet_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "dphileadextrajet_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            "dphiDiFatJet_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            "min_dphiFatJetMet4_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            "min_dphiJetMetAll_tagged": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
        })
        
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
            
        #jets
        
        jet = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.jetpuppi_pt,
            eta = events.jetpuppi_eta,
            phi = events.jetpuppi_phi,
            M = events.jetpuppi_mass,
            copy = False,
        )
        jet['id'] = events.jetpuppi_idpass
        jet['btag'] = events.jetpuppi_btag
        
        
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
        
        fatjet = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.fatjet_pt,
            eta = events.fatjet_eta,
            phi = events.fatjet_phi,
            M = events.fatjet_msoftdrop,        #Using softdrop from now on
            copy = False,
        )
        
        #fatjet['m'] = events.fatjet_mass
        fatjet['tau1'] = events.fatjet_tau1
        fatjet['tau2'] = events.fatjet_tau2
        fatjet['tau3'] = events.fatjet_tau3
        fatjet['tau4'] = events.fatjet_tau4
        
        fatjet = fatjet[np.abs(fatjet.eta) < 3] #eta within tracker range
        fatjet = fatjet[ak.argsort(fatjet.pt, ascending=False)]
        
        tau21 = np.divide(fatjet.tau2, fatjet.tau1)
        
        fatjet_on_h = fatjet[np.abs(fatjet.mass-125)<25]
        on_h = (ak.num(fatjet_on_h) > 0)
        
        lead_fatjet = fatjet[:,0:1]
        #sublead_fatjet = fatjet[:,1:2] 
        
        lead_fatjets = fatjet[:,0:2]
        difatjet = choose(lead_fatjets, 2)
        dphiDiFatJet = np.arccos(np.cos(difatjet['0'].phi-difatjet['1'].phi))
        
        extrajet  = jet[~match(jet, fatjet, deltaRCut=0.8)] # remove AK4 jets that overlap with AK8 jets
        extrabtag = extrajet[extrajet.btag>0] #loose wp for now]
        #extrajet = extrajet[ak.argsort(extrajet.pt, ascending=False)]
        #extrabtag = extrabtag[ak.argsort(extrabtag.pt, ascending=False)]
        lead_extrajet = extrajet[:,0:1]
        lead_extrabtag = extrabtag[:,0:1]
        dphileadextrajet = delta_phi_alt_paf(lead_fatjet, lead_extrajet)
        di_AK8_AK4 = cross(extrajet, fatjet)
        dphi_AK8_AK4 = np.arccos(np.cos(di_AK8_AK4['0'].phi-di_AK8_AK4['1'].phi))
        min_dphi_AK8_AK4 = ak.min(dphi_AK8_AK4, axis=1)
        #dphi_AK8_AK4 = delta_phi_alt(extrajet[:,:1], fatjet[:,:1])  # we don't care about the non-leading ones
        
                
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
        
        w_all = w_0b * zerob + w_1b * oneb + w_2b * twob # + w_1h * onehiggs  # this should work
        if not np.isnan(sum(sum(self.effs[dataset]['1h'].counts))):
#        np.isnan(sum(ak.flatten(w_1h * onehiggs))):
            w_all = w_all + w_1h * onehiggs
        
        #MET
        
        met_pt = ak.flatten(events.metpuppi_pt)
        met_phi = ak.flatten(events.metpuppi_phi)
        
        delta_phi_ml = np.arccos(np.cos(lead_fatjet.phi - met_phi))
        mt_fj_met = mt(fatjet_on_h.pt, fatjet_on_h.phi, met_pt, met_phi)
        min_mt_fj_met = ak.min(mt_fj_met, axis=1, mask_identity=False)
        min_dphiFatJetMet4 = ak.to_numpy(ak.min(np.arccos(np.cos(fatjet[:,:4].phi-met_phi)), axis=1))
        min_dphiJetMetAll = ak.to_numpy(ak.min(np.arccos(np.cos(jet.phi-met_phi)), axis=1))

        #selections
        ele_sel = (ak.num(ele_l)==0)
        mu_sel = ((ak.num(muon_l)==0) & ele_sel)
        tau_sel = ((ak.num(tau_l)==0) & mu_sel)
        met_sel = (tau_sel & (met_pt>200))
        baseline = ((ak.num(fatjet, axis=1)>0) & met_sel)
        min_dphiFatJetMet4_sel = ak.to_numpy((min_dphiFatJetMet4 > 0.5) & baseline)
        dphiDiFatJet_sel = ak.to_numpy((ak.all(dphiDiFatJet < 2.5, axis=1) & min_dphiFatJetMet4_sel))
        on_h_sel = ak.to_numpy(on_h & dphiDiFatJet_sel)
        min_mt = ak.to_numpy((min_mt_fj_met > 500) & on_h_sel)
        
        #weights
        
        weight = Weights(len(events))
        #weight.add("weight", events.genweight)
        weight2 = Weights(len(events))
        weight2.add("tagged", np.nan_to_num(1-ak.prod(1-w_all, axis=1), 0))

        #output
        output['cutflow'][dataset]['total'] += sum(weight.weight())
        output['cutflow'][dataset]['n_ele==0'] += sum(weight.weight()[ele_sel])
        output['cutflow'][dataset]['n_mu==0'] += sum(weight.weight()[mu_sel])
        output['cutflow'][dataset]['n_tau==0'] += sum(weight.weight()[tau_sel])
        output['cutflow'][dataset]['met>200'] += sum(weight.weight()[met_sel])
        output['cutflow'][dataset]['n_ak8>=1'] += sum(weight.weight()[baseline])
        output['cutflow'][dataset]['min_dphiFatJetMet4>0.5'] += sum(weight.weight()[min_dphiFatJetMet4_sel])
        output['cutflow'][dataset]['dphiDiFatJet<2.5'] += sum(weight.weight()[dphiDiFatJet_sel])
        output['cutflow'][dataset]['on-H'] += sum(weight.weight()[on_h_sel])
        output['cutflow'][dataset]['minmth>200'] += sum(weight.weight()[min_mt])
        output['cutflow'][dataset]['N_H>0'] += sum(weight2.weight()[min_mt])
        
        output["met_pt"].fill(
            dataset=dataset,
            pt=met_pt[baseline],
            #weight = weight.weight()[baseline]
        )
        output["ht"].fill(
            dataset=dataset,
            pt = ak.sum(events.jetpuppi_pt[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["min_mt_fj_met"].fill(
            dataset=dataset,
            mt = ak.min(mt_fj_met[(baseline & on_h)], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_pt"].fill(
            dataset=dataset,
            pt=ak.to_numpy(ak.flatten(lead_fatjet.pt[baseline], axis=1)),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_eta"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_fatjet.eta[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_phi"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_fatjet.phi[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_sdmass"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_fatjet.mass[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_tau1"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau1[baseline], axis=1),
            #weight = weight.weight()[min_mt]
        )
        output["lead_fatjet_tau2"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau2[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_tau3"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau3[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_tau4"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau4[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_fatjet_tau21"].fill(
            dataset=dataset,
            tau = ak.flatten(tau21[:,0:1][baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["nfatjet"].fill(
            dataset=dataset,
            multiplicity=ak.num(fatjet[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["dphiDiFatJet"].fill(
            dataset=dataset,
            phi = ak.flatten(dphiDiFatJet[baseline]),
            #weight = weight.weight()[baseline]
        )
        output["dphileadextrajet"].fill(
            dataset=dataset,
            phi = ak.to_numpy(min_dphi_AK8_AK4[baseline]),
            #weight = weight.weight()[baseline]
        )
        output["min_dphiFatJetMet4"].fill(
            dataset=dataset,
            phi = min_dphiFatJetMet4[baseline],
            #weight = weight.weight()[baseline]
        )
        output["min_dphiJetMetAll"].fill(
            dataset=dataset,
            phi = min_dphiJetMetAll[baseline],
            #weight = weight.weight()[baseline]
        )
        output["lead_extrajet_pt"].fill(
            dataset=dataset,
            pt=ak.flatten(lead_extrajet.pt[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_extrajet_eta"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_extrajet.eta[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_extrajet_phi"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_extrajet.phi[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_extrajet_mass"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_extrajet.mass[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["nextrajet"].fill(
            dataset=dataset,
            multiplicity=ak.num(extrajet[baseline]),
            #weight = weight.weight()[baseline]
        )
        output["lead_extrabtag_pt"].fill(
            dataset=dataset,
            pt=ak.flatten(lead_extrabtag.pt[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_extrabtag_eta"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_extrabtag.eta[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_extrabtag_phi"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_extrabtag.phi[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["lead_extrabtag_mass"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_extrabtag.mass[baseline], axis=1),
            #weight = weight.weight()[baseline]
        )
        output["nextrabtag"].fill(
            dataset=dataset,
            multiplicity=ak.num(extrabtag[baseline]),
            #weight = weight.weight()[baseline]
        )
        output["njet"].fill(
            dataset=dataset,
            multiplicity=ak.num(jet[baseline]),
            #weight = weight.weight()[baseline]
        )
        output["n_b_in_AK8"].fill(
            dataset=dataset,
            multiplicity=ak.flatten(nb_in_fat),
            #weight = weight.weight()[baseline]
        )
        output["n_H_in_AK8"].fill(
            dataset=dataset,
            multiplicity=ak.flatten(nhiggs_in_fat),
            #weight = weight.weight()[baseline]
        )
        output["met_pt_tagged"].fill(
            dataset=dataset,
            pt=met_pt[baseline],
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["ht_tagged"].fill(
            dataset=dataset,
            pt = ak.sum(events.jetpuppi_pt[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["min_mt_fj_met_tagged"].fill(
            dataset=dataset,
            mt = ak.min(mt_fj_met[(baseline & on_h)], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline & on_h)], axis=1), 0),
        ) 
        output["lead_fatjet_pt_tagged"].fill(
            dataset=dataset,
            pt=ak.to_numpy(ak.flatten(lead_fatjet.pt[baseline])),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_eta_tagged"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_fatjet.eta[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_phi_tagged"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_fatjet.phi[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_sdmass_tagged"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_fatjet.mass[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_tau1_tagged"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau1[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_tau2_tagged"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau2[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_tau3_tagged"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau3[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_tau4_tagged"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau4[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["lead_fatjet_tau21_tagged"].fill(
            dataset=dataset,
            tau = ak.flatten(tau21[:,0:1][baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline][:,:1], axis=1), 0),
        )
        output["nfatjet_tagged"].fill(
            dataset=dataset,
            multiplicity=ak.num(fatjet[baseline], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output['NH_weight'].fill(
            dataset=dataset,
            multiplicity = np.zeros_like(ak.num(fatjet[baseline], axis=1)),
            weight = np.nan_to_num(ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output['NH_weight'].fill(
            # This already includes the overflow, so everything >0.
            # In the end this is all we care about, we don't differenciate N_H=2 from N_H=1
            dataset=dataset,
            multiplicity = np.ones_like(ak.num(fatjet[baseline], axis=1)),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["lead_extrajet_pt_tagged"].fill(
            dataset=dataset,
            pt=ak.flatten(lead_extrajet.pt[(baseline&(ak.num(extrajet)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrajet)>0))], axis=1), 0),
        )
        output["lead_extrajet_eta_tagged"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_extrajet.eta[(baseline&(ak.num(extrajet)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrajet)>0))], axis=1), 0),
        )
        output["lead_extrajet_phi_tagged"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_extrajet.phi[(baseline&(ak.num(extrajet)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrajet)>0))], axis=1), 0),
        )
        output["lead_extrajet_mass_tagged"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_extrajet.mass[(baseline&(ak.num(extrajet)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrajet)>0))], axis=1), 0),
        )
        output["nextrajet_tagged"].fill(
            dataset=dataset,
            multiplicity=ak.num(extrajet[baseline]),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["lead_extrabtag_pt_tagged"].fill(
            dataset=dataset,
            pt=ak.flatten(lead_extrabtag.pt[(baseline&(ak.num(extrabtag)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrabtag)>0))], axis=1), 0),
        )
        output["lead_extrabtag_eta_tagged"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_extrabtag.eta[(baseline&(ak.num(extrabtag)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrabtag)>0))], axis=1), 0),
        )
        output["lead_extrabtag_phi_tagged"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_extrabtag.phi[(baseline&(ak.num(extrabtag)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrabtag)>0))], axis=1), 0),
        )
        output["lead_extrabtag_mass_tagged"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_extrabtag.mass[(baseline&(ak.num(extrabtag)>0))], axis=1),
            weight = np.nan_to_num(1-ak.prod(1-w_all[(baseline&(ak.num(extrabtag)>0))], axis=1), 0),
        )
        output["nextrabtag_tagged"].fill(
            dataset=dataset,
            multiplicity=ak.num(extrabtag[baseline]),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["njet_tagged"].fill(
            dataset=dataset,
            multiplicity=ak.num(jet[baseline]),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["dphiDiFatJet_tagged"].fill(
            dataset=dataset,
            phi = ak.flatten(dphiDiFatJet[baseline&(ak.num(fatjet)>1)]),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline&(ak.num(fatjet)>1)], axis=1), 0),
        )
        output["dphileadextrajet_tagged"].fill(
            dataset=dataset,
            phi = ak.to_numpy(dphileadextrajet[baseline]),
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["min_dphiFatJetMet4_tagged"].fill(
            dataset=dataset,
            phi = min_dphiFatJetMet4[baseline],
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
        )
        output["min_dphiJetMetAll_tagged"].fill(
            dataset=dataset,
            phi = min_dphiJetMetAll[baseline],
            weight = np.nan_to_num(1-ak.prod(1-w_all[baseline], axis=1), 0),
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
            #'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU']['skim'],
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU']['skim'],
            #'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500']['ntuples'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750']['ntuples'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000']['ntuples'],
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500']['ntuples'],
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750']['ntuples'],
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000': samples['2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000']['ntuples'],
        }

        meta_accumulator = {}
        for sample in fileset:
            if sample not in meta_accumulator:
                meta_accumulator.update({sample: processor.defaultdict_accumulator(int)})
            for f in fileset[sample]:
                meta_accumulator.update({f: processor.defaultdict_accumulator(int)})

        run2_to_delphes = {
            'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU': 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': 'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
            #'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': ,
            'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'W1JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8',
            'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'W2JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8',
            'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8',
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8', 
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT200to300_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT300to500_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT500to700_TuneCP5_13TeV-madgraph-pythia8',
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': 'QCD_bEnriched_HT700to1000_TuneCP5_13TeV-madgraph-pythia8',
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
        
        meta = {}

        for sample in fileset:
            meta[sample] = output_flat[sample]
            meta[sample]['xsec'] = samples[sample]['xsec']
            meta[sample]['nevents'] = samples[sample]['nevents']

        scaled_output = {}
        
        for key in output_flat.keys():
            if type(output_flat[key]) is not type(output_flat['cutflow']):
                scaled_output[key] = scale_and_merge_histos(output_flat[key], meta, fileset, lumi=3000)
        
        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

        N_bins = hist.Bin('multiplicity', r'$N$', 5, -0.5, 4.5)
        N_bins2 = hist.Bin('multiplicity', r'$N$', 7, -0.5, 6.5)
        mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 40, 0, 400)
        mass_bins2 = hist.Bin('mass', r'$M\ (GeV)$', 6, 0, 60)
        ht_bins = hist.Bin('pt', r'$H_{T}\ (GeV)$', 60, 0, 3000)
        mt_bins = hist.Bin('mt', r'$M_{T}\ (GeV)$', 40, 0, 2000)
        pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 80, 200, 1000)
        pt_bins2 = hist.Bin('pt', r'$p_{T}\ (GeV)$', 50, 0, 500)
        met_bins = hist.Bin('pt', r'$MET_{pt}\ (GeV)$', 100, 0, 1000)
        eta_bins = hist.Bin("eta", "$\eta$", 33, -4, 4)
        phi_bins = hist.Bin("phi", "$\phi$", 33, -4, 4)
        phi_bins2 = hist.Bin("phi", "$\phi$", 16, 0, 4)
        deltaR_bins = hist.Bin("deltaR", "$\DeltaR$", 10, 0, 1)
        tau1_bins = hist.Bin("tau", "$\tau_1$", 10, 0, 0.7)
        tau2_bins = hist.Bin("tau", "$\tau_2$", 10, 0, 0.5)
        tau3_bins = hist.Bin("tau", "$\tau_3$", 10, 0, 0.4)
        tau4_bins = hist.Bin("tau", "$\tau_4$", 10, 0, 0.3)
        tau21_bins = hist.Bin("tau", "$\tau_4$", 50, 0, 1.0)

        labels ={
            ('QCD_bEnriched_HT',): r'$QCD\ b-enriched (binned\ by\ HT)$',
            ('ZJetsToNuNu_HT',): r'$ZJets\to\nu\nu\ (binned\ by\ HT)$',
            ('WJetsToLNu_Njet',): r'$WJets\to L\nu\ (binned\ by\ N_{jets})$',
            ('TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',): r'$t\bar{t}$',
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',): '2HDMa_1500_750_10',
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',): '2HDMa_1750_750_10',
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000',): '2HDMa_2000_750_10',
        }

        colors ={
            ('QCD_bEnriched_HT',): '#D23FFE',
            ('ZJetsToNuNu_HT',): '#6BFE3F',
            ('WJetsToLNu_Njet',): '#FED23F',
            ('TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',): '#FE3F6B',
        }
        
        signals = [
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',),
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750',), 
            ('2HDMa_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000',),
        ]
            
        import cloudpickle
        import gzip
        outname = 'for_plotting_test'
        os.system("mkdir -p histos/")
        print('Saving output in %s...'%("histos/" + outname + ".pkl.gz"))
        with gzip.open("histos/" + outname + ".pkl.gz", "wb") as fout:
            cloudpickle.dump(scaled_output, fout)
        print('Done!')
        
        plot_dir = '/home/users/$USER/public_html/HbbMET/background/shape_'
        
        #need to add order to the plots
        makePlot2(scaled_output, 'met_pt', 'pt', met_bins, r'$MET_{pt}\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_pt', 'pt', pt_bins, r'$p_{T}\ (lead\ AK8)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_eta', 'eta', eta_bins, r'$\eta\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_phi', 'phi', phi_bins, r'$\phi\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_sdmass', 'mass', mass_bins, r'$mass\ (lead\ AK8)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau1', 'tau', tau1_bins, r'$\tau_1\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau2', 'tau', tau2_bins, r'$\tau_2\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau3', 'tau', tau3_bins, r'$\tau_3\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau4', 'tau', tau4_bins, r'$\tau_4\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau21', 'tau', tau21_bins, r'$\tau_{2}/\tau_{1}\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'nfatjet', 'multiplicity', N_bins, r'$n_{AK8}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_pt', 'pt', pt_bins2, r'$p_{T}\ (lead\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_eta', 'eta', eta_bins, r'$\eta\ (lead\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_phi', 'phi', phi_bins, r'$\phi\ (lead\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_mass', 'mass', mass_bins2, r'$mass\ (lead\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'nextrajet', 'multiplicity', N_bins2, r'$n_{AK4}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_pt', 'pt', pt_bins2, r'$p_{T}\ (lead\ b-tagged\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_eta', 'eta', eta_bins, r'$\eta\ (lead\ b-tagged\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_phi', 'phi', phi_bins, r'$\phi\ (lead\ b-tagged\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_mass', 'mass', mass_bins2, r'$mass\ (lead\ b-tagged\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'nextrabtag', 'multiplicity', N_bins2, r'$n_{b-tagged\ AK4}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'njet', 'multiplicity', N_bins2, r'$n_{AK4}\ (inclusive)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'ht', 'pt', ht_bins, r'$H_{T}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'dphiDiFatJet', 'phi', phi_bins2, r'$\Delta\phi\ (lead\ AK8,\ sublead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'dphileadextrajet', 'phi', phi_bins2, r'$\Delta\phi\ (lead\ AK8,\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'min_dphiFatJetMet4', 'phi', phi_bins2, r'$min\ \Delta\phi\ (lead\ four\ AK8,\ MET)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'min_dphiJetMetAll', 'phi', phi_bins2, r'$min\ \Delta\phi\ (AK4,\ MET)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'min_mt_fj_met', 'mt', mt_bins, r'$min\ M_{T}(AK8\ on\ H-mass,\ MET)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'n_b_in_AK8', 'multiplicity', N_bins2, r'$n_{b\ in\ AK8}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'n_H_in_AK8', 'multiplicity', N_bins2, r'$n_{b\ in\ AK8}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'NH_weight', 'multiplicity', N_bins, r'$n_{H}$', labels, colors, signals=signals, plot_dir=plot_dir)
        
        makePlot2(scaled_output, 'met_pt_tagged', 'pt', met_bins, r'$MET_{pt}\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_pt_tagged', 'pt', pt_bins, r'$p_{T}\ (lead\ AK8)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_eta_tagged', 'eta', eta_bins, r'$\eta\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_phi_tagged', 'phi', phi_bins, r'$\phi\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_sdmass_tagged', 'mass', mass_bins, r'$mass\ (lead\ AK8)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau1_tagged', 'tau', tau1_bins, r'$\tau_1\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau2_tagged', 'tau', tau2_bins, r'$\tau_2\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau3_tagged', 'tau', tau3_bins, r'$\tau_3\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau4_tagged', 'tau', tau4_bins, r'$\tau_4\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_fatjet_tau21_tagged', 'tau', tau21_bins, r'$\tau_{2}/\tau_{1}\ (lead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'nfatjet_tagged', 'multiplicity', N_bins, r'$n_{AK8}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_pt_tagged', 'pt', pt_bins2, r'$p_{T}\ (lead\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_eta_tagged', 'eta', eta_bins, r'$\eta\ (lead\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_phi_tagged', 'phi', phi_bins, r'$\phi\ (lead\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrajet_mass_tagged', 'mass', mass_bins2, r'$mass\ (lead\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'nextrajet_tagged', 'multiplicity', N_bins2, r'$n_{AK4}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_pt_tagged', 'pt', pt_bins2, r'$p_{T}\ (lead\ b-tagged\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_eta_tagged', 'eta', eta_bins, r'$\eta\ (lead\ b-tagged\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_phi_tagged', 'phi', phi_bins, r'$\phi\ (lead\ b-tagged\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'lead_extrabtag_mass_tagged', 'mass', mass_bins2, r'$mass\ (lead\ b-tagged\ AK4)\ (GeV)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'nextrabtag_tagged', 'multiplicity', N_bins2, r'$n_{b-tagged\ AK4}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'njet_tagged', 'multiplicity', N_bins2, r'$n_{AK4}\ (inclusive)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'ht_tagged', 'pt', ht_bins, r'$H_{T}$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'dphiDiFatJet_tagged', 'phi', phi_bins2, r'$\Delta\phi\ (lead\ AK8,\ sublead\ AK8)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'dphileadextrajet_tagged', 'phi', phi_bins2, r'$\Delta\phi\ (lead\ AK8,\ AK4)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'min_dphiFatJetMet4_tagged', 'phi', phi_bins2, r'$min\ \Delta\phi\ (lead\ four\ AK8,\ MET)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'min_dphiJetMetAll_tagged', 'phi', phi_bins2, r'$min\ \Delta\phi\ (AK4,\ MET)$', labels, colors, signals=signals, plot_dir=plot_dir)
        makePlot2(scaled_output, 'min_mt_fj_met_tagged', 'mt', mt_bins, r'$min\ M_{T}(AK8\ on\ H-mass,\ MET)$', labels, colors, signals=signals, plot_dir=plot_dir)
