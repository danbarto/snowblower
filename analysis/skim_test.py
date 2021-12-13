#!/usr/bin/env python3
import awkward as ak
import uproot
import numpy as np
import glob
from coffea.nanoevents import NanoEventsFactory, DelphesSchema, BaseSchema
from coffea.analysis_tools import Weights
from coffea import hist, processor
# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

from functools import partial

from plots.helpers import makePlot2, scale_and_merge_histos
from tools.helpers import choose, get_four_vec_fromPtEtaPhiM, match, mt

import warnings
warnings.filterwarnings("ignore")

class FlatProcessor(processor.ProcessorABC):
    def __init__(self, accumulator={}):
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
                hist.Bin("multiplicity", "$n_{fatjet}$", 5, 0.5, 5.5),
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
                hist.Bin("multiplicity", "$n_{jet}$", 7, 0.5, 7.5),
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
                hist.Bin("multiplicity", "$n_{jet}$", 7, 0.5, 7.5),
            ),
            "njet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, 0.5, 7.5),
            ),
            "dphiDiFatJet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            "min_dphiFatJetMet4": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 16, 0, 4),
            ),
            'cutflow': processor.defaultdict_accumulator(
                # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int)
            ),
        })
        
        self.accumulator.update(processor.dict_accumulator( accumulator ))

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
        
        gamma = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.gamma_pt,
            eta = events.gamma_eta,
            phi = events.gamma_phi,
            M = events.gamma_mass,
            copy = False,
        )
        gamma['id'] = events.gamma_idpass  # > 0 should be loose
        gamma['iso'] = events.gamma_isopass

        gamma_l = gamma[((gamma['id']>0)&(gamma['iso']>0)&(gamma.pt>20)&(np.abs(gamma.eta)<3))]
            
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
        jet = jet[~match(jet, electron, deltaRCut=0.4)] #remove electron overlap
        jet = jet[~match(jet, muon, deltaRCut=0.4)] #remove muon overlap

        btag = jet[jet.btag>0] #loose wp for now
        
        #fatjets

        # Need FatJets and GenParts
        # FatJets start at pt>200 and go all the way to eta 3.x
        # This should be fine?
        # Objects are defined here: https://twiki.cern.ch/twiki/bin/view/CMS/DelphesInstructions
        # Maybe restrict abs(eta) to 2.8 or 3 (whatever the tracker acceptance of PhaseII CMS is)
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
        
        fatjet_on_h = fatjet[np.abs(fatjet.mass-125)<25]
        on_h = (ak.num(fatjet_on_h) > 0)
        
        lead_fatjet = fatjet[:,0:1]
        #sublead_fatjet = fatjet[:,1:2] 
        
        lead_fatjets = fatjet[:,0:2]
        difatjet = choose(lead_fatjets, 2)
        dphiDiFatJet = np.arccos(np.cos(difatjet['0'].phi-difatjet['1'].phi))
        
        tau21 = np.divide(lead_fatjet.tau2, lead_fatjet.tau1)
        
        extrajet  = jet[~match(jet, fatjet, deltaRCut=1.2)] # remove AK4 jets that overlap with AK8 jets
        extrabtag = extrajet[extrajet.btag>0] #loose wp for now]
        extrajet = extrajet[ak.argsort(extrajet.pt, ascending=False)]
        extrabtag = extrabtag[ak.argsort(extrabtag.pt, ascending=False)]
        lead_extrajet = extrajet[:,0:1]
        lead_extrabtag = extrabtag[:,0:1]
        
                
        #gen
        
        #gen = get_four_vec_fromPtEtaPhiM(
        #    None,
        #    pt = events.genpart_pt,
        #    eta = events.genpart_eta,
        #    phi = events.genpart_phi,
        #    M = events.genpart_mass,
        #    copy = False,
        #)
        #gen['pdgId'] = events.genpart_pid
        #gen['status'] = events.genpart_status

        #higgs = gen[(gen.pdgId==25)][:,-1:]  # only keep the last copy. status codes seem messed up?

        #matched_jet = fatjet[match(fatjet, higgs, deltaRCut=0.8)]
        
        #n_matched_jet = ak.num(matched_jet)
        
        #delta_r_lead_matched = delta_r_paf(matched_jet[:,0:1], lead_fatjet)   #just taking the first matched, in case there are multiple
      
        
        #MET
        
        met_pt = ak.flatten(events.metpuppi_pt)
        met_phi = ak.flatten(events.metpuppi_phi)
        
        delta_phi_ml = np.arccos(np.cos(lead_fatjet.phi - met_phi))
        mt_fj_met = mt(fatjet_on_h.pt, fatjet_on_h.phi, met_pt, met_phi)
        min_mt_fj_met = ak.min(mt_fj_met, axis=1, mask_identity=False)
        min_dphiFatJetMet4 = ak.min(np.arccos(np.cos(fatjet[:,:4].phi-met_phi)), axis=1, mask_identity=False)

        #selections
        ele_sel = (ak.num(ele_l)==0)
        mu_sel = ((ak.num(muon_l)==0) & ele_sel)
        tau_sel = ((ak.num(tau_l)==0) & mu_sel)
        met_sel = (tau_sel & (met_pt>200))
        n_ak8 = ((ak.num(fatjet)>0) & met_sel)
        min_dphiFatJetMet4_sel = ((min_dphiFatJetMet4 > 0.5) & n_ak8)
        dphiDiFatJet_sel = (ak.all(dphiDiFatJet < 2.5, axis=1) & min_dphiFatJetMet4_sel)
        on_h_sel = (on_h & dphiDiFatJet_sel)
        min_mt = ((min_mt_fj_met > 200) & on_h_sel)

        #output
        output['cutflow'][dataset]['total'] += len(events)
        output['cutflow'][dataset]['n_ele==0'] += len(events[ele_sel])
        output['cutflow'][dataset]['n_mu==0'] += len(events[mu_sel])
        output['cutflow'][dataset]['n_tau==0'] += len(events[tau_sel])
        output['cutflow'][dataset]['met>200'] += len(events[met_sel])
        output['cutflow'][dataset]['n_ak8>=1'] += len(events[n_ak8])
        output['cutflow'][dataset]['min_dphiFatJetMet4>0.5'] += len(events[min_dphiFatJetMet4_sel])
        output['cutflow'][dataset]['dphiDiFatJet<2.5'] += len(events[dphiDiFatJet_sel])
        output['cutflow'][dataset]['on-H'] += len(events[on_h_sel])
        output['cutflow'][dataset]['minmth>200'] += len(events[min_mt])
        
        output["met_pt"].fill(
            dataset=dataset,
            pt=met_pt[met_sel],
        )
        output["ht"].fill(
            dataset=dataset,
            pt = ak.sum(events.jetpuppi_pt[met_sel], axis=1),
        )
        output["min_mt_fj_met"].fill(
            dataset=dataset,
            mt = ak.min(mt_fj_met[(met_sel & on_h)], axis=1),
        )
        output["nfatjet"].fill(
            dataset=dataset,
            multiplicity=ak.num(fatjet[met_sel]),
        )
        output["lead_fatjet_pt"].fill(
            dataset=dataset,
            pt=ak.to_numpy(ak.flatten(lead_fatjet.pt[met_sel], axis=1)),
        )
        output["lead_fatjet_eta"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_fatjet.eta[met_sel], axis=1),
        )
        output["lead_fatjet_phi"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_fatjet.phi[met_sel], axis=1),
        )
        output["lead_fatjet_sdmass"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_fatjet.mass[met_sel], axis=1),
        )
        output["lead_fatjet_tau1"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau1[met_sel], axis=1),
        )
        output["lead_fatjet_tau2"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau2[met_sel], axis=1),
        )
        output["lead_fatjet_tau3"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau3[met_sel], axis=1),
        )
        output["lead_fatjet_tau4"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau4[met_sel], axis=1),
        )
        output["lead_fatjet_tau21"].fill(
            dataset=dataset,
            tau = ak.flatten(tau21[met_sel], axis=1),
        )
        output["nfatjet"].fill(
            dataset=dataset,
            multiplicity=ak.num(fatjet[met_sel], axis=1),
        )
        output["lead_extrajet_pt"].fill(
            dataset=dataset,
            pt=ak.flatten(lead_extrajet.pt[met_sel], axis=1),
        )
        output["lead_extrajet_eta"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_extrajet.eta[met_sel], axis=1),
        )
        output["lead_extrajet_phi"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_extrajet.phi[met_sel], axis=1),
        )
        output["lead_extrajet_mass"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_extrajet.mass[met_sel], axis=1),
        )
        output["nextrajet"].fill(
            dataset=dataset,
            multiplicity=ak.num(extrajet[met_sel]),
        )
        output["lead_extrabtag_pt"].fill(
            dataset=dataset,
            pt=ak.flatten(lead_extrabtag.pt[met_sel], axis=1),
        )
        output["lead_extrabtag_eta"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_extrabtag.eta[met_sel], axis=1),
        )
        output["lead_extrabtag_phi"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_extrabtag.phi[met_sel], axis=1),
        )
        output["lead_extrabtag_mass"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_extrabtag.mass[met_sel], axis=1),
        )
        output["nextrabtag"].fill(
            dataset=dataset,
            multiplicity=ak.num(extrabtag[met_sel]),
        )
        output["njet"].fill(
            dataset=dataset,
            multiplicity=ak.num(jet[met_sel]),
        )
        output["dphiDiFatJet"].fill(
            dataset=dataset,
            phi = ak.flatten(dphiDiFatJet[(met_sel & (ak.num(fatjet)>0))]),
        )
        output["min_dphiFatJetMet4"].fill(
            dataset=dataset,
            phi = min_dphiFatJetMet4[(met_sel & (ak.num(fatjet)>0))],
        )
        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':

    from yaml import Loader, Dumper
    import yaml


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
        
        fileset_skim = {
            #'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            #'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            #'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
            'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['skim'],
        }
        
        fileset_ntuples = {
            #'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2': samples['W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            #'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2': samples['W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            #'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2': samples['W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2': samples['W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
        }
        
        meta_accumulator_ntuples = {}
        for sample in fileset_ntuples:
            if sample not in meta_accumulator_ntuples:
                meta_accumulator_ntuples.update({sample: processor.defaultdict_accumulator(int)})
            for f in fileset_ntuples[sample]:
                meta_accumulator_ntuples.update({f: processor.defaultdict_accumulator(int)})

        output_ntuples = processor.run_uproot_job(
            fileset_ntuples,
            treename='myana/mytree',
            processor_instance = FlatProcessor(accumulator=meta_accumulator_ntuples),
            executor = exe,
            executor_args = exe_args,
            chunksize=1000000,
            maxchunks=None,
        )
        
        meta_accumulator_skim = {}
        for sample in fileset_skim:
            if sample not in meta_accumulator_skim:
                meta_accumulator_skim.update({sample: processor.defaultdict_accumulator(int)})
            for f in fileset_skim[sample]:
                meta_accumulator_skim.update({f: processor.defaultdict_accumulator(int)})

        exe = processor.futures_executor
        exe_args = {"schema": BaseSchema, "workers": 20}
        
        output_skim = processor.run_uproot_job(
            fileset_skim,
            treename='mytree',
            processor_instance = FlatProcessor(accumulator=meta_accumulator_skim),
            executor = exe,
            executor_args = exe_args,
            chunksize=1000000,
            maxchunks=None,
        )
        
        meta = {}

        for sample in fileset_skim:
            meta[sample] = output_skim[sample]
            meta[sample]['xsec'] = samples[sample]['xsec']
            meta[sample]['nevents'] = samples[sample]['nevents']
        
        for sample in fileset_ntuples:
            meta[sample] = output_ntuples[sample]
            meta[sample]['xsec'] = samples[sample]['xsec']
            meta[sample]['nevents'] = samples[sample]['nevents']

        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

        N_bins = hist.Bin('multiplicity', r'$N$', 5, 0.5, 5.5)
        N_bins2 = hist.Bin('multiplicity', r'$N$', 7, 0.5, 7.5)
        mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 40, 0, 400)
        ht_bins = hist.Bin('pt', r'$H_{T}\ (GeV)$', 60, 0, 3000)
        mt_bins = hist.Bin('mt', r'$M_{T}\ (GeV)$', 40, 0, 2000)
        pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 80, 200, 1000)
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
            ('WJetsToLNu_Njet1',): r'$WJets\to L\nu\ (skims)$',
            ('WJetsToLNu_Njet2',): r'$WJets\to L\nu\ (ntuples)$',
        }

        colors ={
            ('WJetsToLNu_Njet1',): '#FED23F',
        }
        
        signals = [
            ('WJetsToLNu_Njet2',),
        ]

        scaled_output = {}
        
        for key in output_skim.keys():
            if type(output_skim[key]) is not type(output_skim['cutflow']):
                scaled_output[key] = scale_and_merge_histos(output_skim[key], meta, fileset_skim, lumi=3000)
        for key in output_ntuples.keys():
            if type(output_ntuples[key]) is not type(output_ntuples['cutflow']):
                scaled_output[key] += scale_and_merge_histos(output_ntuples[key], meta, fileset_ntuples, lumi=3000)
            
        #need to add order to the plots
        makePlot2(scaled_output, 'met_pt', 'pt', met_bins, r'$MET_{pt}\ (GeV)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_pt', 'pt', pt_bins, r'$p_{T}\ (GeV)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_eta', 'eta', eta_bins, r'$\eta$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_phi', 'phi', phi_bins, r'$\phi$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_sdmass', 'mass', mass_bins, r'$mass\ (GeV)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_tau1', 'tau', tau1_bins, r'$\tau_1$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_tau2', 'tau', tau2_bins, r'$\tau_2$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_tau3', 'tau', tau3_bins, r'$\tau_3$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_tau4', 'tau', tau4_bins, r'$\tau_4$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_fatjet_tau21', 'tau', tau21_bins, r'$\tau_{2}/\tau_{1}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'nfatjet', 'multiplicity', N_bins, r'$n_{fatjet}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrajet_pt', 'pt', met_bins, r'$p_{T}\ (GeV)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrajet_eta', 'eta', eta_bins, r'$\eta$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrajet_phi', 'phi', phi_bins, r'$\phi$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrajet_mass', 'mass', mass_bins, r'$mass\ (GeV)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'nextrajet', 'multiplicity', N_bins2, r'$n_{jet}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrabtag_pt', 'pt', met_bins, r'$p_{T}\ (GeV)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrabtag_eta', 'eta', eta_bins, r'$\eta$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrabtag_phi', 'phi', phi_bins, r'$\phi$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'lead_extrabtag_mass', 'mass', mass_bins, r'$mass\ (GeV)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'nextrabtag', 'multiplicity', N_bins2, r'$n_{jet}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'njet', 'multiplicity', N_bins2, r'$n_{jet}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'ht', 'pt', ht_bins, r'$H_{T}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'dphiDiFatJet', 'phi', phi_bins2, r'$\Delta\phi\ (lead\ fatjets)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'min_dphiFatJetMet4', 'phi', phi_bins2, r'$min\ \Delta\phi\ (lead\ fatjets,\ MET)$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'min_mt_fj_met', 'mt', mt_bins, r'$M_{T}$', labels, colors, signals=signals)
