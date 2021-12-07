#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

import awkward as ak
import uproot
import numpy as np
import glob
from coffea.nanoevents import NanoEventsFactory, DelphesSchema, BaseSchema
from coffea import hist, processor
# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

from functools import partial

from plots.helpers import makePlot2, scale_and_merge_histos
from tools.helpers import get_four_vec_fromPtEtaPhiM, match, delta_r_paf

class FlatProcessor(processor.ProcessorABC):
    def __init__(self, accumulator={}):
        self._accumulator = processor.dict_accumulator({
            "met_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
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
            "nfatjet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{fatjet}$", 6, -0.5, 5.5),
            ),
            "njet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{jet}$", 7, -0.5, 6.5),
            ),
            "ht": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$H_{T}$ [GeV]", 60, 0, 3000),
            ),
            "delta_phi_ml": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 17, 0, 4),
            ),
            "lead_fatjet_tau21": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_{2}/\tau{1}$", 100, 0, 2.0)
            ),
            #"matched_fatjet_pt": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("pt", "$p_{T}$ [GeV]", 100, 0, 1000),
            #),
            #"matched_fatjet_eta": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("eta", "$\eta$", 33, -4, 4),
            #),
            #"matched_fatjet_phi": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("phi", "$\phi$", 33, -4, 4),
            #),
            #"matched_fatjet_sdmass": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            #),
            #"matched_fatjet_tau1": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("tau", "$\tau_1$", 10, 0, 0.7),
            #),
            #"matched_fatjet_tau2": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("tau", "$\tau_2$", 10, 0, 0.5),
            #),
            #"matched_fatjet_tau3": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("tau", "$\tau_3$", 10, 0, 0.4),
            #),
            #"matched_fatjet_tau4": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("tau", "$\tau_4$", 10, 0, 0.3),
            #),
            #"nmatchedfatjet": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("multiplicity", "$n_{fatjet}$", 6, -0.5, 5.5),
            #),
            #"delta_r_lead_matched": hist.Hist(
            #    "Events",
            #    hist.Cat("dataset", "Dataset"),
            #    hist.Bin("deltaR", "$\Delta R$", 10, 0, 1),
            #),
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
        #sumw2 = np.sum(events['genweight']**2)
        #nevents = sum(events['genEventCount'])

        output[events.metadata['filename']]['sumWeight'] += sumw  # naming for consistency...
        #output[events.metadata['filename']]['sumWeight2'] += sumw2  # naming for consistency...
        #output[events.metadata['filename']]['nevents'] += nevents
        output[events.metadata['filename']]['nChunk'] += 1

        output[dataset]['sumWeight'] += sumw
        #output[dataset]['sumWeight2'] += sumw2
        #output[dataset]['nevents'] += nevents
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
            M = events.fatjet_msoftdrop,        #guess we're using softdrop from now on?
            copy = False,
        )
        
        #fatjet['m'] = events.fatjet_mass
        fatjet['tau1'] = events.fatjet_tau1
        fatjet['tau2'] = events.fatjet_tau2
        fatjet['tau3'] = events.fatjet_tau3
        fatjet['tau4'] = events.fatjet_tau4
        
        fatjet = fatjet[ak.argsort(fatjet.pt, ascending=False)]
        lead_fatjet = fatjet[:,0:1]
         
        tau21 = np.divide(lead_fatjet.tau2, lead_fatjet.tau1)
        #sublead_fatjet = fatjet[:,1:2]
        
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

        #selections
        met_sel = (met_pt>200)
        ele_sel = ((ak.num(ele_l)==0) & met_sel)
        mu_sel = ((ak.num(muon_l)==0) & ele_sel)
        tau_sel = ((ak.num(tau_l)==0) & mu_sel)
        baseline = ((ak.num(gamma_l)==0) & tau_sel)

        #output
        output['cutflow'][dataset]['total'] += len(met_pt)
        output['cutflow'][dataset]['met>200'] += len(met_pt[met_sel])
        output['cutflow'][dataset]['n_ele==0'] += len(met_pt[ele_sel])
        output['cutflow'][dataset]['n_mu==0'] += len(met_pt[mu_sel])
        output['cutflow'][dataset]['n_tau==0'] += len(met_pt[tau_sel])
        output['cutflow'][dataset]['n_gamma==0'] += len(met_pt[baseline])

        output["met_pt"].fill(
            dataset=dataset,
            pt=met_pt[baseline],
        )
        output["ht"].fill(
            dataset=dataset,
            pt = ak.sum(events.jetpuppi_pt[baseline], axis=1),
        )
        output["nfatjet"].fill(
            dataset=dataset,
            multiplicity=ak.num(fatjet[baseline]),
        )
        output["lead_fatjet_pt"].fill(
            dataset=dataset,
            pt=ak.flatten(lead_fatjet.pt[baseline], axis=1),
        )
        output["lead_fatjet_eta"].fill(
            dataset=dataset,
            eta=ak.flatten(lead_fatjet.eta[baseline], axis=1),
        )
        output["lead_fatjet_phi"].fill(
            dataset=dataset,
            phi=ak.flatten(lead_fatjet.phi[baseline], axis=1),
        )
        output["lead_fatjet_sdmass"].fill(
            dataset=dataset,
            mass=ak.flatten(lead_fatjet.mass[baseline], axis=1),
        )
        output["lead_fatjet_tau1"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau1[baseline], axis=1),
        )
        output["lead_fatjet_tau2"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau2[baseline], axis=1),
        )
        output["lead_fatjet_tau3"].fill(
            dataset=dataset,
            tau=ak.flatten(lead_fatjet.tau3[baseline], axis=1),
        )
        output["lead_fatjet_tau4"].fill(
            dataset=dataset,
           tau=ak.flatten(lead_fatjet.tau4[baseline], axis=1)
        )
        output["lead_fatjet_tau21"].fill(
            dataset=dataset,
            tau = ak.flatten(tau21[baseline])
        )
        #output["nfatjet"].fill(
        #    dataset=dataset,
        #    multiplicity=ak.num(fatjet[baseline]),
        #)
        #output["njet"].fill(
        #    dataset=dataset,
        #    multiplicity=ak.num(jet[baseline]),
        #)
        #output["delta_phi_ml"].fill(
        #    dataset=dataset,
        #    phi = ak.flatten(delta_phi_ml[baseline]),
        #)
        #output["nmatchedfatjet"].fill(
        #    dataset=dataset,
        #    multiplicity=n_matched_jet,
        #)
        #output["matched_fatjet_pt"].fill(
        #    dataset=dataset,
        #    pt=ak.flatten(matched_jet.pt[baseline], axis=1),
        #)
        #output["matched_fatjet_eta"].fill(
        #    dataset=dataset,
        #    eta=ak.flatten(matched_jet.eta[baseline], axis=1),
        #)
        #output["matched_fatjet_phi"].fill(
        #    dataset=dataset,
        #    phi=ak.flatten(matched_jet.phi[baseline], axis=1),
        #)
        #output["matched_fatjet_sdmass"].fill(
        #    dataset=dataset,
        #    mass=ak.flatten(matched_jet.mass[baseline], axis=1),
        #)
        #output["matched_fatjet_tau1"].fill(
        #    dataset=dataset,
        #    tau=ak.flatten(matched_jet.tau1[baseline], axis=1),
        #)
        #output["matched_fatjet_tau2"].fill(
        #    dataset=dataset,
        #    tau=ak.flatten(matched_jet.tau2[baseline], axis=1),
        #)
        #output["matched_fatjet_tau3"].fill(
        #    dataset=dataset,
        #    tau=ak.flatten(matched_jet.tau3[baseline], axis=1),
        #)
        #output["matched_fatjet_tau4"].fill(
        #    dataset=dataset,
        #    tau=ak.flatten(matched_jet.tau4[baseline], axis=1)
        #)
        #output["delta_r_lead_matched"].fill(
        #    dataset=dataset,
        #    deltaR=ak.to_numpy(delta_r_lead_matched)
        #)

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
        
        fileset = {
            'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU': samples['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']['ntuples'],
            'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU']['ntuples'],
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU']['ntuples'],
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU']['ntuples'],
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU']['ntuples'],
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU']['ntuples'],
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU': samples['ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU']['ntuples'],
            'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU': samples['W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['ntuples'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500']['ntuples'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500']['ntuples'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_350_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_350_MH2_1500_MHC_1500']['ntuples'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500']['ntuples'],
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500': samples['2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500']['ntuples'],
        }

        meta_accumulator = {}
        for sample in fileset:
            if sample not in meta_accumulator:
                meta_accumulator.update({sample: processor.defaultdict_accumulator(int)})
            for f in fileset[sample]:
                meta_accumulator.update({f: processor.defaultdict_accumulator(int)})

        output_flat = processor.run_uproot_job(
            fileset,
            treename='myana/mytree',
            processor_instance = FlatProcessor(accumulator=meta_accumulator),
            executor = exe,
            executor_args = exe_args,
            chunksize=1000000,
            maxchunks=None,
        )
        
        meta = {}

        for sample in fileset:
            meta[sample] = output_flat[sample]
            meta[sample]['xsec'] = samples[sample]['xsec']

        import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

        N_bins = hist.Bin('multiplicity', r'$N$', 6, -0.5, 5.5)
        N_bins2 = hist.Bin('multiplicity', r'$N$', 7, -0.5, 6.5)
        mass_bins = hist.Bin('mass', r'$M\ (GeV)$', 40, 0, 400)
        ht_bins = hist.Bin('pt', r'$H_{T}\ (GeV)$', 60, 0, 3000)
        pt_bins = hist.Bin('pt', r'$p_{T}\ (GeV)$', 80, 200, 1000)
        eta_bins = hist.Bin("eta", "$\eta$", 33, -4, 4)
        phi_bins = hist.Bin("phi", "$\phi$", 33, -4, 4)
        phi_bins2 = hist.Bin("phi", "$\phi$", 17, 0, 4)
        deltaR_bins = hist.Bin("deltaR", "$\DeltaR$", 10, 0, 1)
        tau1_bins = hist.Bin("tau", "$\tau_1$", 10, 0, 0.7)
        tau2_bins = hist.Bin("tau", "$\tau_2$", 10, 0, 0.5)
        tau3_bins = hist.Bin("tau", "$\tau_3$", 10, 0, 0.4)
        tau4_bins = hist.Bin("tau", "$\tau_4$", 10, 0, 0.3)
        tau21_bins = hist.Bin("tau", "$\tau_4$", 100, 0, 2.0)

        labels ={
            ('ZJetsToNuNu_HT',): r'$ZJets\to\nu\nu\ (binned\ by\ HT)$',
            ('WJetsToLNu_Njet',): r'$WJets\to L\nu\ (binned\ by\ N_{jets})$',
            ('TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',): r'$t\bar{t}$',
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',): '2HDMa_bb_1500_150_10',
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500',): '2HDMa_bb_1500_250_10',
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_350_MH2_1500_MHC_1500',): '2HDMa_bb_1500_350_10',
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500',): '2HDMa_bb_1500_500_10',
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',): '2HDMa_bb_1500_750_10',
        }

        colors ={
            ('ZJetsToNuNu_HT',): '#355C7D',
            ('WJetsToLNu_Njet',): '#FED23F',
            ('TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',): '#EB7DB5',
        }
        
        signals = [
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',),
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500',), 
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_350_MH2_1500_MHC_1500',), 
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500',), 
            ('2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',)
        ]

        scaled_output = {}
        
        for key in output_flat.keys():
            if type(output_flat[key]) is not type(output_flat['cutflow']):
                scaled_output[key] = scale_and_merge_histos(output_flat[key], meta, fileset, lumi=3000)
            
        #need to add order to the plots
        makePlot2(scaled_output, 'met_pt', 'pt', pt_bins, r'$MET_{pt}\ (GeV)$', labels, colors, signals=signals)
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
        makePlot2(scaled_output, 'njet', 'multiplicity', N_bins2, r'$n_{jet}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'ht', 'pt', ht_bins, r'$H_{T}$', labels, colors, signals=signals)
        makePlot2(scaled_output, 'delta_phi_ml', 'phi', phi_bins2, r'$\Delta\phi\ (lead\ fatjet,\ MET)$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_pt', 'pt', pt_bins, r'$p_{T}\ (GeV)$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_eta', 'eta', eta_bins, r'$\eta$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_phi', 'phi', phi_bins, r'$\phi$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_sdmass', 'mass', mass_bins, r'$mass\ (GeV)$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_tau1', 'tau', tau1_bins, r'$\tau_1$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_tau2', 'tau', tau2_bins, r'$\tau_2$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_tau3', 'tau', tau3_bins, r'$\tau_3$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'matched_fatjet_tau4', 'tau', tau4_bins, r'$\tau_4$', labels, colors, signals=signals)
        #makePlot2(output_flat, 'nmatchedfatjet', 'multiplicity', N_bins, r'$n_{fatjet}$', labels, colors, signals=signals)     
        #makePlot2(output_flat, 'delta_r_lead_matched', 'deltaR', deltaR_bins, r'$\Delta R$', labels, colors, signals=signals)
