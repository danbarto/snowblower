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

from tools.helpers import get_four_vec_fromPtEtaPhiM, match


class DelphesProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "met": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']

        met = events.PuppiMissingET

        output["met"].fill(
            dataset=dataset,
            pt=ak.flatten(met.MET, axis=1),
        )

        return output

    def postprocess(self, accumulator):
        return accumulator

class FlatProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "met": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
            "nElectron": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$N", 5, -0.5, 4.5),
            ),
            'cutflow': processor.defaultdict_accumulator(
                # we don't use a lambda function to avoid pickle issues
                partial(processor.defaultdict_accumulator, int)
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']

        output['cutflow'][dataset]['total'] += len(ak.flatten(events.metpuppi_pt))
        output['cutflow'][dataset]['met>200'] += len(events[ak.flatten(events.metpuppi_pt)>200])

        met = events.metpuppi_pt


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
        electron['charge'] = events.elec_charge

        ele_l = electron[((electron['id']>0)&(electron['iso']>0))]


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
            M = events.fatjet_msoftdrop,
            copy = False,
        )
        #fatjet['tau1'] = events.fatjet_tau1

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

        higgs = gen[(gen.pdgId==25)][:,-1:]  # only keep the last copy. status codes seem messed up?

        matched_jet = fatjet[match(fatjet, higgs, deltaRCut=0.8)]
        n_matched_jet = ak.num(matched_jet)
        # now do with that what you want

        baseline = ((ak.num(ele_l)==0) & (events.metpuppi_pt>100))

        met_sel = met[baseline]

        output["met"].fill(
            dataset=dataset,
            pt=ak.flatten(met_sel, axis=1),
        )

        output["nElectron"].fill(
            dataset=dataset,
            multiplicity=ak.num(ele_l),
        )

        return output

    def postprocess(self, accumulator):
        return accumulator

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
