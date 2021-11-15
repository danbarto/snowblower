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
	    "fatjet_pt": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("pt", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
	    "fatjet_eta": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("eta", "$\eta$", 33, -4, 4),
            ),
	    "fatjet_phi": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("phi", "$\phi$", 3, -4, 4),
            ),
	    "fatjet_mass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
	    "fatjet_sdmass": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("mass", "$p_{T}$ [GeV]", 50, 0, 500),
            ),
	    "fatjet_tau1": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_1$", 10, 0, 1),
            ),
	    "fatjet_tau2": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_2$", 10, 0, 1),
            ),
	    "fatjet_tau3": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_3$", 10, 0, 1),
            ),
	    "fatjet_tau4": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("tau", "$\tau_4$", 10, 0, 1),
            ),
	    "nfatjet": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("multiplicity", "$n_{fatjet}$", 6, -0.5, 5.5),
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
	fatjet_pt = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_pt
	fatjet_eta = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_eta
	fatjet_phi = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_phi
	fatjet_mass = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_mass
	fatjet_msoftdrop = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_msoftdrop
	fatjet_tau1 = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_tau1
	fatjet_tau2 = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_tau2
	fatjet_tau3 = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_tau3
	fatjet_tau4 = events[ak.argsort(events.fatjet_pt, ascending=False)].fatjet_tau4
	nfatjet = events.fatjet_size

        output["met"].fill(
            dataset=dataset,
            pt=ak.flatten(met, axis=1),
        )
	output["nfatjet"].fill(
            dataset=dataset,
            multiplicity=nfatjet,
        )
	output["fatjet_pt"].fill(
            dataset=dataset,
            pt=fatjet_pt[:,0:1],
        )
	output["fatjet_eta"].fill(
            dataset=dataset,
            eta=fatjet_eta[:,0:1],
        )
	output["fatjet_phi"].fill(
            dataset=dataset,
            phi=fatjet_phi[:,0:1],
        )
	output["fatjet_mass"].fill(
            dataset=dataset,
            mass=fatjet_mass[:,0:1],
        )
	output["fatjet_msoftdrop"].fill(
            dataset=dataset,
            mass=fatjet_msoftdrop[:,0:1],
        )
	output["fatjet_tau1"].fill(
            dataset=dataset,
            tau=fatjet_tau1[:,0:1],
        )
	output["fatjet_tau2"].fill(
            dataset=dataset,
            tau=fatjet_tau2[:,0:1],
        )
	output["fatjet_tau3"].fill(
            dataset=dataset,
            tau=fatjet_tau3[:,0:1],
        )
	output["fatjet_tau4"].fill(
            dataset=dataset,
            tau=fatjet_tau4[:,0:1],
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
        '/nfs-7/userdata/dspitzba/ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU//ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU_1.root',
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

	import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

	#impliment makePlot once I make sure that everything else is working appropriately


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
	
	import matplotlib.pyplot as plt
        import mplhep as hep
        plt.style.use(hep.style.CMS)

        #impliment makePlot once I make sure that everything else is working appropriately
    #print(output)
