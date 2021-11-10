#!/usr/bin/env python3

#import warnings
#warnings.filterwarnings("ignore")

import awkward as ak
import uproot
import numpy as np
from coffea.nanoevents import NanoEventsFactory, DelphesSchema

from coffea import hist, processor

# register our candidate behaviors
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)

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


if __name__ == '__main__':

    from yaml import Loader, Dumper
    import yaml

    with open('../tools/samples.yaml', 'r') as f:
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


    import time

    tstart = time.time()

    fileset = {
        'ttHbb': [
            f_in,
        ],
        'QCD': samples['QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']['delphes'],
    }

    output = processor.run_uproot_job(
        fileset,
        treename='Delphes',
        processor_instance = DelphesProcessor(),
        executor=processor.futures_executor,
        executor_args={"schema": DelphesSchema, "workers": 4},
        chunksize=100000,
        maxchunks=None,
    )

    elapsed = time.time() - tstart
    print(output)
