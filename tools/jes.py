import os
try:
    import awkward1 as ak
except ImportError:
    import awkward as ak

from coffea.lookup_tools import extractor
import numpy as np

class JES:

    def __init__(self):

        # NOTE file from here: https://twiki.cern.ch/twiki/bin/view/CMS/YR2018Systematics
        # some details in here: https://twiki.cern.ch/twiki/pub/CMS/YR2018Systematics/ATLASsystematics.pdf
        jes_file      = os.path.expandvars("$TWHOME/data/jme/HL_YR_JEC.root")

        print ("Loading JES file: %s"%jes_file)

        self.ext = extractor()
        # several histograms can be imported at once using wildcards (*)
        self.ext.add_weight_sets([
            "jes TOTAL_DIJET_AntiKt4EMTopo_YR2018 %s"%jes_file,  # NOTE: if need be, we can still use separate b-jet corrections
        ])

        self.ext.finalize()
        
        self.evaluator = self.ext.make_evaluator()

    def get(self, jet, variation='central'):
        
        jes = self.evaluator["jes"](jet.pt)

        if variation == 'central':
            # We don't reapply any JES corrections
            # So let's just bounce back the values we got in the first place
            #ak.ones_like(ak.num(jet.pt, axis=1))
            return jet.pt
        elif variation == 'up':
            return jet.pt * (1 + jes)
        elif variation == 'down':
            return jet.pt * (1 - jes)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    jes_corrector = JES()

    print("Evaluators found:")
    for key in jes_corrector.evaluator.keys():
        print("%s:"%key, jes_corrector.evaluator[key])

    ## Load a single file here, get leptons, eval SFs just to be sure everything works
    from coffea.nanoevents import NanoEventsFactory, BaseSchema
    from tools.helpers import get_four_vec_fromPtEtaPhiM

    import awkward as ak
    
    # load a subset of events
    n_max = 5000
    events = NanoEventsFactory.from_root(
        '/nfs-7/userdata/ewallace/Hbb_MET/2HDMA_bb-gg_YY_XX_10_v4/2HDMa_bb_1500_750_10_delphes_ntuple.root',
        schemaclass = BaseSchema,
        treepath = 'myana/mytree',
        entry_stop = n_max).events()

    jets = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.jetpuppi_pt,
            eta = events.jetpuppi_eta,
            phi = events.jetpuppi_phi,
            M = events.jetpuppi_mass,
            copy = False,
        )

    print ("Original jet pts of first event:")
    print (jets.pt[0, :])
    jet_pt_central  = jes_corrector.get(jets, variation='central')
    print ("Central jet pts of first event:")
    print (jet_pt_central[0, :])
    jet_pt_up  = jes_corrector.get(jets, variation='up')
    print ("JES up jet pts of first event:")
    print (jet_pt_up[0, :])
    jet_pt_down  = jes_corrector.get(jets, variation='down')
    print ("JES up jet pts of first event:")
    print (jet_pt_down[0, :])
