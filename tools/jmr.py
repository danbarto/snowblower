#!/usr/bin/env python3

import awkward as ak
import numpy as np
import copy

class JMR:
    def __init__(self, seed=123):
        self.seed = 123

    def get(self, jet, scale=0., res=0.):
        '''
        get rescaled / smeared mass. jet object remains unchanged.
        jet: four vector object
        scale: relative scale correction
        res: relative resolution smearing
        '''
        np.random.seed(seed=self.seed)

        jet_flat = ak.flatten(jet)

        correction = np.maximum(
            1+np.random.normal(
                loc=scale,
                scale=res,
                size=len(jet_flat),
            ),
            0
        )

        jet_new = copy.deepcopy(jet)

        jet_new['mass'] = ak.unflatten(jet_flat.mass * correction, ak.num(jet))

        return jet_new


if __name__ == '__main__':

    from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
    from coffea import hist, processor
    # register our candidate behaviors
    from coffea.nanoevents.methods import candidate
    ak.behavior.update(candidate.behavior)

    from functools import partial

    from tools.helpers import get_four_vec_fromPtEtaPhiM, match, match_count


    events = NanoEventsFactory.from_root(
        '/nfs-7/userdata/dspitzba/merge_ZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_200PU_v16/merge_1.root',
        treepath='mytree',
        schemaclass=BaseSchema,
    ).events()

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

    bquark = gen[((abs(gen.pdgId)==5)&(gen.status==71))]  # I suspect that Delphes does not keep b's with pt less than 20?
    higgs = gen[(abs(gen.pdgId)==25)][:,-1:]  # just get the last Higgs. Delphes is not keeping all the higgses.

    fatjet = get_four_vec_fromPtEtaPhiM(
        None,
        pt = events.fatjet_pt,
        eta = events.fatjet_eta,
        phi = events.fatjet_phi,
        M = events.fatjet_msoftdrop,        #Using softdrop from now on
        copy = False,
    )

    fatjet = fatjet[
        (fatjet.pt>300) &\
        (abs(fatjet.eta)<2.4)
        #(ev.FatJet.jetId>0)
    ]

    nhiggs_in_fat = match_count(fatjet, higgs, deltaRCut=0.8)
    nb_in_fat = match_count(fatjet, bquark, deltaRCut=0.8)

    print ("Original jet masses:")
    print (ak.flatten(fatjet.mass))

    from tools.jmr import JMR
    jmr = JMR(seed=123)
    fatjet_smeared = jmr.get(fatjet, scale=0.05, res=0.1)

    print ("\nCorrected jet masses:")
    print (ak.flatten(fatjet_smeared.mass))
