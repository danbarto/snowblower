from tools.helpers import get_four_vec_fromPtEtaPhiM

#can add pt and eta cuts in later

def getJets(events, corrector, pt_var=''):
    """takes a set of events and a set of weights and
    weights the jet pts of those events according to
    the variation given."""

    jet_presel = (events.jetpuppi_pt>20)  # NOTE: careful with this presel, but needed to speed up the procssor

    jet = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.jetpuppi_pt[jet_presel],
            eta = events.jetpuppi_eta[jet_presel],
            phi = events.jetpuppi_phi[jet_presel],
            M = events.jetpuppi_mass[jet_presel],
            copy = False,
        )
    
    jet_pt_var = corrector.get(jet, pt_var)
    
    jet['pt'] = jet_pt_var[jet_presel]
    
    jet['id'] = events.jetpuppi_idpass[jet_presel]
    jet['btag'] = events.jetpuppi_btag[jet_presel]
    
    return jet


def getFatjets(events, corrector, pt_var=''):
    """takes a set of events and a set of weights and
    weights the fatjet pts of those events according to
    the variation given."""
    
    fatjet = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.fatjet_pt,
            eta = events.fatjet_eta,
            phi = events.fatjet_phi,
            M = events.fatjet_msoftdrop,        #Using softdrop from now on
            copy = False,
        )
    
    fatjet_pt_var = corrector.get(fatjet, pt_var)
    
    fatjet['pt'] = fatjet_pt_var
    
    fatjet['tau1'] = events.fatjet_tau1
    fatjet['tau2'] = events.fatjet_tau2
    fatjet['tau3'] = events.fatjet_tau3
    fatjet['tau4'] = events.fatjet_tau4
    
    return fatjet
