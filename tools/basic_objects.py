from tools.helpers import get_four_vec_fromPtEtaPhiM

#can add pt and eta cuts in later

def getJets(events, corrector, pt_var='central'):
    """takes a set of events and a set of weights and
    weights the jet pts of those events according to
    the variation given."""
    
    jet = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.jetpuppi_pt,
            eta = events.jetpuppi_eta,
            phi = events.jetpuppi_phi,
            M = events.jetpuppi_mass,
            copy = False,
        )
    
    jet_pt_var = corrector.get(jet, pt_var)
    
    jet['pt'] = jet_pt_var
    
    jet['id'] = events.jetpuppi_idpass
    jet['btag'] = events.jetpuppi_btag
    
    return jet


def getFatjets(events, corrector, pt_var='central'):
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
        
    