from tools.helpers import get_four_vec_fromPtEtaPhiM, match
from tools.jmr import JMR
import awkward as ak

#can add pt and eta cuts in later

def getJets(events, corrector, pt_var='', scale_res = '', delphes=False):
    """takes a set of events and a set of weights and
    weights the jet pts of those events according to
    the variation given."""
    
    if delphes == False:
        jet_presel = (events.jetpuppi_pt>20)  # NOTE: careful with this presel, but needed to speed up the processor

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

        if scale_res != '':
            jmr = JMR(seed=123)
            jet = jmr.get(jet, scale=scale_res[0], res=scale_res[1])

        jet['id'] = events.jetpuppi_idpass[jet_presel]
        jet['btag'] = events.jetpuppi_btag[jet_presel]
        
    elif delphes == True:
        jet_l_presel = (events.JetPUPPILoose.PT>20)  # NOTE: careful with this presel, but needed to speed up the processor

        jet_l = get_four_vec_fromPtEtaPhiM(
                None,
                pt = events.JetPUPPILoose.PT[jet_l_presel],
                eta = events.JetPUPPILoose.Eta[jet_l_presel],
                phi = events.JetPUPPILoose.Phi[jet_l_presel],
                M = events.JetPUPPILoose.Mass[jet_l_presel],
                copy = False,
            )
        
        jet_t_presel = (events.JetPUPPITight.PT>20)  # NOTE: careful with this presel, but needed to speed up the processor

        jet_t = get_four_vec_fromPtEtaPhiM(
                None,
                pt = events.JetPUPPITight.PT[jet_t_presel],
                eta = events.JetPUPPITight.Eta[jet_t_presel],
                phi = events.JetPUPPITight.Phi[jet_t_presel],
                M = events.JetPUPPITight.Mass[jet_t_presel],
                copy = False,
            )
        
        jet_t = jet_t[~match(jet_t, jet_l, deltaRCut=0.1)]
        
        #jet_presel = (events.JetPUPPI.PT>20)  # NOTE: careful with this presel, but needed to speed up the processor

        #jet = get_four_vec_fromPtEtaPhiM(
        #        None,
        #        pt = events.JetPUPPI.PT[jet_presel],
        #        eta = events.JetPUPPI.Eta[jet_presel],
        #        phi = events.JetPUPPI.Phi[jet_presel],
        #        M = events.JetPUPPI.Mass[jet_presel],
        #        copy = False,
        #    )
        
        #jet = jet[match(jet, jet_l, deltaRCut=0.1)]
        
        jet = ak.concatenate((jet_l,jet_t),axis=1)
        
        jet_pt_var = corrector.get(jet, pt_var)
        
        jet['pt'] = jet_pt_var

        if scale_res != '':
            jmr = JMR(seed=123)
            jet = jmr.get(jet, scale=scale_res[0], res=scale_res[1])
    
    return jet


def getFatjets(events, corrector, pt_var='', scale_res = '', delphes=False):
    """takes a set of events and a set of weights and
    weights the fatjet pts of those events according to
    the variation given."""
    
    if delphes == False:
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

        if scale_res != '':
            jmr = JMR(seed=123)
            fatjet = jmr.get(fatjet, scale=scale_res[0], res=scale_res[1])

        fatjet['tau1'] = events.fatjet_tau1
        fatjet['tau2'] = events.fatjet_tau2
        fatjet['tau3'] = events.fatjet_tau3
        fatjet['tau4'] = events.fatjet_tau4
        
    elif delphes == True:
        fatjet = get_four_vec_fromPtEtaPhiM(
            None,
            pt = events.JetPUPPIAK8.PT,
            eta = events.JetPUPPIAK8.Eta,
            phi = events.JetPUPPIAK8.Phi,
            M = events.JetPUPPIAK8.SoftDroppedJet.mass,        #Using softdrop from now on
            copy = False,
        )

        fatjet_pt_var = corrector.get(fatjet, pt_var)

        fatjet['pt'] = fatjet_pt_var

        if scale_res != '':
            jmr = JMR(seed=123)
            fatjet = jmr.get(fatjet, scale=scale_res[0], res=scale_res[1])

        fatjet['tau1'] = events.JetPUPPIAK8.Tau_5[:,:,0]
        fatjet['tau2'] = events.JetPUPPIAK8.Tau_5[:,:,1]
        fatjet['tau3'] = events.JetPUPPIAK8.Tau_5[:,:,2]
        fatjet['tau4'] = events.JetPUPPIAK8.Tau_5[:,:,3]
    
    return fatjet
