'''
Just a collection of useful functions
Most of these functions need to be updated for awkward1.
'''
import pandas as pd
import numpy as np
import awkward as ak

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import os
import shutil
import math
import copy

import glob

data_path = os.path.expandvars('$TWHOME/data/')

def choose(first, n=2):
    tmp = ak.combinations(first, n)
    combs = tmp['0']
    for i in range(1,n):
        combs = combs.__add__(tmp[str(i)])
    for i in range(n):
        combs[str(i)] = tmp[str(i)]
    return combs

def cross(first, second):
    tmp = ak.cartesian([first, second])
    combs = (tmp['0'] + tmp['1'])
    combs['0'] = tmp['0']
    combs['1'] = tmp['1']
    return combs

def cutflow_scale_and_merge(cutflow, samples, fileset, lumi=3000):
    """
    Scale cutflow to a physical cross section.
    Merge sample cutflows into categories, e.g. several ttZ cutflows into one ttZ category.
    
    cutflow -- output coffea accumulator
    samples -- samples dictionary that contains the x-sec and sumWeight
    fileset -- fileset dictionary used in the coffea processor
    lumi -- integrated luminosity in 1/fb
    """
    mapping = {
        'ZJetsToNuNu_HT': [
            'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU',
        ],
        'WJetsToLNu_Njet': [
            'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        ],
        #'WJetsToLNu_Njet2': ['W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2', 'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2', 'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2', 'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_2'],
        'QCD_bEnriched_HT': [
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        ],
        'TT': [
            'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',
        ],
        '2HDMa_1500_150': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500'
        ],
        '2HDMa_1500_750': [
                '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500',
                '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500'
        ],
        '2HDMa_1750_750': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_1750_MH4_750_MH2_1750_MHC_1750'
        ],
        '2HDMa_2000_750': [
            '2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000', 
            '2HDMa_gg_sinp_0.35_tanb_1.0_mXd_10_MH3_2000_MH4_750_MH2_2000_MHC_2000'
        ],
    }
    
    merged = {}
   
    for group in mapping:
        combined = {}
        for sample in mapping[group]:
            if sample in fileset:
                temp = cutflow[sample].copy()
                scale = lumi*1000*samples[sample]['xsec']/samples[sample]['nevents']
                # scale according to cross sections
                for key in temp.keys():
                    temp[key] *= scale
                combined += temp
        merged[group] = combined
                   
    return merged

def get_samples(f_in='samples.yaml'):
    with open(data_path+f_in) as f:
        return load(f, Loader=Loader)

def get_scheduler_address():
    with open(os.path.expandvars('$TWHOME/scheduler_address.txt'), 'r') as f:
        lines = f.readlines()
        scheduler_address = lines[0].replace('\n','')
    return scheduler_address

def getName( DAS ):
    split = DAS.split('/')
    if split[-1].count('AOD'):
        return '_'.join(DAS.split('/')[1:3])
    else:
        return '_'.join(DAS.split('/')[-3:-1])
        #return'dummy'
        
def get_weight(effs, pt, eta):
    # NOTE need to load the efficiencies. need 2d yahist lookup.
    # needs to be done individually for each class of jets
    return yahist_2D_lookup(
        effs,
        pt,
        abs(eta),
        )

def dasWrapper(DASname, query='file'):
    sampleName = DASname.rstrip('/')

    dbs='dasgoclient -query="%s dataset=%s"'%(query, sampleName)
    dbsOut = os.popen(dbs).readlines()
    dbsOut = [ l.replace('\n','') for l in dbsOut ]
    return dbsOut

def finalizePlotDir( path ):
    path = os.path.expandvars(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    shutil.copy( os.path.expandvars( '$TWHOME/Tools/php/index.php' ), path )

def pad_and_flatten(val): 
    import awkward as ak
    try:
        return ak.flatten(ak.fill_none(ak.pad_none(val, 1, clip=True), 0))
        #return val.pad(1, clip=True).fillna(0.).flatten()#.reshape(-1, 1)
    except ValueError:
        return ak.flatten(val)

def yahist_1D_lookup(h, ar):
    '''
    takes a yahist 1D histogram (which has a lookup function) and an awkward array.
    '''
    return ak.unflatten(
        h.lookup(
            ak.to_numpy(ak.flatten(ar)) 
        ), ak.num(ar) )

def yahist_2D_lookup(h, ar1, ar2):
    '''
    takes a yahist 2D histogram (which has a lookup function) and an awkward array.
    '''
    return ak.unflatten(
        #np.nan_to_num(  # this is slow, but we have to take care of NaN
            h.lookup(
                ak.to_numpy(ak.flatten(ar1)),
                ak.to_numpy(ak.flatten(ar2)),
        #        ),
        #    0,
        ), ak.num(ar1) )

def build_weight_like(weight, selection, like):
    return ak.flatten(weight[selection] * ak.ones_like(like[selection]))

def fill_multiple(hist, datasets=[], arrays={}, selections=[], weights=[]):
    for i, dataset in enumerate(datasets):
        kw_dict = {'dataset': dataset, 'weight':weights[i]}
        kw_dict.update({x:arrays[x][selections[i]] for x in arrays.keys()})
        hist.fill(**kw_dict)

def getCutFlowTable(output, processes, lines, significantFigures=3, absolute=True, signal=None, total=False):
    '''
    Takes the output of a coffea processor (i.e. a python dictionary) and returns a formated cut-flow table of processes.
    Lines and processes have to follow the naming of the coffea processor output.
    '''
    res = {}
    eff = {}
    for proc in processes:
        res[proc] = {line: "%s +/- %s"%(round(output[proc][line], significantFigures-len(str(int(output[proc][line])))), round(math.sqrt(output[proc][line+'_w2']), significantFigures-len(str(int(output[proc][line]))))) for line in lines}
        
        # for efficiencies. doesn't deal with uncertainties yet
        eff[proc] = {lines[i]: round(output[proc][lines[i]]/output[proc][lines[i-1]], significantFigures) if (i>0 and output[proc][lines[i-1]]>0) else 1. for i,x in enumerate(lines)}
    
    if total:
        res['total'] = {line: "%s"%round( sum([ output[proc][line] for proc in total ] ), significantFigures-len(str(int(sum([ output[proc][line] for proc in total ] ))))) for line in lines }
    
    # if a signal is specified, calculate S/B
    if signal is not None:
        backgrounds = copy.deepcopy(processes)
        for s in signal:
            backgrounds.remove(s)
        res['S/B'] = {line: round( sum([output[s][line] for s in signal])/sum([ output[proc][line] for proc in backgrounds ]) if sum([ output[proc][line] for proc in backgrounds ])>0 else 1, significantFigures) for line in lines }
            
    if not absolute:
        res=eff
    df = pd.DataFrame(res)
    df = df.reindex(lines) # restores the proper order
    return df
        
def get_four_vec(cand):
    from coffea.nanoevents.methods import vector
    ak.behavior.update(vector.behavior)

    vec4 = ak.zip(
        {
            "pt": cand.pt,
            "eta": cand.eta,
            "phi": cand.phi,
            "mass": cand.mass,
        },
        with_name="PtEtaPhiMLorentzVector",
    )
    vec4.__dict__.update(cand.__dict__)
    return vec4

def get_four_vec_fromPtEtaPhiM(cand, pt, eta, phi, M, copy=True):
    '''
    Get a LorentzVector from a NanoAOD candidate with custom pt, eta, phi and mass
    All other properties are copied over from the original candidate
    '''
    from coffea.nanoevents.methods import vector
    ak.behavior.update(vector.behavior)

    vec4 = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": M,
        },
        with_name="PtEtaPhiMLorentzVector",
    )
    if copy:
        vec4.__dict__.update(cand.__dict__)
    return vec4

def scale_four_vec(vec, pt=1, eta=1, phi=1, mass=1):
    from coffea.nanoevents.methods import vector
    ak.behavior.update(vector.behavior)

    vec4 = ak.zip(
        {
            "pt": vec.pt*pt,
            "eta": vec.eta*eta,
            "phi": vec.phi*phi,
            "mass": vec.mass*mass,
        },
        with_name="PtEtaPhiMLorentzVector",
    )
    vec4.__dict__.update(cand.__dict__)
    return vec4

def match(first, second, deltaRCut=0.4):
    drCut2 = deltaRCut**2
    combs = ak.cartesian([first, second], nested=True)
    return ak.any((delta_r2(combs['0'], combs['1'])<drCut2), axis=2)

def match_count(first, second, deltaRCut=0.4):
    drCut2 = deltaRCut**2
    combs = ak.cartesian([first, second], nested=True)
    return ak.sum((delta_r2(combs['0'], combs['1'])<drCut2), axis=2)

def mt(pt1, phi1, pt2, phi2):
    '''
    Calculate MT
    '''
    return np.sqrt( 2*pt1*pt2 * (1 - np.cos(phi1-phi2)) )

def delta_phi(first, second):
    return (first.phi - second.phi + np.pi) % (2 * np.pi) - np.pi

def delta_phi_alt(first, second):
    # my version, seems to be faster (and unsigned)
    return np.arccos(np.cos(first.phi - second.phi))

def delta_r2(first, second):
    return (first.eta - second.eta)**2 + delta_phi_alt(first, second)**2

def delta_r(first, second):
    return np.sqrt(delta_r2(first, second))
