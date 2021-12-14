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

#def pad_and_flatten(val): 
#    import awkward as ak
#    try:
#        return ak.flatten(ak.fill_none(ak.pad_none(val, 1, clip=True), 0))
        #return val.pad(1, clip=True).fillna(0.).flatten()#.reshape(-1, 1)
#    except ValueError:
#        return ak.flatten(val)
    
def pad_and_flatten(val): 
    import awkward as ak
    try:
        return ak.flatten(ak.pad_none(val, 1, clip=True))
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

def delta_phi_alt_paf(first, second):
    return np.arccos(np.cos(pad_and_flatten(first.phi) - pad_and_flatten(second.phi)))
                     
def delta_r2_paf(first, second):
    return (pad_and_flatten(first.eta) - pad_and_flatten(second.eta))**2 + delta_phi_alt_paf(first, second)**2

def delta_r_paf(first, second):
    return np.sqrt(delta_r2_paf(first, second))
