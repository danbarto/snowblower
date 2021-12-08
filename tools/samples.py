'''
Get x-secs. Needs a working cmsenv, like CMSSW_10_6_19

'''


import imp, os, sys
import subprocess, shutil
import glob
import uproot

## default cmsRun cfg file
defaultCFG = """
import FWCore.ParameterSet.Config as cms
process = cms.Process("GenXSec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring('{FILEPATH}') )
process.dummy = cms.EDAnalyzer("GenXSecAnalyzer", genFilterInfoTag = cms.InputTag("genFilterEfficiencyProducer") )
process.p = cms.Path(process.dummy)"""

cfgFile     = 'xsecCfg.py'
identifier  = "After filter: final cross section ="

def das_wrapper(DASname, query='file'):
    sampleName = DASname.rstrip('/')

    dbs='dasgoclient -query="%s dataset=%s"'%(query, sampleName)
    dbsOut = os.popen(dbs).readlines()
    dbsOut = [ l.replace('\n','') for l in dbsOut ]
    return dbsOut

def gfal_wrapper(path, abs_path=True):
    cmd='eval `scram unsetenv -sh`; gfal-ls %s'%(path)
    out = os.popen(cmd).readlines()
    #print (out)
    if abs_path:
        out = [ path + '/' + l.replace('\n','') for l in out ]
    else:
        out = [ l.replace('\n','') for l in out ]
    return out

def get_xsec(f_in):
    replaceString = {'FILEPATH': f_in}
    cmsCfgString = defaultCFG.format( **replaceString )
    
    cmsRunCfg = open(cfgFile, 'w')
    cmsRunCfg.write(cmsCfgString)
    cmsRunCfg.close()

    p = subprocess.Popen(['cmsRun', cfgFile], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.stderr.readlines()
    for line in output:
        #print line
        if line.startswith(identifier): result = line

    xsec, unc = float(result.split(identifier)[1].split('+-')[0]), float(result.split(identifier)[1].split('+-')[1].replace('pb',''))

    return xsec, unc

def get_sample_info(name, delphes_path='root://cmseos.fnal.gov//store/user/snowmass/Snowmass2021/'):
    '''
    For a given sample get
    - files for delphes
    - files for flat ntuples
    - GEN sample
    - x-sec
    '''
    results = {}
    results['delphes'] = gfal_wrapper(delphes_path + '/Delphes/' + name)
    results['ntuples'] = gfal_wrapper(delphes_path + '/DelphesNtuplizer/' + name)
    
    snowmass_das = '/%s/Snowmass*/GEN'%(name.strip('_200PU'))
    phase2_das = '/%s/PhaseII*/GEN'%(name.strip('_200PU'))
    results['gen'] = das_wrapper(snowmass_das, query='') + das_wrapper(phase2_das, query='')

    gen_files = das_wrapper(results['gen'][0], query='file')
    xsec, unc = get_xsec(gen_files[0])
    results['xsec'] = xsec
    results['xsec_sigma'] = unc

    return results

def get_nevents(ntuples, treename='myana/mytree'):
    nevents = 0
    for i, ntuple in enumerate(ntuples):
        print (i/len(ntuples), ntuple)
        with uproot.open(ntuple)[treename] as tree:
            # uproot finally has context management!
            nevents += len(tree.arrays(['metpuppi_pt']))
    return nevents

def chunk(in_list, n):
    return [in_list[i * n:(i + 1) * n] for i in range((len(in_list) + n - 1) // n )]

def xrdcp(source, target):
    cmd = ['xrdcp', '-f', source, target]
    print ("Running cmd: %s"%(" ".join(cmd)))
    #cmd = ['xrdcp', source, target]
    subprocess.call(cmd)
    return os.path.isfile(target)

def hadd(f_out, f_in):
    cmd = ['hadd', f_out] + f_in
    subprocess.call(cmd)
    return os.path.isfile(f_out)

def copy_and_merge(name, target, files, n_chunks):

    if not os.path.isdir(target):
        os.makedirs(target)

    chunks = chunk(files, n_chunks)
    for i, files in enumerate(chunks):
        to_merge = []
        if not os.path.isfile(target+'/'+name+'_%s.root'%i):
            for f in files:
                f_name = f.split('/')[-1]
                success = xrdcp(f, target+'/'+f_name)
                if success:
                    print ("Copy successful.")
                    to_merge.append(target + f_name)
            print ("Now hadding.")
            success = hadd(target+'/'+name+'_%s.root'%i, to_merge)
            if success:
                for f in to_merge:
                    os.remove(f)
        else:
            print ("Output for job %s already exists, skipping."%i)
    return True


if __name__ == '__main__':

    import yaml
    from yaml import Loader, Dumper

    test_job = False

    if test_job:

        all_samples = gfal_wrapper('root://cmseos.fnal.gov//store/user/snowmass/Snowmass2021/Delphes/', abs_path=False)
        print ("Found the following samples:")
        print (all_samples)
        test = get_sample_info(all_samples[0])

    else:

        backgrounds = [
            'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            #'ST_tW_antitop_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU',
            #'ST_tW_top_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU',
            #'ST_tch_14TeV_antitop_incl-powheg-pythia8-madspin_200PU',
            #'ST_tch_14TeV_top_incl-powheg-pythia8-madspin_200PU',
            ##'TTJets_DiLept_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            ##'TTZToLLNuNu_M-10_TuneCP5_14TeV-amcatnlo-pythia8_200PU',
            #'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',
            #'VVTo2L2Nu_14TeV_amcatnloFXFX_madspin_pythia8_200PU',
            'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
            'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU',
            'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU',
            #'ttHTobb_M125_TuneCUETP8M2_14TeV-powheg-pythia8_200PU',
        ]

        try:
            with open('../data/samples.yaml', 'r') as f:
                database = yaml.load(f, Loader=Loader)
        except IOError:
            database = {}

        for bkg in backgrounds:
            print ("Working on %s"%bkg)
            if bkg not in database.keys():
                database[bkg] = get_sample_info(bkg)

                with open('../data/samples.yaml', 'w') as f:
                    yaml.dump(database, f, Dumper=Dumper)


        ## Hacking in stupid samples that are somewhere else...
        overwrite = False
        remote = True
        if not 'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU' in database.keys() or overwrite:
            if remote:
                with open('../data/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU.log', 'r') as f:
                    lines = f.readlines()
                tt_files = [ l.replace('\n', '') for l in lines ]
            else:
                tt_files = glob.glob('/nfs-7/userdata/dspitzba/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU/*.root')
            database['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU'] = {}
            database['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']['delphes'] = []
            database['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']['ntuples'] = tt_files
            database['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']['xsec'] = 900 # FIXME

            with open('../data/samples.yaml', 'w') as f:
                    yaml.dump(database, f, Dumper=Dumper)

    if True:
        for sample in database.keys():
        #for sample in ['QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU']:
            try:
                nevents = database[sample]['nevents']
            except KeyError:
                database[sample]['nevents'] = get_nevents(database[sample]['ntuples'])

                with open('../data/samples.yaml', 'w') as f:
                    yaml.dump(database, f, Dumper=Dumper)

    if True:
        for sample in backgrounds + ['TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU']:
            database[sample]['skim'] = gfal_wrapper('root://eoshome.cern.ch//eos/user/d/dspitzba/snowblower_data/%s_v12/'%sample)
            #print (sample)
            # NOTE need to get the skim directory.
            # Something like
            # gfal-ls root://eoshome.cern.ch//eos/user/d/dspitzba/snowblower_data/QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU_v12/
            # Then verify that they are actually working


    test_merge = False
    if test_merge:
        # We can hadd ~4 delphes samples, and ~20 ntuple files
        #sample_name = 'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU'
        sample_name = 'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU'
        copy_and_merge(
            sample_name,
            '/nfs-7/userdata/dspitzba/%s/'%sample_name,
            database[sample_name]['ntuples'],
            15,
        )
