'''
Get x-secs. Needs a working cmsenv, like CMSSW_10_6_19

'''


import imp, os, sys
import subprocess, shutil

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
        out = [ path+l.replace('\n','') for l in out ]
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


if __name__ == '__main__':
    
    all_samples = gfal_wrapper('root://cmseos.fnal.gov//store/user/snowmass/Snowmass2021/Delphes/', abs_path=False)

    print ("Found the following samples:")
    print (all_samples)

    test = get_sample_info(all_samples[0])

    

