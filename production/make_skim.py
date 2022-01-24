'''
Produce a nanoGEN sample on condor, using metis.
Inspired by https://github.com/aminnj/scouting/blob/master/generation/submit_jobs.py

'''

from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample, DummySample, FilelistSample
from metis.Path import Path
from metis.StatsParser import StatsParser
import time
import os

from yaml import Loader, Dumper
import yaml

def submit():

    with open('../data/samples.yaml', 'r') as f:
        samples = yaml.load(f, Loader = Loader)

    sample_names = list(samples.keys())

    sample_names = [
        'QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'QCD_bEnriched_HT1500to2000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'QCD_bEnriched_HT2000toInf_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'QCD_bEnriched_HT200to300_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'QCD_bEnriched_HT300to500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'QCD_bEnriched_HT500to700_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'QCD_bEnriched_HT700to1000_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU',
        'TT_Mtt1000toInf_TuneCUETP8M1_14TeV-powheg-pythia8_200PU',
        #'W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        #'W1JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        #'W2JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        #'W3JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'WJetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'WJetsToLNu_GenMET-100_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU',
        'ZJetsToNuNu_HT-100To200_14TeV-madgraph_200PU',
        'ZJetsToNuNu_HT-1200To2500_14TeV-madgraph_200PU',
        'ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU',
        'ZJetsToNuNu_HT-400To600_14TeV-madgraph_200PU',
        'ZJetsToNuNu_HT-600To800_14TeV-madgraph_200PU',
        'ZJetsToNuNu_HT-800To1200_14TeV-madgraph_200PU',
        'ZJetsToNuNu_HT2500toInf_HLLHC',
        #'tZq_nunu_4f_14TeV-amcatnlo-madspin-pythia8_200PU',
        'ZH_HToBB_ZToNuNu_M125_13TeV_powheg_pythia8_200PU',
        'WminusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU',
        'WplusH_HToBB_WToLNu_M125_14TeV_powheg_pythia8_200PU',
        'VVTo2L2Nu_14TeV_amcatnloFXFX_madspin_pythia8_200PU',
        #'TTZToLLNuNu_M-10_TuneCP5_14TeV-amcatnlo-pythia8_200PU',
        'ST_tch_14TeV_top_incl-powheg-pythia8-madspin_200PU',
        'ST_tch_14TeV_antitop_incl-powheg-pythia8-madspin_200PU',
        'ST_tW_top_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU',
        'ST_tW_antitop_5f_inclusiveDecays_14TeV-powheg-pythia8_TuneCUETP8M1_200PU',
        #'ST_s-channel_4f_InclusiveDecays_14TeV-amcatnlo-pythia8_200PU',
     ]


    total_summary = {}

    extra_requirements = "true"

    tag = "v16"
    # v15 - use for everything but tt+jets and W+jets
    # v16 - lepton veto removed in skim


    skim_tasks = []
    merge_tasks = []

    for s in sample_names:

        print ("Working on sample: %s"%s)

        sample = FilelistSample(
            dataset=s,
            filelist=samples[s]['ntuples'],
        )

        #print (sample.get_files())

        skim_task = CondorTask(
                sample = sample,
                output_name = "skim.root",
                #executable = "executables/condor_executable_skim_eos.sh",
                executable = "executables/condor_executable_skim.sh",
                #output_dir = "/eos/user/d/dspitzba/snowblower_data/%s_%s/"%(s, tag),
                tarfile = "package.tar.gz",
                open_dataset = False,
                files_per_output = 10,  # was 50 for everything but ttbar
                cmssw_version = "CMSSW_10_6_19",
                scram_arch = "slc7_amd64_gcc820",
                condor_submit_params = {
                    "sites":"T2_US_UCSD", # 
                    "classads": [
                        ["metis_extraargs",""],
                        ["JobBatchName",s],
                        #["SingularityImage", "/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel6-m202006"],
                        ],
                    "requirements_line": 'Requirements = (HAS_SINGULARITY=?=True)'  # && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                tag = tag,
                min_completion_fraction = 0.98,
                )

        skim_tasks.append(skim_task)

        merge_task = CondorTask(
                sample = DirectorySample(
                    dataset="merge_"+sample.get_datasetname(),
                    location=skim_task.get_outputdir(),
                    #use_xrootd = True,
                ),
                output_name = "merge.root",
                #executable = "executables/condor_executable_merge_eos.sh",
                executable = "executables/condor_executable_merge.sh",
                tarfile = "package.tar.gz",
                #output_dir = "/eos/user/d/dspitzba/snowblower_data/merge_%s_%s/"%(s, tag),
                open_dataset = False,
                files_per_output = 10,  # was 10 for everything but ttbar
                cmssw_version = "CMSSW_10_6_19",
                scram_arch = "slc7_amd64_gcc820",
                condor_submit_params = {
                    "sites":"T2_US_UCSD", # 
                    "classads": [
                        ["metis_extraargs",""],
                        ["JobBatchName",s],
                        #["SingularityImage", "/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel6-m202006"],
                        ],
                    "requirements_line": 'Requirements = (HAS_SINGULARITY=?=True)'  # && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                tag = tag,
                min_completion_fraction = 1.0,
                )

        merge_tasks.append(merge_task)



    total_summary = {}


    for skim_task,merge_task in zip(skim_tasks,merge_tasks):

        #if not os.path.isdir(skim_task.get_outputdir()):
        #    os.makedirs(skim_task.get_outputdir())
        #if not os.path.isdir(merge_task.get_outputdir()):
        #    os.makedirs(merge_task.get_outputdir())

        skim_task.process()

        frac = skim_task.complete(return_fraction=True)


        #print (test.get_files())

        if frac >= (skim_task.min_completion_fraction-0.0001):
            print ("merging now")
            merge_task.reset_io_mapping()
            merge_task.update_mapping()
            merge_task.process()
    

        total_summary[skim_task.get_sample().get_datasetname()] = skim_task.get_task_summary()
        total_summary[merge_task.get_sample().get_datasetname()] = merge_task.get_task_summary()

        StatsParser(data=total_summary, webdir="~/public_html/dump/HLLHC_skim/").do()

    
    #for gen_task, delphes_task in zip(gen_tasks,delphes_tasks):

    #    if not os.path.isdir(gen_task.get_outputdir()):
    #        os.makedirs(gen_task.get_outputdir())
    #    if not os.path.isdir(gen_task.get_outputdir()+'/delphes'):
    #        os.makedirs(gen_task.get_outputdir()+'/delphes')

    #    gen_task.process()
    #
    #    frac = gen_task.complete(return_fraction=True)
    #    if frac >= gen_task.min_completion_fraction:
    #        delphes_task.reset_io_mapping()
    #        delphes_task.update_mapping()
    #        delphes_task.process()
    #
    #    total_summary[gen_task.get_sample().get_datasetname()] = gen_task.get_task_summary()
    #    total_summary[delphes_task.get_sample().get_datasetname()] = delphes_task.get_task_summary()


    #    StatsParser(data=total_summary, webdir="~/public_html/dump/HLLHC_GEN/").do()

if __name__ == "__main__":

    for i in range(500):
        submit()
        nap_time = 0.2
        time.sleep(60*60*nap_time)  # take a super-long power nap

