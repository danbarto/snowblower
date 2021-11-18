'''
Produce a nanoGEN sample on condor, using metis.
Inspired by https://github.com/aminnj/scouting/blob/master/generation/submit_jobs.py

'''

from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample,DummySample
from metis.Path import Path
from metis.StatsParser import StatsParser
import time
import os


def submit():

    requests = {
        '2HDMa_bb_1500_150_10_test': '/hadoop/cms/store/user/ewallace/gridpacks/2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_150_MH2_1500_MHC_1500_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
        #'2HDMa_bb_MH4_250': '/hadoop/cms/store/user/ewallace/gridpacks/2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_250_MH2_1500_MHC_1500_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
	#'2HDMa_bb_MH4_350': '/hadoop/cms/store/user/ewallace/gridpacks/2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_350_MH2_1500_MHC_1500_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
	#'2HDMa_bb_MH4_500': '/hadoop/cms/store/user/ewallace/gridpacks/2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_500_MH2_1500_MHC_1500_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
	#'2HDMa_bb_MH4_750': '/hadoop/cms/store/user/ewallace/gridpacks/2HDMa_bb_sinp_0.35_tanb_1.0_mXd_10_MH3_1500_MH4_750_MH2_1500_MHC_1500_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz',
    }

    total_summary = {}

    extra_requirements = "true"

    tag = "v7"
    #events_per_point = 500000 # produced 500k events before
    #events_per_job = 2000 # up to 2000 works
    events_per_point = 50
    events_per_job = 10
    njobs = int(events_per_point)//events_per_job

    gen_tasks = []
    delphes_tasks = []

    for reqname in requests:
        gridpack = requests[reqname]

        #reqname = "TTWJetsToLNuEWK_5f_EFT_myNLO"
        #gridpack = '/hadoop/cms/store/user/dspitzba/tW_scattering/gridpacks/TTWJetsToLNuEWK_5f_EFT_myNLO_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz'

        sample = DummySample(dataset="/%s/HLLHC/GEN"%reqname,N=njobs,nevents=int(events_per_point))

        gen_task = CondorTask(
                sample = sample,
                output_name = "gen.root",
                executable = "executables/condor_executable_gen.sh",
                tarfile = "package.tar.gz",
                #additional_input_files = gridpack,
                #scram_arch = "slc7_amd64_gcc630",
                open_dataset = False,
                files_per_output = 1,
                arguments = gridpack,
                condor_submit_params = {
                    "sites":"T2_US_UCSD", # 
                    "classads": [
                        ["param_nevents",events_per_job],
                        ["metis_extraargs",""],
                        ["JobBatchName",reqname],
                        #["SingularityImage", "/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel6-m202006"],
                        ],
                    "requirements_line": 'Requirements = (HAS_SINGULARITY=?=True)'  # && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                tag = tag,
                min_completion_fraction = 0.90,
                )

        gen_tasks.append(gen_task)

        delphes_task = CondorTask(
            sample = DirectorySample(
                dataset="delphes_"+sample.get_datasetname(),
                location=gen_task.get_outputdir(),
            ),
            executable = "executables/condor_executable_delphes.sh",
            arguments = "CMS_PhaseII_200PU_Snowmass2021_v0.tcl %s"%events_per_job,
            files_per_output = 1,
            output_dir = gen_task.get_outputdir() + "/delphes",
            output_name = "delphes.root",
            output_is_tree = True,
            tag = tag,
            condor_submit_params = {"sites":"T2_US_UCSD,UAF"},
            cmssw_version = "CMSSW_10_0_5",
            scram_arch = "slc7_amd64_gcc700",
            min_completion_fraction = 0.90,
        )
    
        delphes_tasks.append(delphes_task)
    

    total_summary = {}
        
    
    for gen_task, delphes_task in zip(gen_tasks,delphes_tasks):

        if not os.path.isdir(gen_task.get_outputdir()):
            os.makedirs(gen_task.get_outputdir())
        if not os.path.isdir(gen_task.get_outputdir()+'/delphes'):
            os.makedirs(gen_task.get_outputdir()+'/delphes')

        gen_task.process()
    
        frac = gen_task.complete(return_fraction=True)
        if frac >= gen_task.min_completion_fraction:
            delphes_task.reset_io_mapping()
            delphes_task.update_mapping()
            delphes_task.process()
    
        total_summary[gen_task.get_sample().get_datasetname()] = gen_task.get_task_summary()
        total_summary[delphes_task.get_sample().get_datasetname()] = delphes_task.get_task_summary()


        StatsParser(data=total_summary, webdir="~/public_html/dump/HLLHC_GEN/").do()

if __name__ == "__main__":

    for i in range(500):
        submit()
        nap_time = 1
        time.sleep(60*60*nap_time)  # take a super-long power nap

