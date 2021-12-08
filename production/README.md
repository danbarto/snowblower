# Produce Delphes samples

Make sure you have CMSSW set up and ProjectMetis cloned.
Then first run `source make_tarball.sh` to get a tarball containing the relevant psets for cmsRun.
To set the correct paths run `source setup.sh`.

Tasks are submitted with `ipython -i make_delphes.py`.


A custom version of ProjectMetis is needed to stage out on eos.

