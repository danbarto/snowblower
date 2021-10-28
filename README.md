# snowblower
Tools for the Snowmass process.

To setup:
```
git clone --recursive git@github.com:danbarto/snowblower.git

cd snowblower
cmsrel CMSSW_10_6_19
cd CMSSW_10_6_19/src/
git cms-init
cmsenv
scram b -j 8
cd ../../
```


To use the sample caches:
```
pip install klepto --user
```

