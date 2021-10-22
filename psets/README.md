# Event generation

## Gridpacks

We use the central [genproductions](https://github.com/cms-sw/genproductions/) repository to generate gridpacks.
The beam energy and/or center of mass energy have to be adjusted for HL-LHC (7 TeV or 14 TeV, respectively).

## NanoGEN

We can run NanoGEN for quick studies, using generator truth objects.
The `nanogen_cfg.py` config should produce events for LO samples without extra partons.
For other usecases new psets should be created.
Run with `cmsRun nanogen_cfg.py` in `CMSSW_10_6_19` or higher.

## GEN and Delphes

Create a config with the `fragment.py`:

```
export SCRAM_ARCH=slc7_amd64_gcc820
cmsrel CMSSW_11_0_2
cd CMSSW_11_0_2/src/
cmsenv
mkdir -p Configuration/GenProduction/python/
cp ../../fragment.py Configuration/GenProduction/python/
scram b -j 8

cmsDriver.py Configuration/GenProduction/python/fragment.py --fileout file:output_gen.root --mc --eventcontent RAWSIM,LHE --datatier GEN,LHE --conditions 110X_mcRun4_realistic_v3 --beamspot HLLHC14TeV --step LHE,GEN --geometry Extended2026D49 --era Phase2C9 --python_filename gen_cfg.py -n 10 --no_exec
```

