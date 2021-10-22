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

WIP
