#!/usr/bin/env python3
import sys
import ROOT

ROOT.gROOT.ProcessLineSync('.L skimmer.C+')

f_in = sys.argv[1]
f_out = sys.argv[2]

ROOT.skimmer(f_in, f_out)
