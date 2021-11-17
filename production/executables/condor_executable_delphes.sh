#!/bin/bash

#################################### Wrapper submit script for Upgrade production 
#Written by Alexis Kalogeropoulos - July 2014
#Adapted by Julie Hogan - summer 2016, jmhogan@fnal.gov

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700

startTime=`date +%s`

# Condor arguments
OUTPUTDIR=$1
OUTPUTNAME=$2
INPUTFILENAMES=$3
IFILE=$4
CMSSWVERSION=$5
SCRAMARCH=$6
CARD=$7
MAXEVT=$8

NTUPLE=${OUTPUTNAME}_ntuple


function stageout {
    COPY_SRC=$1
    COPY_DEST=$2
    retries=0
    COPY_STATUS=1
    until [ $retries -ge 3 ]
    do
        echo "Stageout attempt $((retries+1)): env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}"
        env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}
        COPY_STATUS=$?
        if [ $COPY_STATUS -ne 0 ]; then
            echo "Failed stageout attempt $((retries+1))"
        else
            echo "Successful stageout with $retries retries"
            break
        fi
        retries=$[$retries+1]
        echo "Sleeping for 30m"
        sleep 30m
    done
    if [ $COPY_STATUS -ne 0 ]; then
        echo "Removing output file because gfal-copy crashed with code $COPY_STATUS"
        env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-rm --verbose ${COPY_DEST}
        REMOVE_STATUS=$?
        if [ $REMOVE_STATUS -ne 0 ]; then
            echo "Uhh, gfal-copy crashed and then the gfal-rm also crashed with code $REMOVE_STATUS"
            echo "You probably have a corrupt file sitting on hadoop now."
            exit 1
        fi
    fi
}



echo "Starting job on " `date`
echo "Running on " `uname -a`
echo "System release " `cat /etc/redhat-release`
echo
echo "-------------- ARGUMENTS ------------"
echo "Card name: ${CARD}"
echo "Input file: ${INPUTFILENAMES}"
echo "XROOTD setting: ${URL}"

echo "Setting SkipEvents to 0, no argument given"
SKIPEVT=0

DOTUPLES=1
echo "SkipEvents: ${SKIPEVT}"
echo "Run ntuples? ${DOTUPLES}"
echo "----------------------------------------"
echo 

# Set variables
energy=14
DelphesVersion=tags/3.5.0
nPU=`echo $CARD | cut -d '_' -f 3 | cut -d '.' -f 1`
process=`echo $OUTPUTNAME | cut -d '/' -f -1 | cut -d '_' -f 1-3`

# Copy and unpack the tarball
echo "xrdcp source tarball and pileup file"
xrdcp -f root://eoscms.cern.ch//store/group/upgrade/RTB/delphes_tarballs/Delphes350_NtuplizerV0.tar tarball.tar
XRDEXIT=$?
if [[ $XRDEXIT -ne 0 ]]; then
    echo "exit code $XRDEXIT, failure in xrdcp of Delphes tarball"
    exit $XRDEXIT
fi

tar -xf tarball.tar
rm -f tarball.tar 
cd DelphesNtuplizer/CMSSW_10_0_5

eval `scram runtime -sh`
cd ../

# Copy in the MinBias file
xrdcp -f root://cmseos.fnal.gov//store/user/snowmass/DelphesSubmissionLPCcondor/MinBias_100k.pileup delphes/
XRDEXIT=$?
if [[ $XRDEXIT -ne 0 ]]; then
    echo "exit code $XRDEXIT, failure in xrdcp of MinBias_100k.pileup"
    exit $XRDEXIT
fi

setupTime=`date +%s`

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#run MiniAOD through Delphes

echo "Prepping card"
sed -i "s|SKIPEVENTS|${SKIPEVT}|g" cards/$CARD
sed -i "s|MAXEVENTS|${MAXEVT}|g" cards/$CARD

grep 'SkipEvents' cards/$CARD
grep 'MaxEvents' cards/$CARD

echo "Compiling delphes, will take a few minutes..."
cd delphes/
make >& /dev/null

compileTime=`date +%s`

echo "Running delphes with DelphesNtuplizer/cards/$CARD"

./DelphesCMSFWLite ../cards/$CARD ${OUTPUTNAME}.root ${INPUTFILENAMES}
DELPHESEXIT=$?
if [[ $DELPHESEXIT -ne 0 ]]; then
    echo "exit code $DELPHESEXIT, failure in DelphesCMSFWLite (maybe from xrootd)"
    exit $DELPHESEXIT
fi

DelphesTime=`date +%s`

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Run DelphesNtuplizer

if [[ $DOTUPLES == 1 ]] ; then
    echo "Running Delphes Ntuplizer on $OUTPUTNAME to produce $NTUPLE"
    python ../bin/Ntuplizer.py -i ${OUTPUTNAME}.root -o ${NTUPLE}.root
fi
NtupleTime=`date +%s`

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#metadata

echo "--------------- METADATA ---------------"
echo "User: " `eval whoami`
echo "Date: " `date` 
echo 

echo "Process: " $process 
echo "Pileup Conditions: " $nPU 
echo "Energy: " $energy 
echo 

echo "Input MiniAOD: " $INPUTFILENAMES
echo 

echo "Delphes Output: " ${OUTPUTNAME}.root
echo "Delphes Version: " $DelphesVersion 
echo "Detector Card: " $CARD 
echo 

echo "Minutes spent setting up job: " `expr $setupTime / 60 - $startTime / 60` 
echo "Minutes spent compiling Delphes: " `expr $compileTime / 60 - $setupTime / 60` 
echo "Minutes spent running Delphes: " `expr $DelphesTime / 60 - $compileTime / 60` 
echo "Minutes spent running Ntuplizer: " `expr $NtupleTime / 60 - $DelphesTime / 60`
echo "---------------------------------------------------"
echo 

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# copy output to eos

export REP="/store"
OUTPUTDIR="${OUTPUTDIR/\/hadoop\/cms\/store/$REP}"

echo "Final output path for xrootd:"
echo ${OUTPUTDIR}

# we need to copy ${OUTPUTNAME} and ${NTUPLE}

COPY_SRC="file://`pwd`/${OUTPUTNAME}.root"
COPY_DEST=" davs://redirector.t2.ucsd.edu:1094/${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root"
stageout $COPY_SRC $COPY_DEST

COPY_SRC="file://`pwd`/${NTUPLE}.root"
COPY_DEST=" davs://redirector.t2.ucsd.edu:1094/${OUTPUTDIR}/${NTUPLE}_${IFILE}.root"
stageout $COPY_SRC $COPY_DEST

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
endTime=`date +%s`
echo "Time spent copying output (s): " `expr $endTime - $NtupleTime`
echo "Total runtime (m): " `expr $endTime / 60 - $startTime / 60`

echo "removing inputs from condor"
rm -f ${OUTPUTNAME}.root
rm -f ${NTUPLE}.root
rm -f *.root MinBias_100k.pileup
