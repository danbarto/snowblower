#!/bin/bash

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


function stageout {
    COPY_SRC=$1
    COPY_DEST=$2
    retries=0
    COPY_STATUS=1
    until [ $retries -ge 3 ]
    do
        #echo "Stageout attempt $((retries+1)): env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}"
        echo "Stageout attempt $((retries+1))"
        env -i X509_USER_PROXY=${X509_USER_PROXY} xrdcp -f ${COPY_SRC} ${COPY_DEST}
        #env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}
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

function setup_cmssw {
  CMSSW=$1
  export SCRAM_ARCH=$2
  scram p CMSSW $CMSSW
  cd $CMSSW
  eval $(scramv1 runtime -sh)
  cd -
}

function activate_cmssw {
  CMSSW=$1
  export SCRAM_ARCH=$2
  cd $CMSSW
  eval $(scramv1 runtime -sh)
  cd -
}

echo "Starting job on " `date`
echo "Running on " `uname -a`
echo "System release " `cat /etc/redhat-release`
echo
echo "-------------- ARGUMENTS ------------"
echo "Input file: ${INPUTFILENAMES}"
echo "XROOTD setting: ${URL}"

# setup cmssw
setup_cmssw $CMSSWVERSION $SCRAMARCH

echo "Current directory:"
echo `pwd`

IFS=',' read -r -a array <<< "$INPUTFILENAMES"
PREFIX="root://eosproject.cern.ch/"
array=( "${array[@]/#/${PREFIX}}" )
INPUTFILENAMES=$(IFS=' '; echo "${array[*]}")

echo "Hadding the following files:"
echo $INPUTFILENAMES

hadd ${OUTPUTNAME}.root $INPUTFILENAMES

echo "After running"
pwd
ls -la

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# copy output to eos

echo "Final output path for xrootd:"
echo ${OUTPUTDIR}

# we need to copy ${OUTPUTNAME} and ${NTUPLE}
COPY_SRC="file://`pwd`/${OUTPUTNAME}.root"
COPY_DEST=" root://eosproject.cern.ch//${OUTPUTDIR}/${OUTPUTNAME}_${IFILE}.root"
stageout $COPY_SRC $COPY_DEST


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
endTime=`date +%s`
echo "Total runtime (m): " `expr $endTime / 60 - $startTime / 60`

echo "removing inputs from condor"
rm -f ${OUTPUTNAME}.root
