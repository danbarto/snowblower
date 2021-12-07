//R__LOAD_LIBRARY($ROOTSYS/test/libEvent.so)

#include "TROOT.h"
#include "Riostream.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TString.h"
//#include "TSeq.h"

void skimmer(TString file_path="", TString out_path="")
{
   // Get old file, old tree and set top branch address
   //TString dir = "/nfs-7/userdata/dspitzba/ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU/ZJetsToNuNu_HT-200To400_14TeV-madgraph_200PU_68.root";
//    TString dir = "root://cmseos.fnal.gov//store/user/snowmass/Snowmass2021//DelphesNtuplizer/QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU/QCD_bEnriched_HT1000to1500_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_ntuple_167_1.root";
//    TString dir = "root://cmseos.fnal.gov//store/user/snowmass/Snowmass2021//DelphesNtuplizer/W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_200PU/W0JetsToLNu_TuneCUETP8M1_14TeV-madgraphMLM-pythia8_ntuple_194_0.root";
    TString dir = file_path;
   //gSystem->ExpandPathName(dir);
//   const auto filename = gSystem->AccessPathName(dir);


   std::unique_ptr<TFile> oldfile(TFile::Open(dir));
   //TFile oldfile(TFile::Open(dir));
   TTree *oldtree;
   oldfile->GetObject("myana/mytree", oldtree);

   const auto nentries = oldtree->GetEntries();

   oldtree->SetBranchStatus("*", 1);

   //int *true_int = 0;
   //float *metpuppi_pt;
   //Event *event = nullptr;
   //TBranch * b_true_int = oldtree->Branch("trueInteractions",&true_int,"trueInteractions/I");
   //oldtree->SetBranchAddress("trueInteractions", &true_int, &b_true_int);
   //oldtree->SetBranchAddress("trueInteractions", &true_int);
   //oldtree->SetBranchAddress("metpuppi_pt", &metpuppi_pt);

   // Create a new file + a clone of old tree in new file
   TFile newfile(out_path, "recreate");
   auto newtree = oldtree->CloneTree(0);

   for (int i=0; i < nentries; i++){
//   for (auto i : ROOT::TSeqI(nentries)) {
       oldtree->GetEntry(i);
       //cout << i << endl;
       //cout << oldtree->GetLeaf("metpuppi_pt")->GetValue() << endl;
       //
       // we want to skim to 0 leptons and met > 100?

       //int nbtagCSV=0;
       //
       int n_elec = oldtree->GetLeaf("elec_size")->GetValue();
       int n_muon = oldtree->GetLeaf("muon_size")->GetValue();

       bool pass = false;
       int n_elec_loose = 0;
       int n_muon_loose = 0;

       //for (auto i : ROOT::TSeqI(n_elec)) {
       for (int j=0; j < n_elec; j++) {
           if ( (oldtree->GetLeaf("elec_pt")->GetValue(j) > 10) && (oldtree->GetLeaf("elec_idpass")->GetValue(j) > 0)) {
               n_elec_loose++;
           }
       }

       //for (auto i : ROOT::TSeqI(n_muon)) {
       for (int k=0; k < n_muon; k++) {
           if ( (oldtree->GetLeaf("muon_pt")->GetValue(k) > 4) && (oldtree->GetLeaf("muon_idpass")->GetValue(k) > 0)) {
               n_muon_loose++;
           }
       }

       if ((n_elec_loose+n_muon_loose)==0 && oldtree->GetLeaf("metpuppi_pt")->GetValue() > 100) {
           pass = true;
       }

       if  (pass) {
           newtree->Fill();
       //    //event->Clear();
       }

   }

   newtree->Print();

   cout << "Skim efficiency:" << endl;
   cout << (newtree->GetEntries() / float(nentries)) << endl;

   newfile.Write();
}
