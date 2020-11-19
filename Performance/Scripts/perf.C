#include <TGraph.h>

using namespace ROOT;

// See http://techforcurious.website/cern-root-tutorial-2-plotting-graph-using-tgraph/

void  perf() {
  gStyle->SetOptFit(0011);
  gStyle->SetOptStat(0000);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadBottomMargin(0.15);
	
  //std::string fileName = "data/perf.txt";
  std::string fileName = "data/perfNoSmearing.txt";

  TGraph* gr1 = new TGraph(fileName.c_str(), "%lg %lg %*lg %*lg",",;"); 
  TGraph* gr2 = new TGraph(fileName.c_str(), "%lg %*lg %lg %*lg",",;"); 
  TGraph* gr3 = new TGraph(fileName.c_str(), "%lg %*lg %*lg %lg",",;"); 
  gr1->SetMarkerStyle(20); 
  gr2->SetMarkerStyle(24); 
  gr3->SetMarkerStyle(24); 
 
  gr1->SetLineWidth(2);
  gr2->SetLineWidth(2);
  gr3->SetLineWidth(2);

  gr1->SetLineColor(1); 
  gr2->SetLineColor(2); 
  gr3->SetLineColor(4); 
  
  gr1->SetMarkerColor(1); 
  gr2->SetMarkerColor(2); 
  gr3->SetMarkerColor(4); 
 

   auto c1 = new TCanvas("c1","c1",
                   200,10,700,500);
   //c1->SetGrid();
   c1->SetLogx();
   c1->SetLogy();

   // create a multigraph and draw it
   TMultiGraph  *mg  = new TMultiGraph();
   mg->Add(gr1);
   mg->Add(gr2);
   mg->Add(gr3);
   mg->GetXaxis()->SetTitle("Number of Tracks");
   mg->GetYaxis()->SetTitle("Fitting time (ms)");
   mg->GetXaxis()->SetTitleSize(0.04);
   mg->GetYaxis()->SetTitleSize(0.04);
   mg->GetXaxis()->SetLabelSize(0.05); 
   mg->GetYaxis()->SetLabelSize(0.05); 
   mg->GetXaxis()->SetTitleOffset(1.7);
   mg->GetYaxis()->SetTitleOffset(1.4);
   mg->Draw("ALP");

   TLegend* legend = new TLegend(0.15, 0.7, 0.65, 0.9);
   legend->AddEntry(gr1, "Intel's i7-8559U (one thread)", "lp");
   legend->AddEntry(gr2, "Nvidia GTX 1650 1 thread per track", "lp");
   legend->AddEntry(gr3, "Nvidia GTX 1650 1 block per track", "lp");
   legend->SetBorderSize(0);
   legend->SetFillStyle(0);
   legend->SetTextFont(42);
   legend->Draw();
}
