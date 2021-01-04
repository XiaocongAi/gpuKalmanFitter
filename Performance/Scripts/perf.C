#include <TGraph.h>

using namespace ROOT;

void setGraphStyle(TGraph*, short, short);


// See http://techforcurious.website/cern-root-tutorial-2-plotting-graph-using-tgraph/
void  perf() {
  gStyle->SetOptFit(0011);
  gStyle->SetOptStat(0000);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadBottomMargin(0.15);
	
  std::string fileName = "data/perf.txt";

  TGraph* gr1 = new TGraph(fileName.c_str(), "%lg %*lg %*lg %lg %*lg %*lg %*lg %*lg %*lg",",;"); 
  TGraph* gr2 = new TGraph(fileName.c_str(), "%lg %*lg %*lg %*lg %lg %*lg %*lg %*lg %*lg",",;"); 
  TGraph* gr3 = new TGraph(fileName.c_str(), "%lg %*lg %*lg %*lg %*lg %*lg %*lg %lg %*lg",",;"); 
  TGraph* gr4 = new TGraph(fileName.c_str(), "%lg %*lg %*lg %*lg %*lg %*lg %*lg %*lg %lg",",;"); 
  
  setGraphStyle(gr1, 20, 14);
  setGraphStyle(gr2, 20, 6);
  setGraphStyle(gr3, 24, 4);
  setGraphStyle(gr4, 24, 3);

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
   mg->Add(gr4);
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
   legend->AddEntry(gr1, "Haswell (one thread)", "lp");
   legend->AddEntry(gr2, "Haswell (60 threads)", "lp");
   legend->AddEntry(gr3, "Nvidia V100 1 thread per track", "lp");
   legend->AddEntry(gr4, "Nvidia V100 1 block  per track", "lp");
   legend->SetBorderSize(0);
   legend->SetFillStyle(0);
   legend->SetTextFont(42);
   legend->Draw();
}

// function to set up the graph style
void
setGraphStyle(TGraph* graph, short markerStyle = 20, short color = 1)
{
  graph->GetXaxis()->SetNdivisions(505);
  graph->SetMarkerStyle(markerStyle);
  graph->SetMarkerSize(1.1);
  graph->SetLineWidth(2);
  graph->SetLineColor(color);
  graph->SetMarkerColor(color);
}
