#include <TGraph.h>
#include <TGraphAsymmErrors.h>

using namespace ROOT;

void setGraphStyle(TGraph*, short, short);


// See http://techforcurious.website/cern-root-tutorial-2-plotting-graph-using-tgraph/
// See https://root.cern.ch/doc/v620/TGraphAsymmErrors_8cxx_source.html for constructor 
//TGraphAsymmErrors::TGraphAsymmErrors	(	const char * 	filename,
//const char * 	format = "%lg %lg %lg %lg %lg %lg",
//Option_t * 	option = "" 
//)	

void  plotTime() {
  gStyle->SetOptFit(0011);
  gStyle->SetOptStat(0000);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadBottomMargin(0.15);

  std::string filePath = "../../externals/gpuKFPerformance/data/timing";
  
  std::vector<std::string> fileNames={
  "Results_timing_Haswell_CustomInverter_OMP_NumThreads_1.csv",
  "Results_timing_Haswell_CustomInverter_OMP_NumThreads_60.csv",
  "Results_timing_Tesla_V100-SXM2-16GB_nStreams_1_gridSize_100000x1x1_blockSize_8x8x1_sharedMemory_0.csv",
  "Results_timing_Tesla_V100-SXM2-16GB_nStreams_1_gridSize_100000x1x1_blockSize_8x8x1_sharedMemory_1.csv",
  };
 
  std::vector<std::string> machines={
   "Haswell 1 thread",
   "Haswell 60 threads",
   "NVIDIA V100 1 track per thread",
   "NVIDIA V100 1 block per thread", 
  };

  std::vector<int> styles = {20, 20, 24, 24};
  std::vector<int> colors = {14, 6, 4, 3};
  
  if(fileNames.size()!=machines.size()){
    throw std::runtime_error("The fileNames size must be equal to the machines size!"); 
  }
  if(styles.size() != fileNames.size()){
    throw std::runtime_error("The styles size must be equal to the fileNames size!"); 
  }
  if(colors.size() != fileNames.size()){
    throw std::runtime_error("The colors size must be equal to the fileNames size!"); 
  }

  std::vector<TGraphAsymmErrors*> grs(fileNames.size());
 
  size_t ifile=0; 
  for(const auto& fileName : fileNames){
    std::string file = filePath+fileName;
    grs[ifile] = new TGraphAsymmErrors(file.c_str(), "%lg,%lg,%lg,%lg", ""); 
    setGraphStyle(grs[ifile], styles[ifile], colors[ifile]); 
    ifile++; 
  } 

   auto c1 = new TCanvas("c1","c1",
                   200,10,700,500);
   //c1->SetGrid();
   c1->SetLogx();
   c1->SetLogy();

   // create a multigraph and draw it
   TMultiGraph  *mg  = new TMultiGraph();
   for(const auto& gr : grs){
    mg->Add(gr);
   } 
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
   size_t igr=0; 
   for(const auto& gr : grs){
     legend->AddEntry(gr, machines[igr].c_str(), "lp");
     igr++; 
   } 
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
