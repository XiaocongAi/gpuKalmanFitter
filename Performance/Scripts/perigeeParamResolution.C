#include <TF1.h>
#include <TH1F.h>
#include <TMath.h>
#include <TTree.h>
#include <iostream>
#include <map>
#include <vector>

using namespace ROOT;

void
setHistStyle(TH1F* hist, short color);

// To run this script, use e.g. "root -l 'perigeeParamResolution.C("../../build/Run/GPU/fitted_param_gpu_nTracks_10000.root","params")'"

void
perigeeParamResolution(const std::string& inFile,
          const std::string& treeName)
{
  gStyle->SetOptFit(0000);
  gStyle->SetOptStat(0000);
  gStyle->SetPadLeftMargin(0.20);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadBottomMargin(0.12);
 
  TF1 *gaussFit = new TF1("gaussFit", "gaus", -5, 5);
  gaussFit->SetLineColor(4); 
  
  // Open root file written by RootTrajectoryWriter
  std::cout << "Opening file: " << inFile << std::endl;
  TFile*  file = TFile::Open(inFile.c_str(), "read");
  std::cout << "Reading tree: " << treeName << std::endl;
  TTree* tree = (TTree*)file->Get(treeName.c_str());
 
  // Track parameter name
  std::vector<std::string> paramNames
      = {"loc1", "loc2", "#phi", "#theta", "q/p" ,"t"};

  // Residual range
  std::map<std::string, double> paramResidualRange = {{"loc1", 0.25},
                                                      {"loc2", 0.25},
                                                      {"#phi", 0.08},
                                                      {"#theta", 0.08},
                                                      {"q/p", 0.002},
                                                      {"t", 5000}};
  // Pull range
  double pullRange = 5;

  map<string, TH1F*> res_fit;
  map<string, TH1F*> pull_fit;

  // Create the hists and set up
  for (const auto& [par, resRange] : paramResidualRange) {
    // residual hists
    res_fit[par] = new TH1F(Form("res_%s", par.c_str()),
                            Form("residual of %s", par.c_str()),
                            100,
                            -1 * resRange,
                            resRange);

    // pull hists
    pull_fit[par] = new TH1F(Form("pull_%s", par.c_str()),
                             Form("pull of %s", par.c_str()),
                             100,
                             -1 * pullRange,
                             pullRange);

    res_fit[par]->GetXaxis()->SetTitle(Form("r(%s)", par.c_str()));
    res_fit[par]->GetYaxis()->SetTitle("Entries");
    pull_fit[par]->GetXaxis()->SetTitle(Form("pull(%s)", par.c_str()));
    pull_fit[par]->GetYaxis()->SetTitle("Entries/ 0.1");

    // set style
    setHistStyle(res_fit[par], 1);
    setHistStyle(pull_fit[par], 1);
  }

 int t_charge{0};
  float t_time{0};
  float t_vx{-99.};
  float t_vy{-99.};
  float t_vz{-99.};
  float t_px{-99.};
  float t_py{-99.};
  float t_pz{-99.};
  float t_theta{-99.};
  float t_phi{-99.};
  float t_pT{-99.};
  float t_eta{-99.};
  std::array<float, 6> params_fit = {-99, -99, -99, -99, -99, -99};
  std::array<float, 6> err_params_fit = {-99, -99, -99, -99, -99, -99};
  std::array<float, 6> res_params = {-99, -99, -99, -99, -99, -99};
  std::array<float, 6> pull_params = {-99, -99, -99, -99, -99, -99};

    tree->SetBranchAddress("t_charge", &t_charge);
    tree->SetBranchAddress("t_time", &t_time);
    tree->SetBranchAddress("t_vx", &t_vx);
    tree->SetBranchAddress("t_vy", &t_vy);
    tree->SetBranchAddress("t_vz", &t_vz);
    tree->SetBranchAddress("t_px", &t_px);
    tree->SetBranchAddress("t_py", &t_py);
    tree->SetBranchAddress("t_pz", &t_pz);
    tree->SetBranchAddress("t_theta", &t_theta);
    tree->SetBranchAddress("t_phi", &t_phi);
    tree->SetBranchAddress("t_eta", &t_eta);
    tree->SetBranchAddress("t_pT", &t_pT);

    tree->SetBranchAddress("eLOC0_fit", &params_fit[0]);
    tree->SetBranchAddress("eLOC1_fit", &params_fit[1]);
    tree->SetBranchAddress("ePHI_fit", &params_fit[2]);
    tree->SetBranchAddress("eTHETA_fit", &params_fit[3]);
    tree->SetBranchAddress("eQOP_fit", &params_fit[4]);
    tree->SetBranchAddress("eT_fit", &params_fit[5]);

    tree->SetBranchAddress("err_eLOC0_fit", &err_params_fit[0]);
    tree->SetBranchAddress("err_eLOC1_fit", &err_params_fit[1]);
    tree->SetBranchAddress("err_ePHI_fit", &err_params_fit[2]);
    tree->SetBranchAddress("err_eTHETA_fit", &err_params_fit[3]);
    tree->SetBranchAddress("err_eQOP_fit", &err_params_fit[4]);
    tree->SetBranchAddress("err_eT_fit", &err_params_fit[5]);

    tree->SetBranchAddress("res_eLOC0", &res_params[0]);
    tree->SetBranchAddress("res_eLOC1", &res_params[1]);
    tree->SetBranchAddress("res_ePHI", &res_params[2]);
    tree->SetBranchAddress("res_eTHETA", &res_params[3]);
    tree->SetBranchAddress("res_eQOP", &res_params[4]);
    tree->SetBranchAddress("res_eT", &res_params[5]);

    tree->SetBranchAddress("pull_eLOC0", &pull_params[0]);
    tree->SetBranchAddress("pull_eLOC1", &pull_params[1]);
    tree->SetBranchAddress("pull_ePHI", &pull_params[2]);
    tree->SetBranchAddress("pull_eTHETA", &pull_params[3]);
    tree->SetBranchAddress("pull_eQOP", &pull_params[4]);
    tree->SetBranchAddress("pull_eT", &pull_params[5]);


  Int_t entries = tree->GetEntries();
  for (int j = 0; j < entries; j++) {
    tree->GetEvent(j);
    for (unsigned int ipar = 0; ipar <  paramNames.size(); ipar++) {
      res_fit[paramNames.at(ipar)]->Fill(res_params[ipar]);
      pull_fit[paramNames.at(ipar)]->Fill(pull_params[ipar]);
    }
  }

  // plotting residual
  TCanvas* c1 = new TCanvas("c1", "c1", 1200, 800);
  c1->Divide(3, 2);
  for (size_t ipar = 0; ipar < paramNames.size(); ipar++) {
    c1->cd(ipar + 1);
    res_fit[paramNames.at(ipar)]->Draw("e");

    int binmax     = res_fit[paramNames.at(ipar)]->GetMaximumBin();
    int bincontent = res_fit[paramNames.at(ipar)]->GetBinContent(binmax);

    res_fit[paramNames.at(ipar)]->GetYaxis()->SetRangeUser(0, bincontent * 1.2);
  }

  // plotting pull
  TCanvas* c2 = new TCanvas("c2", "c2", 1200, 700);
  c2->Divide(3, 2);
  for (size_t ipar = 0; ipar < paramNames.size(); ipar++) {
    c2->cd(ipar + 1);
    pull_fit[paramNames.at(ipar)]->Draw("e");
     
    pull_fit[paramNames.at(ipar)]->Fit("gaussFit");
    if(ipar==0){
      pull_fit[paramNames.at(ipar)]->GetXaxis()->SetTitle("pull(d_{0})");
    } else if (ipar==1){
      pull_fit[paramNames.at(ipar)]->GetXaxis()->SetTitle("pull(z_{0})");
    }

    //gaussFit->Draw("same");
    
    auto mean = gaussFit->GetParameter(1);
    auto mean_err = gaussFit->GetParError(1);
    auto sigma = gaussFit->GetParameter(2);
    auto sigma_err = gaussFit->GetParError(2);
    //std::cout<<"mean = " << mean <<", sigma = " << sigma << std::endl;

    int binmax     = pull_fit[paramNames.at(ipar)]->GetMaximumBin();
    int bincontent = pull_fit[paramNames.at(ipar)]->GetBinContent(binmax);

    TLatex latex;
    latex.SetTextSize(0.045);
//    latex.DrawLatex(1.2, bincontent*1.2, Form("#font[2]{Mean = %.3f}", mean));
//    latex.DrawLatex(1.2, bincontent*1.1, Form("#font[2]{Sigma = %.2f}", sigma));
    latex.DrawLatex(-0.5, bincontent*1.2, Form("#mu = %.3f #pm %0.3f", mean, mean_err));
    latex.DrawLatex(-0.5, bincontent*1.1, Form("#sigma = %.2f #pm %0.2f", sigma, sigma_err));
    latex.SetTextFont(6);
    latex.Draw();

    pull_fit[paramNames.at(ipar)]->GetYaxis()->SetRangeUser(0,
                                                            bincontent * 1.3);

   
  }

  
}

// function to set up the histgram style
void
setHistStyle(TH1F* hist, short color = 1)
{
  hist->GetXaxis()->SetTitleSize(0.05);
  hist->GetYaxis()->SetTitleSize(0.05);
  hist->GetXaxis()->SetLabelSize(0.05);
  hist->GetYaxis()->SetLabelSize(0.05);
  hist->GetXaxis()->SetTitleOffset(1.);
  hist->GetYaxis()->SetTitleOffset(1.8);
  hist->GetXaxis()->SetNdivisions(505);
  hist->SetMarkerStyle(20);
  hist->SetMarkerSize(0.8);
  hist->SetLineWidth(2);
  hist->SetTitle("");
  hist->SetLineColor(color);
  hist->SetMarkerColor(color);
}
