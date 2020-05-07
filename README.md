# GLAD
Graph Laplacian with Dropouts

## Requirements
1. Matlab > R2016a

## How to use?
1. Run **setup.m** to perform initial setup.<br/>
1. Run **divRun_Lm.m** to obtain classifier's performance with base graph Laplacian (m>1 for iterated Laplacian).<br/>
1. Run **divRun_pt.m** to obtain classifier's performance with GLAD (pt-threshold probability).<br/>

## Results
<figure>
  <figcaption>Table: Mean error (± Standard deviation) LapRLSC (Test)</figcaption>
  <img src="./results/01_LapRLSC_Test.png" style="width:100%">
</figure><br/>
<figure>
  <figcaption>Table: Mean error (± Standard deviation) LapRLSC (Unlabeled)</figcaption>
  <img src="./results/02_LapRLSC_Unlabeled.png" style="width:100%">
</figure><br/>
<figure>
  <img src="./results/03_Facebook.png" style="width:100%">
  <figcaption>Mean error on Facebook metrics</figcaption>
</figure><br/>
<figure>
  <img src="./results/04_GLAD.png" style="width:100%">
  <figcaption>WSN localization using GLAD</figcaption>
</figure>
<br/>
<figure>
  <img src="./results/05_iL2PA.png" style="width:100%">
  <figcaption>WSN localization using Iterative Laplacian with LPA</figcaption>
</figure>
<br/>
<figure>
  <img src="./results/06_ExcludePercentVsRms.png" style="width:100%">
  <figcaption>Exclude percent vs RMSE (WSN)</figcaption>
</figure>
<br/>
<figure>
  <img src="./results/07_ExcludePercentVsRounds.png" style="width:100%">
  <figcaption>Exclude percent vs number of rounds (WSN)</figcaption>
</figure>
