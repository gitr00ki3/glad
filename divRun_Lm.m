%% FOR NORMAL LAPLACIAN
if ~exist('elapnormL1','var')
    close all;clear;clc;
    addpath(genpath('./data'),genpath('./graphLaplacian'));
    fname={'usps','BciHaLT_A','Hasy2','cifar01','UCMercedLand',...
        'mnist','CVPRv2','LegoBricks',...
        'NaturalImages'};
    gammaA=[0.05,0.5,0.05,0.05,0.9,0.5,0.05,0.01,...
        0.09];
    gammaI=[0.005,0.05,0.005,0.07,0.05,0.05,0.9,0.5,...
        0.5];
    NN=[6,6,20,8,20,6,30,5,7];
    t=[1,6,1,1000,100,1,40000,30,10];
    m=1;
    methods={'laprlsc'};
    weight={'heat'};
    elapnormL1=struct();
    i=1;j=1;k=1;
end
%%
for i=i:length(fname)
    elapnormL1(i).fname=cell2mat(fname(i));
    experiment='experiment_all_Lm';
    for j=j:length(methods)
        elapnormL1(i).methods(j).method=cell2mat(methods(j));
        for k=k:length(weight)
            [et, eu] = feval(experiment,...
                cell2mat(methods(j)),gammaA(i),...
                gammaI(i), NN(i), cell2mat(weight(k)), t(i),...
                cell2mat(fname(i)),m);
            
            elapnormL1(i).methods(j).weights(k).weight=...
                cell2mat(weight(k));
            elapnormL1(i).methods(j).weights(k).et=et;
            elapnormL1(i).methods(j).weights(k).eu=eu;
        end
        k=1;
    end
    j=1;
    save('result_v4.mat','elapnormL1','-append');
end