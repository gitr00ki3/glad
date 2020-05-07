%% FOR NORMAL LAPLACIAN
if ~exist('elapdrop25','var')
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
    pt=ones(1,length(NN))*0.25;
    methods={'laprlsc'};
    weight={'heat'};
    elapdrop25=struct();
    i=1;j=1;k=1;
end
%%
for i=i:length(fname)
    elapdrop25(i).fname=cell2mat(fname(i));
    experiment='experiment_diverse';
    for j=j:length(methods)
        elapdrop25(i).methods(j).method=cell2mat(methods(j));
        for k=k:length(weight)
            clc;
            [et, eu] = feval(experiment,...
                cell2mat(methods(j)),gammaA(i),...
                gammaI(i), NN(i), cell2mat(weight(k)), t(i),...
                cell2mat(fname(i)), pt(i));
            
            elapdrop25(i).methods(j).weights(k).weight=...
                cell2mat(weight(k));
            elapdrop25(i).methods(j).weights(k).et=et;
            elapdrop25(i).methods(j).weights(k).eu=eu;
            save('result_v4.mat','elapdrop25','-append');
        end
        k=1;
    end
    j=1;
    save('result_v4.mat','elapdrop25','-append');
end