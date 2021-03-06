clear;
load result_v4.mat;
clearvars -except elapnormL1 elapnormL2 elapnormL4 elapnormL8 elapemr24 elapemr72...
    elapdrop25 elapdrop50 elapdrop75;
findex=[1,3,6,2,4,8,5,7,9];
%%
stdw = 1;
m=1;
tbs=cell(length(elapnormL1),2);
for loop=1:length(elapnormL1)
    i=findex(loop);
    for j=1:length(elapnormL1(i).methods)
        meanEt=[]; stdEt=[]; meanEu=[]; stdEu=[];
        for k=1:length(elapnormL1(i).methods(j).weights)
            meanEt=[mean(mean(elapnormL1(i).methods(j).weights(k).et,2)),...
                mean(mean(elapnormL2(i).methods(j).weights(k).et,2)),...
                mean(mean(elapnormL4(i).methods(j).weights(k).et,2)),...
                mean(mean(elapnormL8(i).methods(j).weights(k).et,2)),...
                mean(mean(elapemr24(i).methods(2).weights(k).et,2)),...
                mean(mean(elapemr72(i).methods(2).weights(k).et,2)),...
                mean(mean(elapdrop25(i).methods(j).weights(k).et,2)),...
                mean(mean(elapdrop50(i).methods(j).weights(k).et,2)),...
                mean(mean(elapdrop75(i).methods(j).weights(k).et,2))];
            stdEt=[std(std(elapnormL1(i).methods(j).weights(k).et,stdw,2)),...
                std(std(elapnormL2(i).methods(j).weights(k).et,stdw,2)),...
                std(std(elapnormL4(i).methods(j).weights(k).et,stdw,2)),...
                std(std(elapnormL8(i).methods(j).weights(k).et,stdw,2)),...
                std(std(elapemr24(i).methods(2).weights(k).et,stdw,2)),...
                std(std(elapemr72(i).methods(2).weights(k).et,stdw,2)),...
                std(std(elapdrop25(i).methods(j).weights(k).et,stdw,2)),...
                std(std(elapdrop50(i).methods(j).weights(k).et,stdw,2)),...
                std(std(elapdrop75(i).methods(j).weights(k).et,stdw,2))];
            meanEu=[mean(mean(elapnormL1(i).methods(j).weights(k).eu,2)),...
                mean(mean(elapnormL2(i).methods(j).weights(k).eu,2)),...
                mean(mean(elapnormL4(i).methods(j).weights(k).eu,2)),...
                mean(mean(elapnormL8(i).methods(j).weights(k).eu,2)),...
                mean(mean(elapemr24(i).methods(2).weights(k).eu,2)),...
                mean(mean(elapemr72(i).methods(2).weights(k).eu,2)),...
                mean(mean(elapdrop25(i).methods(j).weights(k).eu,2)),...
                mean(mean(elapdrop50(i).methods(j).weights(k).eu,2)),...
                mean(mean(elapdrop75(i).methods(j).weights(k).eu,2))];
            stdEu=[std(std(elapnormL1(i).methods(j).weights(k).eu,stdw,2)),...
                std(std(elapnormL2(i).methods(j).weights(k).eu,stdw,2)),...
                std(std(elapnormL4(i).methods(j).weights(k).eu,stdw,2)),...
                std(std(elapnormL8(i).methods(j).weights(k).eu,stdw,2)),...
                std(std(elapemr24(i).methods(2).weights(k).eu,stdw,2)),...
                std(std(elapemr72(i).methods(2).weights(k).eu,stdw,2)),...
                std(std(elapdrop25(i).methods(j).weights(k).eu,stdw,2)),...
                std(std(elapdrop50(i).methods(j).weights(k).eu,stdw,2)),...
                std(std(elapdrop75(i).methods(j).weights(k).eu,stdw,2))];
        end
        [~,minEt]=min(meanEt); [~,minEu]=min(meanEu);
        tabEt=cell(1); tabEu= cell(1);
        for l=1:length(meanEt)
            if minEt==l
                temp=['\tdatabm{' num2str(meanEt(l)) '}{\pm' ...
                    num2str(stdEt(l)) '}'];
            else
                temp=['\tdatam{' num2str(meanEt(l)) '}{\pm' ...
                    num2str(stdEt(l)) '}'];
            end
            tabEt{1}=[tabEt{1} ' & ' temp];

            if minEu==l
                temp=['\tdatabm{' num2str(meanEu(l)) '}{\pm' ...
                    num2str(stdEu(l)) '}'];
            else
                temp=['\tdatam{' num2str(meanEu(l)) '}{\pm' ...
                    num2str(stdEu(l)) '}'];
            end
            tabEu{1}=[tabEu{1} ' & ' temp];
        end
        index=((j-1)*j)+1;
        tbs{m,index}=[elapnormL1(i).fname ' & ' elapnormL1(i).methods(j).method ' & Test'...
            tabEt{1} '\\\hline'];
        tbs{m,index+1}=[elapnormL1(i).fname ' & ' elapnormL1(i).methods(j).method ' & Unlab'...
            tabEu{1} '\\\hline'];
    end
    m=m+1;
end
clearvars -except tbs;