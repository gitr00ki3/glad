close all;clear;clc;
sDir='graphLaplacian';
url='http://manifold.cs.uchicago.edu/manifold_regularization/';
fname='ManifoldLearn';
%%
load('myFiles.mat','files');
if ~exist(sDir,'dir')
    mkdir(sDir);
else
    sfiles = dir(sDir);
    names = {sfiles.name};
    dirFlags = ~[sfiles.isdir] & ~strcmp(names, '.') & ~strcmp(names, '..')...
        & ~strcmp(names, 'mexGramSVMTrain.mexw64');
    sfiles = sfiles(dirFlags);
    if sum(strcmpi({sfiles.name},{files.name}))==length(files)
        return;
    end
end
cd(sDir);
gunzip([url fname '.tar.gz']);
untar([fname '.tar']);
%%
for i = 1:length(files)
    movefile([fname '\' files(i).name],['.\' files(i).name]);
end
rmdir(fname,'s');
delete([fname '.tar']);
%mex('mexGramSVMTrain.cpp','svmprecomputed.cpp');
cd('..');