function [et,eu,alphas,bias,sup_vecs]=experiment_all_Lm(method,...
    gamma_A, gamma_I, NN, G_WEIGHT, GW_PARAM, file, m)
if ~exist([file '.mat'],'file')
    fprintf('%s.mat does not exists',file);
    et=inf;eu=inf;
    return;
else
    load(file);
end
[MAX,n]=size(train);
if MAX<n
    MAX=n;
end
l=size(Labels, 3)/2;

options=ml_options('Kernel', kernel, 'KernelParam', kernelParam, ...
    'NN', NN, 'gamma_A', gamma_A, 'gamma_I', gamma_I, ...
    'GraphWeights', G_WEIGHT, 'GraphWeightParam', GW_PARAM);

p=0;
for i=1:MAX
    for j=i+1:MAX
        p=p+1;
        x=[train{i};train{j}];
        xt=[test{i};test{j}];
        if isa(x,'uint8')||~isempty(strfind(lower(file),'hasy'))...
            ||~isempty(strfind(lower(file),'cvpr'))...
            ||~isempty(strfind(lower(file),'ucmerced'))
            x=zscore(im2double(x));
            xt=zscore(im2double(xt));
        end
        yt=[ones(size(test{i},1),1); -ones(size(test{j},1),1)];
        ytrue=[ones(size(train{i},1),1); -ones(size(train{j},1),1)];
        disp('Computing Kernels');
        K=calckernel(kernel,kernelParam,x);
        KT=calckernel(kernel,kernelParam,x,xt);
        disp('Done.');
        
        if strcmp(method,'lapsvm') || strcmp(method, 'laprlsc')
            L=(laplacian(x,'nn',options)).^m;
        end
        
        for k=1:size(Labels, 2)
            ypos=zeros(size(train{i},1),1);
            ypos(Labels(p,k,1:l))=1;
            yneg=zeros(size(train{j},1),1);
            yneg(Labels(p,k,l+1:2*l))=-1;
            y=[ypos;yneg];
            lab=find(y);
            unlab=find(y==0);
            yu=ytrue(unlab);
            
            switch method
                case 'rlsc'
                    [alpha,b]=rlsc(K(lab,lab),y(lab),options.gamma_A);
                    fu=K(unlab,lab)*alpha;
                    ft=KT(:,lab)*alpha;
                case 'svm'
                    [alpha,b]=svm(K(lab,lab),y(lab),options.gamma_A);
                    fu=K(unlab,lab)*alpha;
                    ft=KT(:,lab)*alpha;
                case 'laprlsc'
                    [alpha,b]=laprlsc(K,y,L,options.gamma_A,options.gamma_I);
                    fu=K(unlab,:)*alpha;
                    ft=KT*alpha;
                case 'lapsvm'
                    [alpha,b]=lapsvm(K,y,L,options.gamma_A,options.gamma_I);
                    fu=K(unlab,:)*alpha;
                    ft=KT*alpha;
                case 'tsvm'
                    [alpha,b,svs]=readtsvmmodel(['results_tsvm/model_' num2str(k) '.' num2str(p)]);
                    fu=calckernel('poly',3,svs,x(unlab,:))*alpha;
                    ft=calckernel('poly',3,svs,xt)*alpha;
                    sup_vecs{p,k}=svs;
            end
            
            bt=breakeven(ft,yt,@pre_rec_equal);
            bu=breakeven(fu,yu,@pre_rec_equal);
            
            et(p,k)=evaluate(sign(ft-bt),yt);
            eu(p,k)=evaluate(sign(fu-bu),yu);
            
            alphas{p,k}=alpha;
            bias(p,k)=b;
            
            disp([p k et(p,k) eu(p,k)]);
            
        end
    end
end

function [alpha,b,svs]=readtsvmmodel(file,i)
[alpha,svs]=svmlread(file);
b=alpha(9);
alpha=alpha(10:end);
svs=svs(10:end,:);

function e=evaluate(a,b)
e=sum(a~=b)/length(b)*100;

function b=get_bias(v,r)
v1=sort(v);
p=ceil(r*length(v));
b=v1(p);