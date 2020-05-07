function [et,eu]=experiment_diverse(method,...
    gamma_A, gamma_I, NN, G_WEIGHT, GW_PARAM, file, pt)
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
            y=[ones(size(train{i},1),1); -ones(size(train{j},1),1)];
            yt=[ones(size(test{i},1),1); -ones(size(test{j},1),1)];
            for k=1:size(Labels, 2)
                ypos=zeros(size(train{i},1),1);
                ypos(Labels(p,k,1:l))=1;
                yneg=zeros(size(train{j},1),1);
                yneg(Labels(p,k,l+1:2*l))=-1;
                ytr=[ypos;yneg];
                lab=find(ytr);
                unlab=find(ytr==0);

                xprime=x(lab,:);    % LABELED INSTANCES
                yprime=y(lab);
                xxprime=x(unlab,:);    % UNLABELED INSTANCES
                yyprime=y(unlab);
                
                ysoft=ytr(unlab);   % Placeholder for soft-labels
                
                round=1;
                flag=zeros(size(xxprime,1),1);
                
                while(~isempty(find(flag==0,1)))
                    px = rand(size(flag));
                    indx = find(px>=pt);
                    ei = px<pt;
                    flag(ei)=1;
                
                    xtr=xxprime(indx,:);  % TRAINING
                    ytr=yyprime(indx);      % TRUE LABELS
                    
                    ytmp=ysoft(indx);   % SOFT-LABELS
                    
                    [alpha,K]=myclassification(method,[xprime;xtr],...
                        [yprime;ytmp],options);
                    fu=K(size(xprime,1)+1:end,:)*alpha;
                    bu=breakeven(fu,ytr,@pre_rec_equal);
                    ytmp=sign(fu-bu);
                    fprintf('round %d=%0.4f\n',round,evaluate(ytmp,ytr));

                    ysoft(indx)=ytmp;   % ASSIGN SOFT-LABELS
                    round=round+1;
                end

                [alpha,K]=myclassification(method,[xprime;xxprime],...
                    [yprime;ysoft],options);
                fu=K(size(xprime,1)+1:end,:)*alpha;
                bu=breakeven(fu,yyprime,@pre_rec_equal);
                ytmp=sign(fu-bu);
                eu(p,k)=evaluate(ytmp,yyprime);

                KT=calckernel(options.Kernel,options.KernelParam,[xprime;xxprime],xt);
                ft=KT*alpha;
                bt=breakeven(ft,yt,@pre_rec_equal);
                et(p,k)=evaluate(sign(ft-bt),yt);

                fprintf('%d %d %0.4f %0.4f\n',p,k,et(p,k),eu(p,k));
            end
        end
    end

function e=evaluate(a,b)
    e=sum(a~=b)/length(b)*100;

function [alpha,K]=myclassification(method,xtr,ytr,options)
    K=calckernel(options.Kernel,options.KernelParam,xtr);
    L = laplacian(xtr, 'nn', options);
    switch method
        case 'laprlsc'
            [alpha,~]=laprlsc(K,ytr,L,options.gamma_A,options.gamma_I);
        case 'lapsvm'
            [alpha,~]=lapsvm(K,ytr,L,options.gamma_A,options.gamma_I);
    end