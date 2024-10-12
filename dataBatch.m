function [xbatch,batchnum]=dataBatch(X,batchsize)
% 将X按照batchsize分成多个batch
    N=size(X,1);
    batchnum=ceil(N/batchsize);
    xbatch=cell(batchnum,1);
    for i=1:batchnum
        l=(i-1)*batchsize+1;
        r=i*batchsize;
        if r>N
            r=N;
        end
        t_1=X(l:r,:);
        xbatch{i}=X(l:r,:);
    end

end