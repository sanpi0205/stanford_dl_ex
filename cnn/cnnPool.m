function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%

%% 向量方法
pooled_total = poolDim^2;
average_k = ones(poolDim)/pooled_total ;

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        current_image = squeeze(convolvedFeatures(:,:,filterNum,imageNum));
        Img_conv = conv2(current_image, average_k, 'valid');
        aux = downsample(Img_conv,poolDim);
        aux1 = downsample(aux',poolDim);
        aux1 = aux1';
        
        pooledFeatures(:,:,filterNum,imageNum) = aux1;
    end
end


%% 循环方法
% numRows = size(pooledFeatures,1);
% numCols = size(pooledFeatures,2);
% 
% for i=1:numImages
%     for j=1:numFilters
%         
%         for k = 1:numRows
%            for m=1:numCols
%               sub_feature = convolvedFeatures( poolDim*(k-1)+1:poolDim*k ,poolDim*(m-1)+1:poolDim*m ,j,i);
%               pooledFeatures(k,m,j,i) = mean2(sub_feature);
%            end
%         end
%     end
% end
             
end

