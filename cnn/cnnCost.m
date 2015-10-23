function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
activationsPooled = cnnPool(poolDim, activations);


% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%

% reference£ºhttps://github.com/PedroCV/UFLDL-Tutorial-Solutions/blob/master/Additional_3_Convolutional_Neural_Network/cnnCost.m
% calculate z_nl
z_nl = Wd*activationsPooled + repmat(bd, 1, numImages);

% minus the max value of z_nl to make it less than 1
tmp = bsxfun(@minus, z_nl, max(z_nl, [], 1));
% calculate the a_nl value
a_nl = exp(tmp);
probs = bsxfun(@rdivide, a_nl, sum(a_nl));

clear tmp;

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%

M = size(images,3);
groundTruth = full(sparse(labels, 1:M, 1));
aux4 = groundTruth.*probs;
aux5 = log(aux4(aux4 ~= 0)); 

cost = -mean(aux5);  % have not include the weight decay

clear aux4;
clear aux5;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%

error_term_nl = (-1/M).*(groundTruth - probs);
clear groundTruth;

% not include the weight decay
% activationsPooled is the a(l)
Wd_grad = error_term_nl*activationsPooled'; 
clear activationsPooled;
bd_grad = error_term_nl*ones(M,1);


%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%

% calculate the pooled error term
error_term_pooled = Wd'*error_term_nl;
error_term_pooled = reshape(error_term_pooled,outputDim,outputDim,numFilters,numImages);
clear error_term_nl;

% define the error term for convolution
error_term_convolution = zeros(convDim,convDim,numFilters,numImages);

for imageNum = 1:numImages
    im = squeeze(images(:,:,imageNum));
    for filterNum = 1:numFilters
        
        % the error term from pooled to convolution is been averaged by
        % poolDim
        error_term_pooled_to_convolution = (1/(poolDim^2)).*kron(squeeze(error_term_pooled(:,:,filterNum,imageNum)),ones(poolDim));
        
        % final error term is times by a*(1-a), also f'(z)
        error_term_convolution(:,:,filterNum,imageNum) = error_term_pooled_to_convolution.*activations(:,:,filterNum,imageNum).*(1-activations(:,:,filterNum,imageNum));
        
        % calculate the delta and for a(1) is the input.
        delta_1 = squeeze(error_term_convolution(:,:,filterNum,imageNum));
        gradient_wc = conv2(im,rot90(squeeze(delta_1),2), 'valid');
        
        Wc_grad(:,:,filterNum) = squeeze(Wc_grad(:,:,filterNum)) + gradient_wc; 
        bc_grad(filterNum) = bc_grad(filterNum) + sum(delta_1(:));
        
    end    
end



%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
