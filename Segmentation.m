close all;
clear;
clc;

%rng(0);         % Seed random generator
nClusters = 4;  % Number of clusters
nMaxIterations = 100; % Maximal number of iterations

I = imread('cameraman.tif');
IVec = double( I(:) );  % transform to vector of pixels (of type double)
nPixels = length(I); % Number of pixels
imageMax = max(IVec); % Maximal level in the image
imageMin = min(IVec); % Minimal level in the image

% Randomly initialize cluster centroids
centroids = rand(nClusters, 1) * ( imageMax - imageMin ) + imageMin;

% Main loop of k-means clustering
for i = 1:nMaxIterations
    labels = cluster( IVec, centroids );
    newCentroids = findCentroids( IVec, labels, nClusters );
    % Check convergence
    relChange = norm(newCentroids - centroids)/norm(centroids);
    if relChange <= 0.001  % relative change is too small
        break;            % stop the process
    end
    centroids = newCentroids;
end

%% Show results
disp("K-means clustering finished after " + i + " iterations");
% Create and show image of the found clusters
labels = cluster( IVec, centroids );
labels = reshape(labels, size(I));
figure; imagesc(labels);
title("Clustering result");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cluster all the pixels in the image using given centroids
% For each pixel the closest centroid to it is found and returned in labels
% vector.
function labels = cluster( IVec, centroids )
    nClusters = length( centroids );
    nPixels   = length(IVec);
    centroids = centroids * ones( 1, nPixels ); % duplicate centroids
    pixels    = ( IVec * ones( 1, nClusters ) ).'; % duplicate pixels
    distsAbs = abs( centroids - pixels );
    [~,labels] = min( distsAbs );
end

% Find the centroids corresponding to the clustering in labels
function centroids = findCentroids( IVec, labels, nClusters )    
    centroids = zeros(nClusters, 1);

    % Run on each cluster
    for i = 1:nClusters        
        IVeci = IVec( labels == i );  % Gray levels of all pixels in cluster i
        centroids(i) = mean( IVeci ); % new centroid
    end    
end
