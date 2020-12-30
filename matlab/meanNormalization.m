function features = meanNormalization (features)
%meanNormalization Performs feature scaling can make gradient descent
%converge much more quickly.

avgFeatures = mean(features);
ranFeatures = range(features);
features = (features - avgFeatures) / ranFeatures;

end
