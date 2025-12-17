function [fitness, selectedMask, coeffs] = LASSO(featureMatrix, nominalClasses, lambda)
    Y = dummyvar(nominalClasses);

    numFeatures = zeros(featureMatrix,2);
    selectedMask = false(1,numFeatures);
   
    for i = 1:size(Y,2)
        [beta, FitInfo] = lasso(featureMatrix, Y(:,i), 'Lambda', lambda);
        [~, minMSEIndex] = min(FitInfo.MSE);
        selectedMask = selectedMask | (beta(:,minMSEIndex)' ~= 0);
        coeffs = coeffs + abs(beta(:,minMSEIndex)); 
    end

    numSelected = sum(selectedMask);
    if numSelected == 0
        fitness = -inf;
    else
        fitness = Fitness_criteria;
    end
end
