function [BestFeatures, OptimalFitness, OptFit] = hybridFOA(dataMatrix, populationSize, maxIterations)

    %% Load labels
    load ColonCancerExtLbls.mat;
    labels = ColonCancerlbl;

    [numSamples, numFeatures] = size(dataMatrix);

    %% Initialize population
    agents = rand(populationSize, numFeatures); 
    fitness = zeros(populationSize, 1);

    BestFitness = -inf;
   
    %% 2. FOA main loop
    for iter = 1:maxIterations
       
       
        for i = 1:populationSize
            mask = agents(i,:) > 0.5; 

            if sum(mask) < 2
                fitness(i) = -inf;
                continue;
            end

            reducedData = dataMatrix(:, mask);
            fitness(i) = LASSO(reducedData, labels, 0.001); 
        end

        % ---- Update global best ----
        [currentBestFitness, bestIdx] = max(fitness);
        if currentBestFitness > BestFitness
            BestFitness = currentBestFitness;
        end

        % ---- Exploration / Exploitation ----
        validFitness = fitness(~isinf(fitness));
        threshold = mean(validFitness);

        for i = 1:populationSize
            randIdx = randi(populationSize);
            % distance = agents(i,:) - agents(randIdx,:);

            if fitness(i) < threshold
                % Exploration
                agents(i,:) = agents(i,:) + rand .* distance;
            else
                % Exploitation
                agents(i,:) = agents(i,:) + 0.5 * rand(1,numFeatures) .* ...
                             (agents(bestIdx,:) - agents(i,:));
            end

            agents(i,:) = min(max(agents(i,:), 0), 1);
        end
    end

    finalData = dataMatrix(:, BestMask); 
    [~, finalMask, finalCoeffs] = LASSO(finalData, labels, 0.0001);

    BestFeatures = finalData(:, finalMask);           
    %selectedCoeffs = finalCoeffs(finalMask);  
    [numSamples, k] = size(BestFeatures);
    OptFit = zeros(numSamples, k);

    for f = 1:k
        OptFit(:, f) = BestFeatures(:, f) * selectedCoeffs(f);
    end

   % OptFit = BestFitness;

end
