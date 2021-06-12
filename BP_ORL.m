function [Train_Percent, trainACCList, Test_Percent, testACCList] = BP_ORL(prototypeFACE)
    % only need to train and test total 100 records
    % prototypeFACE = prototypeFACE(1:length(prototypeFACE) / 4, :);
    prototypeFACE = prototypeFACE(1:length(prototypeFACE) / 10, :);

    % pre-process
    % value-min/max-min in all vlaue prototypeFACE (正規化)
    % end

    %一層隱藏層就OK 30個神經元

    %output softmax
    %iris只有一個output
    %但是因為有20個目標(類別) 所以要有20個output神經元
    %最低分80分softmax就有90分以上

    % result
    % test acc = 0.34
    % train acc = 0.80

    input = prototypeFACE(1:2:end, :);

    % tag of the dataset
    target = [];

    % 40 * 5 = 200
    for i = 1:40

        for j = 1:5
            target = [target; i];
        end

    end

    tmpNUM = 50;
    length_i = length(input);
    length_j = 20;
    epochMax = 250; % 100-500 is enough
    learningRate = 0.025;

    % initialize the weight matrix
    outputmatrix = zeros(tmpNUM, 1);

    for i = 1:1:tmpNUM

        for j = 1:1:1
            outputmatrix(i, j) = rand;
        end

    end

    % size(outputmatrix) = 50 1

    hiddenmatrix = zeros(length_j, tmpNUM);

    for i = 1:1:length_j

        for j = 1:1:tmpNUM
            hiddenmatrix(i, j) = rand;
        end

    end

    RMSE = zeros(epochMax, 1);

    % Training
    for epoch = 1:1:epochMax
        t = [];

        for iter = 1:1:length_i

            % 前傳部分
            hiddensigma = input(iter, :) * hiddenmatrix;
            % size(hiddensigma) = 1 50
            hiddennet = logsig(hiddensigma);

            outputsigma = hiddennet * outputmatrix;
            outputnet = purelin(outputsigma);
            % size(outputsigma) = 1 1
            % size(outputnet) = 1 1

            % 倒傳部分
            % 輸出層的 delta
            doutputnet = dpurelin(outputsigma);
            deltaoutput = (target(iter) - outputnet) * doutputnet;
            error = target(iter) - outputnet;
            t = [t; error.^2];

            % 隱藏層的 delta
            % deltahidden = -(deltaoutput * hiddennet); % bad acc = 0.0250 - 0.03, 0.0250
            tempdelta = deltaoutput * outputmatrix;

            % size(deltaoutput) = 1 1
            % size(outputmatrix) = 50 1
            % size(tempdelta) = 50 1

            transfer = dlogsig(hiddensigma, logsig(hiddensigma));
            deltahidden = [];

            for i = 1:1:tmpNUM
                deltahidden = [deltahidden; tempdelta(i) * transfer(i)];
            end

            % 輸出層權重更新
            outputmatrix = outputmatrix + learningRate * (deltaoutput * hiddennet)';

            % 隱藏層權重更新
            newhiddenmatrix = hiddenmatrix;

            for i = 1:1:tmpNUM

                for j = 1:1:length_j
                    newhiddenmatrix(j, i) = hiddenmatrix(j, i) + learningRate * deltahidden(i) * input(j);
                    % maybe can change the weight of deltahidden
                end

            end

            hiddenmatrix = newhiddenmatrix;
        end

        RMSE(epoch) = sqrt(sum(t) / length_i);
        fprintf('epoch %.0f:  RMSE = %.3f\n', epoch, RMSE(epoch));
    end

    fprintf('\nTotal number of epochs: %g\n', epoch);
    fprintf('Final RMSE: %g\n', RMSE(epoch));
    plot(1:epoch, RMSE(1:epoch));
    legend('Training');
    ylabel('RMSE'); xlabel('Epoch');

    % train
    Train_Correct = 0;
    trainACCList = [];

    for i = 1:length_i
        hiddensigma = input(i, :) * hiddenmatrix;
        hiddennet = logsig(hiddensigma);
        outputsigma = hiddennet * outputmatrix;
        outputnet = purelin(outputsigma);

        if outputnet > target(i) - 0.5 & outputnet <= target(i) + 0.5
            Train_Correct = Train_Correct + 1;
            trainACCList = [trainACCList; i];
        end

    end

    Train_Percent = Train_Correct / length_i;
    fprintf("%g, %g, %g\n", Train_Correct, length_i, Train_Percent);

    % test
    input = prototypeFACE(2:2:end, :);

    Test_Correct = 0;
    testACCList = [];

    for i = 1:length(input)
        hiddensigma = input(i, :) * hiddenmatrix;
        hiddennet = logsig(hiddensigma);
        outputsigma = hiddennet * outputmatrix;
        outputnet = purelin(outputsigma);

        if outputnet > target(i) - 0.5 & outputnet <= target(i) + 0.5
            Test_Correct = Test_Correct + 1;
            testACCList = [testACCList; i];
        end

    end

    Test_Percent = (Test_Correct) / (length(input));
    fprintf("%g, %g, %g\n", Test_Correct, length(input), Test_Percent);

    % acc > 0.9
end
