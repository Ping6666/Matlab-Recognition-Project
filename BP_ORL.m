function [Train_Percent, trainACCList, Test_Percent, testACCList] = BP_ORL(prototypeFACE)
    % only need to train and test total 100 records
    % prototypeFACE = prototypeFACE(1:length(prototypeFACE) / 4, :);

    % value-min/max-min in all vlaue prototypeFACE (正規化)
    minNum = inf; maxNum = -inf;

    for i = 1:length(prototypeFACE)

        for j = 1:length(prototypeFACE(i))

            if minNum > prototypeFACE(i, j)
                minNum = prototypeFACE(i, j);
            end

            if maxNum < prototypeFACE(i, j)
                maxNum = prototypeFACE(i, j);
            end

        end

    end

    for i = 1:length(prototypeFACE)

        for j = 1:length(prototypeFACE(i))

            prototypeFACE(i, j) = (prototypeFACE(i, j) - minNum) / (maxNum - minNum);

        end

    end

    %一層隱藏層就OK 30個神經元
    %最低分80分softmax就有90分以上

    input = prototypeFACE(1:2:end, :);

    targetNum = 40; % tag num = 40 (個分類)
    tmpNUM = 30; % node num
    length_i = length(input);
    length_j = 20; % data 寬度 = ldanum

    epochMax = 50;
    learningRate = 0.025;

    % tag of the dataset
    target = [];

    % 40 * 5 = 200
    for i = 1:targetNum

        for j = 1:5
            temp = zeros(targetNum, 1);
            temp(i) = 1;
            target = cat(3, target, temp);
        end

    end

    % size(target) = 40 1 200

    % initialize the weight matrix
    outputmatrix = [];

    for k = 1:1:targetNum
        tempM = zeros(tmpNUM, 1);

        for i = 1:1:tmpNUM

            for j = 1:1:1
                tempM(i, j) = rand;
            end

        end

        outputmatrix = cat(3, outputmatrix, tempM);
    end

    % size(outputmatrix) = 30 1 40

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
            hiddennet = logsig(hiddensigma);

            outputsigma = [];
            outputnet = [];

            for i = 1:targetNum
                outputsigma = [outputsigma, hiddennet * outputmatrix(:, :, i)];
                outputnet = [outputnet, purelin(outputsigma(i))];
            end

            % 倒傳部分
            % 輸出層的 delta

            doutputnet = [];
            deltaoutput = [];
            tempError = 0; %error

            for i = 1:targetNum
                doutputnet = [doutputnet, dpurelin(outputsigma(i))];
                temp_ = target(iter) - outputnet(i);
                deltaoutput = [deltaoutput, temp_ * doutputnet(i)];
                tempError = tempError + temp_;
            end

            t = [t; tempError.^2];

            % 隱藏層的 delta
            % tempdelta = deltaoutput(1) * outputmatrix(:, :, 1);

            % for i = 2:targetNum
            %     tempdelta = tempdelta + deltaoutput(i) * outputmatrix(:, :, i);
            % end

            % transfer = dlogsig(hiddensigma, logsig(hiddensigma));

            % deltahidden = [];

            % for i = 1:1:tmpNUM
            %     deltahidden = cat(3, deltahidden, tempdelta * transfer);
            % end

            tempdelta = [];

            for i = 1:targetNum
                tempdelta = [tempdelta; deltaoutput(i) * outputmatrix(:, :, i)];
            end

            transfer = dlogsig(hiddensigma, logsig(hiddensigma));

            deltahidden = [];

            for i = 1:1:tmpNUM
                tmp_ = [];

                for j = 1:1:targetNum
                    tmp_ = [tmp_; tempdelta(j) * transfer];
                end

                deltahidden = cat(3, deltahidden, tmp_);
            end

            % 輸出層權重更新
            for i = 1:targetNum
                outputmatrix(:, :, i) = outputmatrix(:, :, i) + learningRate * (deltaoutput(i) * hiddennet)';
            end

            % 隱藏層權重更新
            newhiddenmatrix = hiddenmatrix;

            for i = 1:1:tmpNUM

                for j = 1:1:length_j
                    newhiddenmatrix(j, i) = hiddenmatrix(j, i) + learningRate * deltahidden(i) * input(j);
                    % maybe can change the weight of deltahidden
                end

            end

            if learningRate > 0.001
                learningRate = learningRate - 0.001;
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

        outputsigma = [];
        outputnet = [];
        maxNum = -inf; maxNumIdx = 0;

        for j = 1:targetNum
            outputsigma = [outputsigma, hiddennet * outputmatrix(:, :, j)];
            outputnet = [outputnet, purelin(outputsigma(j))];
        end

        for j = 1:targetNum

            if maxNum < outputnet(j)
                maxNum = outputnet(j);
                maxNumIdx = j;
            end

        end

        maxNum = -inf; correctIdx = 0;

        for j = 1:targetNum

            if maxNum < target(j, :, i)
                maxNum = target(j, :, i);
                correctIdx = j;
            end

        end

        if maxNumIdx == correctIdx
            Train_Correct = Train_Correct + 1;
            ansList = [i, correctIdx];
            trainACCList = [trainACCList; ansList];
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

        outputsigma = [];
        outputnet = [];
        maxNum = -inf; maxNumIdx = 0;

        for j = 1:targetNum
            outputsigma = [outputsigma, hiddennet * outputmatrix(:, :, j)];
            outputnet = [outputnet, purelin(outputsigma(j))];
        end

        for j = 1:targetNum

            if maxNum < outputnet(j)
                maxNum = outputnet(j);
                maxNumIdx = j;
            end

        end

        maxNum = -inf; correctIdx = 0;

        for j = 1:targetNum

            if maxNum < target(j, :, i)
                maxNum = target(j, :, i);
                correctIdx = j;
            end

        end

        if maxNumIdx == correctIdx
            Test_Correct = Test_Correct + 1;
            ansList = [i, correctIdx];
            testACCList = [testACCList; ansList];
        end

    end

    Test_Percent = (Test_Correct) / (length(input));
    fprintf("%g, %g, %g\n", Test_Correct, length(input), Test_Percent);

    % acc > 0.9
end
