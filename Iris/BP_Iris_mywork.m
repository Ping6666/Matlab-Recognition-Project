function BP_Iris_mywork

    load IRIS_IN.csv;
    load IRIS_OUT.csv;
    input = IRIS_IN;
    target = IRIS_OUT;

    % initialize the weight matrix
    outputmatrix = zeros(12, 1);

    for i = 1:1:12

        for j = 1:1:1
            outputmatrix(i, j) = rand;
        end

    end

    hiddenmatrix = zeros(4, 12);

    for i = 1:1:4

        for j = 1:1:12
            hiddenmatrix(i, j) = rand;
        end

    end

    RMSE = zeros(100, 1);

    % Training
    for epoch = 1:1:100
        t = [];

        for iter = 1:1:75

            % 前傳部分
            hiddensigma = input(iter, :) * hiddenmatrix;
            hiddennet = logsig(hiddensigma);

            outputsigma = hiddennet * outputmatrix;
            outputnet = purelin(outputsigma);

            % 倒傳部分
            % 輸出層的 delta
            doutputnet = dpurelin(outputsigma);
            deltaoutput = (target(iter) - outputnet) * doutputnet;
            error = target(iter) - outputnet;
            t = [t; error.^2];

            % 隱藏層的 delta
            % deltahidden = - deltaoutput * hiddennet; % 會飽和 不過 acc = 0.1067 - 0.1333 up is - down is +
            % deltahidden = -(deltaoutput * hiddennet) .* error; % 不會飽和 但是 acc = 0 - 0.0667
            deltahidden = -(deltaoutput * hiddennet);

            % 輸出層權重更新
            % outputmatrix = outputmatrix - deltaoutput; % acc = 0.0267 up is + down is -
            % outputmatrix = outputmatrix + 0.05 * (deltaoutput * hiddennet)'; % acc = 0.1 - 0.55
            outputmatrix = outputmatrix + 0.15 * (deltaoutput * hiddennet)';

            % 隱藏層權重更新
            newhiddenmatrix = hiddenmatrix;

            for i = 1:1:12

                for j = 1:1:4
                    newhiddenmatrix(j, i) = hiddenmatrix(j, i) + 0.45 * deltahidden(i) * input(j);
                    % maybe can change the weight of deltahidden
                end

            end

            hiddenmatrix = newhiddenmatrix;
        end

        RMSE(epoch) = sqrt(sum(t) / 75);
        fprintf('epoch %.0f:  RMSE = %.3f\n', epoch, sqrt(sum(t) / 75));
    end

    fprintf('\nTotal number of epochs: %g\n', epoch);
    fprintf('Final RMSE: %g\n', RMSE(epoch));
    plot(1:epoch, RMSE(1:epoch));
    legend('Training');
    ylabel('RMSE'); xlabel('Epoch');

    Tot_Correct = 0;

    for i = 76:length(input)

        hiddensigma = input(i, :) * hiddenmatrix;
        hiddennet = logsig(hiddensigma);
        outputsigma = hiddennet * outputmatrix;
        outputnet = purelin(outputsigma);

        if outputnet > target(i) - 0.5 & outputnet <= target(i) + 0.5
            Tot_Correct = Tot_Correct + 1;
        end

    end

    Tot_Percent = (Tot_Correct) / (length(input) - 75);
    Test_correct_percent = Tot_Percent
    % acc > 0.5
end
