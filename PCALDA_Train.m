% function [projectPCA, LDA, prototypeFACE, TotalMeanFACE] = PCALDA_Train
function [prototypeFACE] = PCALDA_Train
    ldanum = 20;

    people = 40;

    withinsample = 10;

    principlenum = 50;

    FFACE = [];

    for k = 1:1:people

        for m = 1:1:10
            matchstring = ['orl3232' '\' num2str(k) '\' num2str(m) '.bmp'];
            matchX = imread(matchstring);
            matchX = double(matchX);

            if (k == 1 && m == 1)
                [row, col] = size(matchX);
            end

            matchtempF = [];

            for n = 1:row
                matchtempF = [matchtempF, matchX(n, :)];
                % , will output row direction
                % ; will output col direction
            end

            FFACE = [FFACE; matchtempF];
        end

    end

    TotalMeanFACE = mean(FFACE);
    zeromeanTotalFACE = FFACE;

    for i = 1:1:withinsample * people

        for j = 1:1:(row) * (col)
            zeromeanTotalFACE(i, j) = zeromeanTotalFACE(i, j) - TotalMeanFACE(j);
        end

    end

    SST = zeromeanTotalFACE' * zeromeanTotalFACE;
    % pcaSST = cov(zeromeanTotalFACE);

    [PCA, latent] = eig(SST);
    % the PCA here is eigenvector

    eigenvalue = diag(latent);

    [junk, index] = sort(eigenvalue, 'descend');

    PCA = PCA(:, index); % PCA(row, col)

    % useless in the following
    eigenvalue = eigenvalue(index);

    % extract the principle compenent
    projectPCA = PCA(:, 1:principlenum);

    pcaTotalFACE = [];

    for i = 1:1:withinsample * people
        tempFACE = zeromeanTotalFACE(i, :);
        tempFACE = tempFACE * projectPCA; % 內積求新座標值
        pcaTotalFACE = [pcaTotalFACE; tempFACE];
        % 儲存所有投影至 PCA 空間中的訓練影像
        % 投影之後的資料與資料之間距離是最大化的 (因為特徵與之垂直)
        % 由此所求出的特徵才會是我們所在乎的
    end

    % LDA transform
    for i = 1:withinsample:withinsample * people
        withinFACE = pcaTotalFACE(i:i + withinsample - 1, :);
        % 暫存單一類別PCA空間中訓練影像
        if (i == 1)
            SW = withinFACE' * withinFACE;
            % SW = cov(withinFACE);
            ClassMean = mean(withinFACE);
        end

        if (i > 1)
            SW = SW + withinFACE' * withinFACE;
            ClassMean = [ClassMean; mean(withinFACE)];
        end

    end

    pcaTotalmean = mean(pcaTotalFACE);

    % for i = 1:people

    %     for j = 1:principlenum
    %         ClassMean(i, j) = ClassMean(i, j) - pcaTotalmean(j);
    %     end

    % end

    SB = ClassMean' * ClassMean;

    [LDA, latent] = eig(inv(SW) * SB);
    % the LDA here is eigenvector

    eigenvalue = diag(latent);

    [junk, index] = sort(eigenvalue, 'descend');

    % useless in the following
    eigenvalue = eigenvalue(index);

    % extract the principle compenent
    LDA = LDA(:, index);

    prototypeFACE = pcaTotalFACE * LDA(:, 1:ldanum);
end
