% the program file name has to be the function (inner) name
function [FFACE, zeromeanTotalFACE, TotalMeanFACE, pcaTotalFACE] = PCA_Train

    % for all data the origin PCA is named FFACE
    % first find the zero mean vector : zeromeanTotalFACE
    % and use it to compute latent vector and the eigenvalues
    % for this (PCA) situation here we need to find the biggest eigenvalue
    % to calculate the direction of all point
    % make all point on this dir. is most focus (smallest distantce)

    people = 40;

    withinsample = 5;

    principlenum = 50;

    FFACE = [];

    for k = 1:1:people

        for m = 1:2:10
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

end
