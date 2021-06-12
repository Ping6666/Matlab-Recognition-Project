function [prototypeFACE, testFACE, ans_acc_num] = PCALDA_Test(projectPCA, LDA, prototypeFACE, TotalMeanFACE)
    principlenum = 50;
    ldanum = 10;
    % 這會影響辨識率

    people = 40;
    withinsample = 5;
    totalcount = 0;
    correct = 0;

    testFACE = [];

    projectLDA = LDA(:, 1:ldanum);

    for i = 1:1:people

        for j = 2:2:10
            totalcount = totalcount + 1;
            tempF = [];
            s = ['orl3232' '\' num2str(i) '\' num2str(j) '.bmp'];

            test_ = imread(s);
            [row, col] = size(test_);
            test_ = double(test_);

            % imshow(test, map)

            for k = 1:row
                tempF = [tempF, test_(k, :)];
            end

            tempF = tempF - TotalMeanFACE;

            tempF_ = tempF * projectPCA; % row

            % write your code here start %
            ldanum = 20;

            resultF = tempF_ * LDA(:, 1:ldanum);

            OAF_v = [];

            for p = 1:1:people * withinsample % 200
                OAF = resultF - prototypeFACE(p, :);
                Eucdis = OAF * OAF';

                OAF_v = [OAF_v, Eucdis];
            end

            testFACE = [testFACE; resultF];

            % OAF_v 1*200
            [junk, index] = sort(OAF_v, 'descend');

            OAF_v = OAF_v(:, index);

            ans_num = ceil(index(end) / 5);

            if (ans_num == i)
                correct = correct + 1;
            end

            % write your code here end %

            % teacher's code start %

            % Global_min_dis = Inf % (id=?)

            % Local_min_dis = Inf

            % Testing floor taking out id

            % teacher's code end %
        end

    end

    ans_acc_num = correct / totalcount;

end
