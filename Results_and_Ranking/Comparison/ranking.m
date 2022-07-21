clc;
clear all;
%% methods
M1 = 'Co-PPSO';
M2 = 'EA4eigN100_10';
M3 = 'IMPML-SHADE';
M4 = 'IUMOEAII';
M5 = 'jSObinexpEig';
M6 = 'MTT_SHADE';
M7 = 'NL-SHADE-LBC';
M8 = 'NL-SHADE-RSP-MID';
M9 = 'OMCSOMA';
M10 = 'S_LSHADE_DP';
M11 = 'SPHH_Ensemble';
M12 = 'ZOCMAES';
M13 = 'NLSOMACLP';
%% Dimension
D1 = '10';
D2 = '20';
%% ranking
for i = 1:2
    %% comparsion
    for k = 1:13
        for l = k+1:13   
            %% method-1
            eval(['file1 = M',num2str(k),';']);
            eval(['file3 = D', num2str(i),';']);
            method1 = [file1,'_',file3,'.xlsx'];
            eval(['A1 = xlsread(method1,',num2str(1),');'])
            eval(['A2 = xlsread(method1,',num2str(2),');'])
            %% method-2
            eval(['file1 = M',num2str(l),';']);
            method2 = [file1,'_',file3,'.xlsx'];
            eval(['B1 = xlsread(method2,',num2str(1),');'])
            eval(['B2 = xlsread(method2,',num2str(2),');'])
            %% comparsion
            for p = 1:12
                    S1 = 0;
                    S2 = 0;
                    for m = 1:30
                        for n = 1:30
                            if A1(m, p) < B1(n, p)
                                S1 = S1+1;
                            elseif A1(m, p) > B1(n, p)
                                S2 = S2+1;
                            else
                                if A2(m, p) < B2(n, p)
                                    S1 = S1+1;
                                elseif A2(m, p) > B2(n, p)
                                    S2 = S2+1;
                                elseif A2(m, p) == B2(n, p)
                                    S1 = S1+0.5;
                                    S2 = S2+0.5;
                                end   
                            end
                        end 
                      
                    end

                    filename1 = ['S',num2str(k),num2str(l),'_',num2str(i)];%[filename1,'(',num2str(p),') = S1;']
                    eval([filename1,'(',num2str(p),') = S1;'])
                    filename2 = ['S',num2str(l),num2str(k),'_',num2str(i)];%[filename2,'(',num2str(p),') = S2;']
                    eval([filename2,'(',num2str(p),') = S2;'])
                    %pause
                end
            end
        end
end
for i = 1:2
        %% comparsion
        file1 = sprintf('Score%d%d',i);
        eval([file1,'=zeros(9);'])
        for k = 1:13
            for l = 1:13
                file2 = [file1,'(k,l)'];
                file3 = ['sum(S',num2str(k),num2str(l),'_',num2str(i),')'];
                if k == l
                else
                    eval([file2,'=',file3,';'])
                end
            end
        end
end