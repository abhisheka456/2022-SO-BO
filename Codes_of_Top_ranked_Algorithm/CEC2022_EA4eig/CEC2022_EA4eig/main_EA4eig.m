% EA4 by P. Bujok, 2021
% petr.bujok@osu.cz
clc;
clear;
% mex cec21_func.cpp -DWINDOWS

clear mex;


format long;
Runs = 11;
fhd=@cec22_test_func;
N = 100;

fistar = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700];


func = 1:12;

for D = [10 20]
    switch D
        case 10
            maxfes = 200000;
        case 20
            maxfes = 1000000;
        otherwise
            disp('Error..')
    end
    
    fprintf('\n-------------------------------------------------------\n\n')
    Run_EA4eig(Runs, fhd, D, func, maxfes, fistar);
end

