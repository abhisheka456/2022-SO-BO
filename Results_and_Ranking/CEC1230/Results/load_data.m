clc;
clear all;
KK = [10,20];
filename2 = 'NL-SHADE-LBC';

for j = 1:2
    for k = 1:12
        filename = [filename2,'_',num2str(k),'_',num2str(KK(j)),'.txt'];
        eval('A=load(filename);');
        A(1:16,:) = A(1:16,:).*(A(1:16,:) > 1e-8);
        filename3 = [filename2,'_',num2str(KK(j)), '_full', '.xlsx'];
        xlswrite(filename3,A,k)
        B(:,k) = A(16,:)';
        C(:,k) = A(17,:)';
    end
    filename4 = [filename2,'_',num2str(KK(j)),'.xlsx'];
    xlswrite(filename4,B,1);
    xlswrite(filename4,C,2);
end
