clc;
clear all;
KK = [10,20];
filename2 = 'EA4eigN100_10';
jjj = [200000, 1000000];
for j = 1:2
    for k = 1:12
        filename = [filename2,'_',num2str(k),'_',num2str(KK(j)),'.txt'];
        eval('A=load(filename);');
        A(1:16,:) = A(1:16,:).*(A(1:16,:) > 1e-8);
        for i = 1:30
            a = find(A(1:16,i)==0);
            if isempty(a)
            else
               A(17,i) = ceil(10.^((min(a)-1)./5-3)*jjj(j));
            end
        end
        filename3 = [filename2,'_',num2str(KK(j)), '_full', '.xlsx'];
        xlswrite(filename3,A,k)
        B(:,k) = A(16,:)';
        C(:,k) = A(17,:)';
    end
    filename4 = [filename2,'_',num2str(KK(j)),'.xlsx'];
    xlswrite(filename4,B,1);
    xlswrite(filename4,C,2);
end
