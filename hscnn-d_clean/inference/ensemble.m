clear variables; %clc;

%% model 11
names = {'Fri_1215-1050'; 'Fri_1215-1221'; 'Sat_1223-1151'; 'Sat_1223-1321'; 'Sat_1223-1550'};
for i = 1 : 5
	load(['./test_results_1/', cell2mat(names(i)), '.mat']);
	rad11(:, :, :, i) = rad;
	clear rad;
end

%% model 12
for i = 1 : 5
	load(['./test_results_1/', cell2mat(names(i)), '_ex.mat']);
	rad12(:, :, :, i) = rad;
	clear rad;
end

%% rad1
for i = 1 : 5
	rad1(:, :, :, i) = (rad11(:, :, :, i) + rad12(:, :, :, i)) / 2;
end
clear rad11 rad12;

%% model 21
for i = 1 : 5
	load(['./test_results_2/', cell2mat(names(i)), '.mat']);
	rad21(:, :, :, i) = rad;
	clear rad;
end

%% model 22
for i = 1 : 5
	load(['./test_results_2/', cell2mat(names(i)), '_ex.mat']);
	rad22(:, :, :, i) = rad;
	clear rad;
end

%% rad2
for i = 1 : 5
	rad2(:, :, :, i) = (rad21(:, :, :, i) + rad22(:, :, :, i)) / 2;
end
clear rad21 rad22;

%% model 31
for i = 1 : 5
	load(['./test_results_3/', cell2mat(names(i)), '.mat']);
	rad31(:, :, :, i) = rad;
	clear rad;
end

%% model 32
for i = 1 : 5
	load(['./test_results_3/', cell2mat(names(i)), '_ex.mat']);
	rad32(:, :, :, i) = rad;
	clear rad;
end

%% rad3
for i = 1 : 5
	rad3(:, :, :, i) = (rad31(:, :, :, i) + rad32(:, :, :, i)) / 2;
end
clear rad31 rad32;

%% save
for i = 1 : 5
	rad = 0.4*rad1(:, :, :, i) + 0.3*rad2(:, :, :, i) + 0.3*rad3(:, :, :, i);
	save(['./test_results/', cell2mat(names(i)), '.mat'], 'rad', '-v7.3');
	clear rad;
end
