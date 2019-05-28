clear variables; clc;

%%
%files = dir(fullfile('./test_results'));
files = dir(fullfile('./test_results_1'));
%files = dir(fullfile('./test_results_2'));
%files = dir(fullfile('./test_results_3'));
files(1:2) = [];

for i = 1 : size(files, 1)
	name = ['./test_results_1/', files(i).name];
	load(name);
	
	rad = max(rad, 0);
	rad = min(rad, 4095);
	
	save(name, 'rad', '-v7.3');
	clear rad; disp(i);
end
