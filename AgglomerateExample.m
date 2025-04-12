%% Example code for agglomerating a mesh: Poisson problem on brain slice
clear
clc
close all
% import lymph
run("lymph new/Physics/ImportLymphPaths.m")
% Create simulation data: Poisson
run("lymph new/Physics/Laplacian/InputData/DataBrainLaplacian.m")
SimType = 'laplacian';
mesh_path = 'datasets/BrainCoronalNoHoles.vtu';
output_path = fullfile('lymph new/Physics/Laplacian/InputMesh', 'Brain_sage_nref8.mat');
% agglomeration parameters
model = 'SAGEBase2D';
mode = 'Nref';
param = 8;

[~, ~] = Agglomerate(mesh_path, output_path, ...
                        Data, SimType, ...
                        model, mode, int64(param),param,int64(param));
%% numerical solution
run("lymph new/Physics/Laplacian/RunMainLaplacian.m")
%% Heat problem: mesh agglomeration
% Create simulation data: Heat
run("lymph new/Physics/Laplacian/InputData/HeatAggTest.m")
SimType = 'laplacian';
mesh_path = 'datasets/double_circle_coarse.vtk';
output_path = fullfile('lymph/Physics/Heat/InputMesh', 'DCcoarse_kmeans128.mat');
% agglomeration parameters
model = 'KMEANS';
mode = 'direct_kway';
param = 128;

[~, ~] = Agglomerate(mesh_path, output_path, ...
                        Data, SimType, ...
                        model, mode, int64(param),param,int64(param));



%% numerical solution
% run("lymph/Physics/Laplacian/RunMainLaplacian.m")
run("lymph/Physics/Heat/MainHeatAggTest.m")
%% save nice plot
f = gcf;
exportgraphics(f,'Laplacian_example.png','Resolution',300)
%% h convergence test agglomeration: kmeans with 'same' edges
clear
clc
close all
% import lymph
run("lymph new/Physics/ImportLymphPaths.m")
%simulation data
% run("lymph/Physics/Laplacian/InputData/HeatAggTest.m")
run("lymph new/Physics/Laplacian/InputData/AgghConvTestLap.m")
SimType = 'laplacian';

% output_path = fullfile('lymph/Physics/Heat/InputMesh', 'veryfine_sage2048.mat');
% agglomeration parameters
model = 'KMEANS';
mode = 'direct_kway';
squares = [500, 16000]; %
for i = 1:length(squares)
    param = squares(i)/10;
    output_path = fullfile('lymph new/Physics/Laplacian/InputMesh', strcat("kmeans_squares_",string(param),".mat"));
    mesh_path = strcat('datasets/squares_for_convergence_test/square',string(squares(i)),'.vtk');
    % output_path = fullfile('lymph new/Physics/Laplacian/InputMesh', strcat("sage_finesquare_nref",string(param),".mat"));

    [~, ~] = Agglomerate(mesh_path, output_path, ...
                            Data, SimType, ...
                            model, mode, int64(param),param,int64(param));
end

%% h convergence test agglomeration: kmeans with 'same' edges
clear
clc
close all
% import lymph
run("lymph new/Physics/ImportLymphPaths.m")
%simulation data
run("lymph new/Physics/Laplacian/InputData/AgghConvTestLap.m")
SimType = 'laplacian';
% agglomeration parameters
mesh_path = 'datasets/fine_square.vtk';
model = 'SAGEBase2D';
mode = 'Nref';
params = [5,6,7,8,9,10]; %
for param = params
    output_path = fullfile('lymph new/Physics/Laplacian/InputMesh', strcat("sage_finesquare_nref",string(param),".mat"));
    [~, ~] = Agglomerate(mesh_path, output_path, ...
                            Data, SimType, ...
                            model, mode, int64(param),param,int64(param));
end
%% Heat DC confront models
clear
clc
close all
% import lymph
run("lymph new/Physics/ImportLymphPaths.m")
%simulation data
run("lymph new/Physics/Laplacian/InputData/AgghConvTestLap.m")
SimType = 'laplacian';
% agglomeration parameters
mesh_path = 'datasets/double_circle.vtk';

% model = 'SAGEBase2D';
% mode = 'Nref';
model = 'multiSAGE';
mode = 'multilevel';
param = 8;
output_path = fullfile('lymph new/Physics/Heat/InputMesh', "DCfineheat_sage_nref8");
[~, ~] = Agglomerate(mesh_path, output_path, ...
                            Data, SimType, ...
                            model, mode, int64(param),param,int64(param));

model = 'METIS';
mode = 'direct_kway';
param = 256;
output_path = fullfile('lymph new/Physics/Heat/InputMesh', "DCfineheat_metis_256");
[~, ~] = Agglomerate(mesh_path, output_path, ...
                            Data, SimType, ...
                            model, mode, int64(param),param,int64(param));

model = 'KMEANS';
mode = 'direct_kway';
param = 256;
output_path = fullfile('lymph new/Physics/Heat/InputMesh', "DCfineheat_kmeans_256");
[~, ~] = Agglomerate(mesh_path, output_path, ...
                            Data, SimType, ...
                            model, mode, int64(param),param,int64(param));

%%
clear
clc
close all
% import lymph
run("lymph new/Physics/ImportLymphPaths.m")
%simulation data
run("lymph new/Physics/Laplacian/InputData/AgghConvTestLap.m")
SimType = 'laplacian';
% agglomeration parameters
mesh_path = 'datasets/double_circle.vtk';
model = 'KMEANS';
mode = 'direct_kway';
param = 256;
output_path = fullfile('lymph new/Physics/Laplacian/InputMesh', "DCfineheat_kmeans_256");
[~, ~] = Agglomerate(mesh_path, output_path, ...
                            Data, SimType, ...
                            model, mode, int64(param),param,int64(param));
%%