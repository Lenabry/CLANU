clear all;
close all;
%clc;
addpath('../');

     %Ajout du chemin � suivre
addpath('D:\COURS\3A\SEM2\Maths\CLANU\Code_CLANU_V4\Code_CLANU_V4');
%-- Nicolas: 'C:\Users\nicow\Documents\INSA LYON\2019-2020\GE\maths\CLANU\Code_CLANU_VX'
%-- L�na 
        %Chargement de la base de donn�es tri�e
        %elle contient les donn�es d'entrainement X_train
        %les donn�es de validation X_valid
        %les donn�es de test X_test
        
filename='../data/database_tri.mat';
load(filename);


%-- Build a model with a n_h-dimensional hidden layer
num_iterations = 10000;

%param�tres d'inertie de la m�thode d'adam (reprise des valeurs du pdf)
b1 = 0.9;
b2=0.99;
delta = 10^(-9); 
learning_rate=0.001;

print_cost = true;
nX = size(database.X_train,1);
layers_dims = [nX, 6, 1];

%adaptation des param�tres d'entr�e d�e � l'utilisation de la m�thode
%d'Adam qui a 4 param�tres d'iniertie au lieu d'un seul  dans la m�thode du gradient 
[parameters,costs] = L_layers_nn.model(database, layers_dims, num_iterations, print_cost, learning_rate, b1, b2, delta);


%-- Compute accuracy
X_train = database.X_train;
Y_train = database.Y_train;
X_valid = database.X_valid;
Y_valid = database.Y_valid;
X_test = database.X_test;
Y_test = database.Y_test;
Y_prediction_train = L_layers_nn.predict(parameters, X_train);
Y_prediction_valid = L_layers_nn.predict(parameters, X_valid);


%-- Print train/test Errors
disp(['train accuracy: ', num2str(100 - mean(abs(Y_prediction_train - Y_train)) * 100), ' %'])
disp(['valid accuracy: ', num2str(100 - mean(abs(Y_prediction_valid - Y_valid)) * 100), ' %'])    
% Fig1=figure('Name','Comparaion des erreurs des m�thodes')
% hold on
% plot(1:N,num2str(100 - mean(abs(Y_prediction_train - Y_train))*100));
% plot(1:N,num2str(100 - mean(abs(Y_prediction_valid - Y_valid))*100));
