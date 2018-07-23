function [Minimum] = minimum(dt,Rayon_A,Position_A, Vitesse_A,Axe,Face)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Minimum] = minimum(dt,Rayon_A,Position_A, Vitesse_A,Axe,Face)
%
%% Description :
%  Cette fonction permet de calculer le temps de la prochaine sortie du
%  cube par un ellipsoide
%
%% Variable(s) d'entrée :
%  dt:         Temps pour lequel on cvalcul les racines  t = t0 + dt
%  Rayon_A:    Rayons de l'ellipsoide A
%  Position_A: Position de l'ellipsoide A
%  Vitesse_A:  Vitesse de l'ellipsoide A
%  Axe:        L'axe perpendiculaire au plan. Axe = 1 si alpha = x, Axe = 2 si 
%              alpha = y Axe = 3 si alpha = z      
%  Face:       Abscisse de l'intersection entre le plan et l'axe alpha.

%% Variable de sortie:
%  Minimum: Minimum de la quadratique résultante de l'intersection 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% On elève dt au carré pour eviter les solution négatif
dt = dt^2;

% Matrice de l'ellipsoide
A = matrice(dt,Rayon_A,Position_A,Vitesse_A);

% Calcul de la quadratique résultant de l'intersection entre l'ellipsoide et la
% face du cube
[A2, D2, F2] = quadratique_plan(A,Axe,Face);

% Calcul du minimum de la quadratique
temp = inv(A2);
Minimum = (-0.25*(D2'*temp)*D2 + F2);  % (car temp est symétrique)
% Minimum = (0.5*D2'*(0.5*temp'- temp)*D2 + F2); 