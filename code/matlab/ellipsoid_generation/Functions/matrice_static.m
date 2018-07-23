function [A,R1] = matrice_static(Rayon_A,Position_A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [A] = matrice(Rayon,Position)
%
%% Description : 
%  Cette fonction permet de déterminer les matrices A et B des ellipsoides
%  en question apres un temps dt.
%
%% Variable(s) d'entrée
%  Rayon:      Rayons de l'ellipsoide 
%  Position:   Position de l'ellipsoide 

%% Variable de sortie:
%  A : Matrice [4x4]caractéristique de l'ellipsoide
%  R1: Matrice de rotation [3x3]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Caractéristique de l'ellipsoide
Rayon = Rayon_A(1:3);
Position = Position_A(1:3);
Quaternion = Position_A(4:7);

% Matrice A sous forme canonique 
%A_canonique = [1/(Rayon(1)^2) 0 0 0;0 1/(Rayon(2)^2) 0 0;0 0 1/(Rayon(3)^2) 0;0 0 0 -1];
A_canonique = zeros(4,4);
A_canonique(1,1) = 1/(Rayon(1)^2);
A_canonique(2,2) = 1/(Rayon(2)^2);
A_canonique(3,3) = 1/(Rayon(3)^2);
A_canonique(4,4) = -1;

% Matrice de rotation 3D
e0 = Quaternion(1);
e1 = Quaternion(2);
e2 = Quaternion(3);
e3 = Quaternion(4);
R1 = [1-2*(e2^2+e3^2)  2*(e1*e2-e0*e3) 2*(e1*e3+e0*e2);2*(e1*e2+e0*e3) 1-2*(e1^2+e3^2) 2*(e2*e3-e0*e1);2*(e1*e3-e0*e2) 2*(e2*e3+e0*e1) 1-2*(e1^2+e2^2)];
R1t = R1';

% Matrice Position
T1  =  Position'; 

% Matrice de rotation 4D
M1_inv = [R1t -R1t*T1;[0 0 0 1]];

% Matrice des ellipsoides A l'instant t+dt
A = M1_inv'*(A_canonique*M1_inv);

