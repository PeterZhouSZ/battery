function [Valeur] = verifier_collision(Rayon_A,Rayon_B,Position_A,Position_B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Valeur] = verifier_collision(Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B)
%
%% Description :
%  Cette fonction permet de v�rifier si deux ellipsoides se p�n�trent
%
%% Variable(s) d'entr�e :
%  Rayon_A:    Rayons de l'ellipsoide A
%  Rayon_B:    Rayons de l'ellipsoide B
%  Position_A: Position de l'ellipsoide A
%  Position_B: Position de l'ellipsoide B
%  Vitesse_A:  Vitesse de l'ellipsoide A
%  Vitesse_B:  Vitesse de l'ellipsoide B
%
%% Variable de sortie:
% Valeur : Valeur = 0 si pas de p�n�tration et Valeur = 1 s'il ya p�n�tration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Impl�mentation

% Initialisation de la variable Valeur
Valeur = 0;

% Calcul des matrices A et B des ellipsoides
A = matrice_static(Rayon_A,Position_A);
B = matrice_static(Rayon_B,Position_B);

% Marge
Marge = 1e-4;

% Calcul des 4 racines
Racine = eig(B,A);
% Verifier s'il y a p�n�tration
if sum(abs(imag(Racine))) > Marge  || sum(Racine>0) == 4
    Valeur = 1;
end

