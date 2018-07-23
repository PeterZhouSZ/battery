function [A2, D2, F2] = quadratique_plan(A1,Axe,Face)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [A2, D2, F2] = quadratique_plan(A1,Axe,Face)
%
%% Description :
%  Cette fonction permet de trouver les matrice de la quadratique 
%  X'A2X + D2'X + F2 = 0 qui résulte de l'intersection de l'ellipsoide 
%  X'A1X = 0 avec le plan alpha = Face
%
%% Variable(s) d'entrée :
%  A1:   La matrice de départ X'A1X = 0 où A1 = [4x4]
%  Axe:  L'axe perpendiculaire au plan. Axe = 1 si alpha = x, Axe = 2 si 
%        alpha = y Axe = 3 si alpha = z      
%  Face: Abscisse de l'intersection entre le plan et l'axe alpha.
%
%  Exemple : Si on s'intéresse à l'intersection de l'ellipsoide avec le plan
%  y = 1, on mettra Axe = 2 et Face = 1.
%
%% Variable de sortie:
% A2, D2, F2 : Matrice de la quadratique X'A2X + D2'X + F2 = 0.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implementation

% Ecriture sous la forme X'AX + D'X + F = 0 (Dimension 3)
A = A1(1:3,1:3);
D = 2*A1(1:3,4); % (car A1 est symétrique)
F = A1(4,4);

% Matrice de Passage P et Q apres avoir posé alpha = a
if Axe == 1
    P = [0 0;1 0;0 1];
    Q = [Face 0 0]';
elseif Axe == 2
    P = [1 0;0 0;0 1];
    Q = [0 Face 0]'; 
else
    P = [1 0;0 1;0 0];
    Q = [0 0 Face]';
end

% Ecriture sous for la forme X'AX + D'X + F = 0 (Dimension 2)
A2 = (P'*A)*P;
D2 = 2*(P'*A)*Q + P'*D; % (car A est symétrique)
F2 = (Q'*A)*Q + D'*Q + F;