function [Point1, Point2] = ellipse_droite(A1,D1,F1,Point_M)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function ellipse_droite(A1,D1,F1,Point)
%
%% Description :
%  Cette fonction permet de trouver les 2 points d'intersection d'une
%  ellipse et d'une droite passant par son centre et par un Point MN
%
%% Variable(s) d'entrée :
% A1, D1, F1 : Matrice de la quadratique X'A1X + D1'X + F1 = 0.
% Point_M    : Coordonnées du point M [x y]'
%
%% Variable de sortie:
% Point1, Point 2 : Vecteurs contenant les coordonnées [x y] des points
% d'intersection
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implementation

% Calcul du centre de l'ellipse
Centre = -0.5*inv(A1)*D1;
% Equation de la droite reliant le centre de l'ellipsee au point M : y=ax +b
a = (Point_M(2) - Centre(2))/(Point_M(1) - Centre(1));
b = (Centre(2)*Point_M(1) - Centre(1)*Point_M(2))/(Point_M(1) - Centre(1));

% Matrice de Passage P et Q
P = [1 a]';
Q = [0 b]';

% Ecriture sous for la forme AX^2+ BX + C = 0 (Dimension 1)
A = P'*A1*P;
B = P'*(A1+A1')*Q + P'*D1;
C = Q'*A1*Q + D1'*Q + F1;

% Résolution du système d'équation
% Point 1
Point1 = zeros(2,1);
Point1(1) = (-B - sqrt(B^2 - 4*A*C))/(2*A);
if abs(imag(Point1(1))) < 1e-3
    Point1(1) = real(Point1(1));
end
Point1(2) = a*Point1(1) + b;
% Point 2
Point2 = zeros(2,1);
Point2(1) = (-B + sqrt(B^2 - 4*A*C))/(2*A);
if abs(imag(Point2(1))) < 1e-3
    Point2(1) = real(Point2(1));
end
Point2(2) = a*Point2(1) + b;