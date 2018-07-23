function [sr] = polynome(dt,Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B,Output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [sr] = polynome(dt,Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B)
%
%% Description : 
%  Cette fonction permet de calculer la valeur des polynomes sr (Voir
%  article X. Jia et al (2011)
%
%% Variable(s) d'entrée
%  dt:         Temps pour lequel on calcul les polynomes  t = t0 + dt
%  Rayon_A:    Rayons de l'ellipsoide A
%  Rayon_B:    Rayons de l'ellipsoide B
%  Position_A: Position de l'ellipsoide A
%  Position_B: Position de l'ellipsoide B
%  Vitesse_A:  Vitesse de l'ellipsoide A
%  Vitesse_B:  Vitesse de l'ellipsoide B
%
%% Variable de sortie:
%  sr : Vecteur contenant la valeur des polynomes sr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% On elève dt au carré pour eviter les solution négatif
dt = dt^2;

% Calcul des matrices A et B des ellipsoides à l'instant t+dt
[~,~,MA,Ac] = matrice(dt,Rayon_A,Position_A,Vitesse_A);
[B] = matrice(dt,Rayon_B,Position_B,Vitesse_B);

%% Voir article Choi et al (2009) p.320

% Calcul de la matrice Bij 
Bij = MA'*B*MA;

% Calcul des coefficient a,b,c,d et e du polynome caractéristique
% det(lamda*A+B) = 0  (pour les coefficients b et d, on ajoute un signe "-")
a =  - (Ac(1,1)*Ac(2,2)*Ac(3,3));

b = -  (Bij(1,1)*Ac(2,2)*Ac(3,3) + Bij(2,2)*Ac(1,1)*Ac(3,3) + Bij(3,3)*Ac(1,1)*Ac(2,2) - Bij(4,4)*Ac(1,1)*Ac(2,2)*Ac(3,3));

c = Ac(1,1)*Ac(2,2)*(Bij(3,3)*Bij(4,4) - Bij(3,4)*Bij(4,3)) + Ac(2,2)*Ac(3,3)*(Bij(1,1)*Bij(4,4) - Bij(1,4)*Bij(4,1)) + Ac(1,1)*Ac(3,3)*(Bij(2,2)*Bij(4,4) - Bij(2,4)*Bij(4,2)) ...
  + Ac(1,1)*(Bij(2,3)*Bij(3,2) - Bij(2,2)*Bij(3,3))  +  Ac(2,2)*(Bij(1,3)*Bij(3,1) - Bij(1,1)*Bij(3,3)) + Ac(3,3)*(Bij(1,2)*Bij(2,1) - Bij(1,1)*Bij(2,2)) ;

d = Ac(1,1)*(-Bij(2,2)*Bij(3,3)*Bij(4,4) + Bij(2,2)*Bij(3,4)*Bij(4,3) + Bij(3,3)*Bij(4,2)*Bij(2,4)) ...
  + Ac(1,1)*(Bij(4,4)*Bij(3,2)*Bij(2,3)  - Bij(3,2)*Bij(2,4)*Bij(4,3) - Bij(4,2)*Bij(2,3)*Bij(3,4)) ...
                                                                                                    ...
  + Ac(2,2)*(-Bij(1,1)*Bij(3,3)*Bij(4,4) + Bij(1,1)*Bij(3,4)*Bij(4,3) + Bij(3,3)*Bij(1,4)*Bij(4,1)) ...
  + Ac(2,2)*(Bij(4,4)*Bij(1,3)*Bij(3,1)  - Bij(3,1)*Bij(1,4)*Bij(4,3) - Bij(4,1)*Bij(1,3)*Bij(3,4)) ...
                                                                                                    ...
  + Ac(3,3)*(-Bij(1,1)*Bij(2,2)*Bij(4,4) + Bij(1,1)*Bij(2,4)*Bij(4,2) + Bij(2,2)*Bij(1,4)*Bij(4,1)) ...
  + Ac(3,3)*(Bij(4,4)*Bij(1,2)*Bij(2,1)  - Bij(2,1)*Bij(1,4)*Bij(4,2) - Bij(4,1)*Bij(1,2)*Bij(2,4)) ...
                                                                                                    ...
  + Bij(1,1)*Bij(2,2)*Bij(3,3) - Bij(1,1)*Bij(2,3)*Bij(3,2) - Bij(2,2)*Bij(1,3)*Bij(3,1) - Bij(3,3)*Bij(1,2)*Bij(2,1) ...
  + Bij(2,1)*Bij(1,3)*Bij(3,2) + Bij(3,1)*Bij(1,2)*Bij(2,3);

d=-d;
     
e = det(B);

%% (Voir article X. Jia et al (2011)

% On écrit l'éqautino sous la forme x4 + ax3 + bx2 + cx + d = 0
Coefficient = [a b c d e]/a;
a = Coefficient(2);
b = Coefficient(3);
c = Coefficient(4);
d = Coefficient(5);

% Calcul des polynomes sr (voir p. 170)
b2 = -a/4; 
c2 = b/6;
d2 = -c/4;
e2 = d;

Delta2 = b2^2 - c2;
Delta3 = c2^2 - b2*d2;

W1 = d2 - b2*c2;
%W2 = b2*e2 - c2*d2;
W3 = e2 - b2*d2;

T = -9*W1^2 + 27*Delta2*Delta3 - 3*W3*Delta2;
A = W3 + 3*Delta3;
B = -d2*W1 - e2*Delta2 - c2*Delta3;
T2 = A*W1 - 3*b2*B;
Delta1 = A^3 - 27*B^2;

sr22 = Delta2;
sr20 = -W3;
sr11 = T;
sr10 = T2;
sr0 = Delta1;

if Output == 1
    sr = sr0;
%     for i = 1:length(solution)
%     sr = sr/(dt-solution(i)^2);
%     end
elseif Output  == 2
    sr = sr11;
else
    sr = [sr22 sr20 sr11 sr10 sr0];
end


