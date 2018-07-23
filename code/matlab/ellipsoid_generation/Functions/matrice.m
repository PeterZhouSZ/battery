function [A,R1,M1,A_canonique] = matrice(dt,Rayon_A,Position_A, Vitesse_A)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [A,B] = matrice(dt,Rayon_A,Position_A, Vitesse_A)
%
%% Description : 
%  Cette fonction permet de déterminer les matrices A et B des ellipsoides
%  en question apres un temps dt.
%
%% Variable(s) d'entrée
%  dt:         Temps pour lequel on cvalcul les racines  t = t0 + dt
%  Rayon_A:    Rayons de l'ellipsoide 
%  Position_A: Position de l'ellipsoide 
%  Vitesse_A:  Vitesse de l'ellipsoide 
%
%% Variable de sortie:
%  A          : Matrice [4x4]caractéristique de l'ellipsoide à l'instant
%               t+dt
%  R1         : Matrcie [3x3] de rotation
%  M1         : Matrice [4x4] qui combine rotation + translation
%  A_canonique: Matrice [4x4] de l'ellipsoide à l'instant t+dt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Caractéristique de l'ellipsoide
Rayon = Rayon_A(1:3);
a = Rayon_A(4);
b = Rayon_A(5);
c = Rayon_A(6);
Position = Position_A(1:3);
Quaternion = Position_A(4:7);
Vitesse = Vitesse_A(1:3);
Angulaire = Vitesse_A(4:6);

% Mise à jour du Quaternion
w = sqrt(Angulaire(1)^2 + Angulaire(2)^2 + Angulaire(3)^2);
temp = Quaternion;
angle = w*dt/2;
qdt = [cos(angle) (sin(angle)/w)*Angulaire];
Quaternion(1) = temp(1)*qdt(1) - (temp(2)*qdt(2) + temp(3)*qdt(3) + temp(4)*qdt(4));
Quaternion(2) = temp(1)*qdt(2) + qdt(1)*temp(2) - (temp(3)*qdt(4)-temp(4)*qdt(3));
Quaternion(3) = temp(1)*qdt(3) + qdt(1)*temp(3) - (temp(4)*qdt(2)-temp(2)*qdt(4));
Quaternion(4) = temp(1)*qdt(4) + qdt(1)*temp(4) - (temp(2)*qdt(3)-temp(3)*qdt(2));
%Quaternion(2:4) = temp(1)*qdt(2:4) + qdt(1)*temp(2:4) - [temp(3)*qdt(4)-temp(4)*qdt(3),temp(4)*qdt(2)-temp(2)*qdt(4),temp(2)*qdt(3)-temp(3)*qdt(2)];


% Matrice A sous forme canonique 
%A_canonique = [1/((Rayon(1)+a*dt)^2) 0 0 0;0 1/((Rayon(2)+b*dt)^2) 0 0;0 0 1/((Rayon(3)+c*dt)^2) 0;0 0 0 -1];
A_canonique = zeros(4,4);
A_canonique(1,1) = 1/((Rayon(1)+a*dt)^2);
A_canonique(2,2) = 1/((Rayon(2)+b*dt)^2);
A_canonique(3,3) = 1/((Rayon(3)+c*dt)^2);
A_canonique(4,4) = -1;

% Matrice e_tilde
%e1_tilde = [0 -Quaternion(4) Quaternion(3);Quaternion(4) 0 -Quaternion(2);-Quaternion(3) Quaternion(2) 0];

% Matrice de rotation 3D
e0 = Quaternion(1);
e1 = Quaternion(2);
e2 = Quaternion(3);
e3 = Quaternion(4);
R1 = [1-2*(e2^2+e3^2)  2*(e1*e2-e0*e3) 2*(e1*e3+e0*e2);2*(e1*e2+e0*e3) 1-2*(e1^2+e3^2) 2*(e2*e3-e0*e1);2*(e1*e3-e0*e2) 2*(e2*e3+e0*e1) 1-2*(e1^2+e2^2)];
R1t = R1';
%R1 = (2*Quaternion(1)^2 - 1)*[1 0 0;0 1 0;0 0 1] + 2*Quaternion(2:4)'*Quaternion(2:4) + 2*Quaternion(1)*e1_tilde

% Matrice Position
T1  =  Position + Vitesse*dt;
T1t = T1';

% Matrice de rotation 4D
M1 = [R1 T1t;[0 0 0 1]];
M1_inv = [R1t -R1t*T1t;[0 0 0 1]];

% Matrice des ellipsoides A l'instant t+dt
A = M1_inv'*(A_canonique*M1_inv);

