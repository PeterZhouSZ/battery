   function [nouveau_evenement] = avancer(evenement,temps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [nouveau_evenement] = avancer(evenement,temps)
%
%% Description : 
%  Cette fonction permet de mettre à jour l'état des ellipsoides
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides
%
%% Variable de sortie:
%  nouveau_evenement : ensemble de cellule contenant l'ensemble des
%  ellipsoides mis à jour.           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Copie de l'ensemble de cellule
nouveau_evenement = evenement;

% Extraction des matrice de l'ensemble de cellules
Temps = nouveau_evenement{2};
Rayon = nouveau_evenement{3};
Position = nouveau_evenement{4}(:,1:3);
Quaternion = nouveau_evenement{4}(:,4:7);
Vitesse = nouveau_evenement{5}(:,1:3);
Angulaire = nouveau_evenement{5}(:,4:6);

% Mise à jour de la matrice temps
Temps(:,2) = Temps(:,1);
Temps(:,1) = Temps(:,1) + temps;

% Mise à jour du rayon
Rayon(:,1:3) = Rayon(:,1:3) + ( temps*Rayon(:,4:6) );

% Mise à jour de la Position
Position = Position + (temps*Vitesse);

% Mise à jour des Quaternions

for i = 1:size(Rayon,1)
    
    angle = Angulaire(i,:);
    w = norm(angle);
    temp = Quaternion(i,:);
    qdt = [cos(w*temps/2) (sin(w*temps/2)/w)*angle];
    Quaternion(i,1) = temp(1)*qdt(1) - (temp(2)*qdt(2) + temp(3)*qdt(3) + temp(4)*qdt(4));
    Quaternion(i,2:4) = temp(1)*qdt(2:4) + qdt(1)*temp(2:4) - [temp(3)*qdt(4)-temp(4)*qdt(3),temp(4)*qdt(2)-temp(2)*qdt(4),temp(2)*qdt(3)-temp(3)*qdt(2)];
    
end

% Insertion des matrices dans l'ensemble de cellule
nouveau_evenement{2} = Temps;
nouveau_evenement{3} = Rayon;
nouveau_evenement{4} = [Position Quaternion];

    
    