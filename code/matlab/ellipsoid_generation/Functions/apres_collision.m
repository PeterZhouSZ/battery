function [nouveau_evenement] = apres_collision(evenement,Ellipsoide_1,Ellipsoide_2,rc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function [nouveau_evenement] = apres_collision(evenement,Ellipsoide_1,Ellipsoide_2,rc)
%
%% Description :
%  Cette fonction permet de calculer les vitesses des 2 ellipsoides après leur
%  collision
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides
%  Ellipsoide_1 : Numéro d'identification du premier ellipsoide impliqué dans
%                 la collision
%  Ellipsoide_2 : Numéro d'identification du second ellipsoide impliqué dans
%                 la collision
%
%% Variable de sortie:
%  nouveau_evenement : ensemble de cellule contenant l'ensemble des
%  elliposides mis à jour.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Extraction des matrices de l'ensemble de cellule.
Identite = evenement{1};
Rayon = evenement{3};
Position = evenement{4};
Vitesse = evenement{5};
Partenaire = evenement{7};

% Recherche des 2 ellipsoides dans l'ensemble de cellule evenement

% Initialisation des positions
p=0;
q=0;

for i=1:size(Identite,1)
    
    if Identite(i) == Ellipsoide_1
        
        % Enregistrement de la position
        p=i;
        
        % Vecteur position
        Position1 = Position(i,1:3);
        % Quaternion
        Quaternion1 = Position(i,4:7);
        % Vecteur vitesse linéaire avant collision
        Vitesse1 = Vitesse(i,1:3);
        % Vecteur vitesse angulaire avant collision
        Angulaire1 = Vitesse(i,4:6);
        % Rayons de l'ellipsoide
        a1 = Rayon(i,1);
        b1 = Rayon(i,2);
        c1 = Rayon(i,3);
        % Taux de croissance du rayon
        a0_1 = max(Rayon(i,4:6));
        % Identification des partenaires( au cas où une partie de la sphère
        % est à l'extérieur du cube)
        partenaire_1 = Partenaire(i,:);
        
        % Position relative entre le centre de l'ellipsoide 1 et le point
        % de contact
        r1 = Position1 - rc;
        
        % Calcul de la matrice de rotation de l'ellipsoide 1
        % e1_tilde = [0 -Quaternion1(4) Quaternion1(3);Quaternion1(4) 0 -Quaternion1(2);-Quaternion1(3) Quaternion1(2) 0];
        % R1 = (2*Quaternion1(1)^2 - 1)*eye(3) + 2*Quaternion1(2:4)'*Quaternion1(2:4) + 2*Quaternion1(1)*e1_tilde;
        e0 = Quaternion1(1);
        e1 = Quaternion1(2);
        e2 = Quaternion1(3);
        e3 = Quaternion1(4);
        R1 = [1-2*(e2^2+e3^2)  2*(e1*e2-e0*e3) 2*(e1*e3+e0*e2);2*(e1*e2+e0*e3) 1-2*(e1^2+e3^2) 2*(e2*e3-e0*e1);2*(e1*e3-e0*e2) 2*(e2*e3+e0*e1) 1-2*(e1^2+e2^2)];
        
        % Calcul du volume et de l'inertie de l'ellipsoide 1 dans le repère
        % global
        Volume1 = (4*pi/3)*a1*b1*c1;
        I1 = (Volume1/5)*[b1^2+c1^2 0 0;0 a1^2+c1^2 0;0 0 a1^2+b1^2];
        I1 = R1*I1*R1';
        
    elseif Identite(i) == Ellipsoide_2
        
        % Enregistrement de la position
        q=i;
        
        % Vecteur position
        Position2 = Position(i,1:3);
        % Quaternion
        Quaternion2 = Position(i,4:7);
        % Vecteur vitesse linéaire avant collision
        Vitesse2 = Vitesse(i,1:3);
        % Vecteur vitesse angulaire avant collision
        Angulaire2 = Vitesse(i,4:6);
        % Rayons de l'ellipsoide
        a2 = Rayon(i,1);
        b2 = Rayon(i,2);
        c2 = Rayon(i,3);
        % Taux de croissance du rayon
        a0_2 = max(Rayon(i,4:6));
        % Identification des partenaires( au cas où une partie de la sphère
        % est à l'extérieur du cube)
        partenaire_2 = Partenaire(i,:);
        
        % Position relative entre le centre de l'ellipsoide 2 et le point
        % de contact
        r2 = Position2 - rc;
        
        % Calcul de la matrice de rotation de l'ellipsoide 2
        %e2_tilde = [0 -Quaternion2(4) Quaternion2(3);Quaternion2(4) 0 -Quaternion2(2);-Quaternion2(3) Quaternion2(2) 0];
        %R2 = (2*Quaternion2(1)^2 - 1)*eye(3) + 2*Quaternion2(2:4)'*Quaternion2(2:4) + 2*Quaternion2(1)*e2_tilde;
        e0 = Quaternion2(1);
        e1 = Quaternion2(2);
        e2 = Quaternion2(3);
        e3 = Quaternion2(4);
        R2 = [1-2*(e2^2+e3^2)  2*(e1*e2-e0*e3) 2*(e1*e3+e0*e2);2*(e1*e2+e0*e3) 1-2*(e1^2+e3^2) 2*(e2*e3-e0*e1);2*(e1*e3-e0*e2) 2*(e2*e3+e0*e1) 1-2*(e1^2+e2^2)];
        
        % Calcul du volume et de l'inertie de l'ellipsoide 2 dans le repère
        % global
        Volume2 = (4*pi/3)*a2*b2*c2;
        I2 = (Volume2/5)*[b2^2+c2^2 0 0;0 a2^2+c2^2 0;0 0 a2^2+b2^2];
        I2 = R2*I2*R2';
        
    end
end


if p~=0 && q~=0
    
    % Calcul du vecteur normal (allant de A à B) en déterminant le gradient
    [At] = matrice(0,evenement{3}(p,:),evenement{4}(p,:),evenement{5}(p,:));
    n = 2*At(1:3,1:3)*rc' + (At(1:3,4) + At(4,1:3)');
    n = n'/norm(n);
    % Calcul des vecteur t1 t2 tel que t1,t2 et n forment une base
    % orthonormée
    if n(2) ~= 0
        t1(1) = 1;
        t1(2) = -n(1)/n(2);
    else
        t1(1) = 0;
        t1(2) = 1;
    end
    t1(3) = 0;
    t1 = t1/norm(t1);
    t2 = [n(2)*t1(3)-n(3)*t1(2), n(3)*t1(1)-n(1)*t1(3), n(1)*t1(2)-n(2)*t1(1)];
    
    % Initialisation du système Inconnue1*X = Inconnue2
    Inconnue1 = zeros(12,12);
    Inconnue2 = zeros(12,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Conservation de la quantité de mouvement de l'Ellipsoide 1 selon t1
    Inconnue1(1,1:3) = t1;
    Inconnue2(1) = Vitesse1(1)*t1(1) + Vitesse1(2)*t1(2) + Vitesse1(3)*t1(3);
    
    % Conservation de la quantité de mouvement de l'Ellipsoide 1 selon t2
    Inconnue1(2,1:3) = t2;
    Inconnue2(2) = Vitesse1(1)*t2(1) + Vitesse1(2)*t2(2) + Vitesse1(3)*t2(3);
    
    % Conservation de la quantité de mouvement de l'Ellipsoide 2 selon t1
    Inconnue1(3,4:6) = t1;
    Inconnue2(3) = Vitesse2(1)*t1(1) + Vitesse2(2)*t1(2) + Vitesse2(3)*t1(3);
    
    % Conservation de la quantité de mouvement de l'Ellipsoide 2 selon t2
    Inconnue1(4,4:6) = t2;
    Inconnue2(4) = Vitesse2(1)*t2(1) + Vitesse2(2)*t2(2) + Vitesse2(3)*t2(3);
    
    % Conservation de la quantité de mouvement total selon n
    Inconnue1(5,1:3) = Volume1*n;
    Inconnue1(5,4:6) = Volume2*n;
    Inconnue2(5) = Volume1*(Vitesse1(1)*n(1)+Vitesse1(2)*n(2)+Vitesse1(3)*n(3)) + Volume2*(Vitesse2(1)*n(1)+Vitesse2(2)*n(2)+Vitesse2(3)*n(3));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Conservation du moment cinétique de l'Ellipsoide 1 autour du point de
    % contact
    
    % En x
    Inconnue1(6,2) = -Volume1*r1(3);
    Inconnue1(6,3) = Volume1*r1(2);
    Inconnue1(6,7:9) = I1(1,1:3);
    
    % En y
    Inconnue1(7,1) = Volume1*r1(3);
    Inconnue1(7,3) = -Volume1*r1(1);
    Inconnue1(7,7:9) = I1(2,1:3);
    
    % En z
    Inconnue1(8,1) = -Volume1*r1(2);
    Inconnue1(8,2) = Volume1*r1(1);
    Inconnue1(8,7:9) = I1(3,1:3);
    
    Inconnue2(6:8) = I1*Angulaire1' + (Volume1*[r1(2)*Vitesse1(3)-r1(3)*Vitesse1(2); r1(3)*Vitesse1(1)-r1(1)*Vitesse1(3); r1(1)*Vitesse1(2)-r1(2)*Vitesse1(1)]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Conservation du moment cinétique de l'Ellipsoide 2 autour du point de
    % contact
    
    % En x
    Inconnue1(9,5) = -Volume2*r2(3);
    Inconnue1(9,6) = Volume2*r2(2);
    Inconnue1(9,10:12) = I2(1,1:3);
    
    % En y
    Inconnue1(10,4) = Volume2*r2(3);
    Inconnue1(10,6) = -Volume2*r2(1);
    Inconnue1(10,10:12) = I2(2,1:3);
    
    % En z
    Inconnue1(11,4) = -Volume2*r2(2);
    Inconnue1(11,5) = Volume2*r2(1);
    Inconnue1(11,10:12) = I2(3,1:3);
    
    Inconnue2(9:11) = I2*Angulaire2' + (Volume2*[r2(2)*Vitesse2(3)-r2(3)*Vitesse2(2); r2(3)*Vitesse2(1)-r2(1)*Vitesse2(3); r2(1)*Vitesse2(2)-r2(2)*Vitesse2(1)]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Coefficient de restitution appliqué selon le vecteur n
    
    temp1 = Vitesse1 - [Angulaire1(2)*r1(3) - Angulaire1(3)*r1(2), Angulaire1(3)*r1(1) - Angulaire1(1)*r1(3), Angulaire1(1)*r1(2) - Angulaire1(2)*r1(1)]; %(V1 + cross(w1,-r1))
    temp2 = Vitesse2 - [Angulaire2(2)*r2(3) - Angulaire2(3)*r2(2), Angulaire2(3)*r2(1) - Angulaire2(1)*r2(3), Angulaire2(1)*r2(2) - Angulaire2(2)*r2(1)]; %(V2 + cross(w2,-r2))
    temp = temp1 - temp2;
    Inconnue2(12) = - (temp(1)*n(1) + temp(2)*n(2) + temp(3)*n(3)) - 2*(a0_1+a0_2); % 2*(a0_1+a0_2) pour s'assurer que les 2 ellipsoides ne se toucheront pas à t+dt.
    
    Inconnue1(12,1:3) = n;  % V1.n
    Inconnue1(12,4:6) = -n; %-V2.n
    
    Inconnue1(12,7) = n(2)*r1(3) - n(3)*r1(2);
    Inconnue1(12,8) = n(3)*r1(1) - n(1)*r1(3);      % (w1xr1).n
    Inconnue1(12,9) = n(1)*r1(2) - n(2)*r1(1);
    
    Inconnue1(12,10) = - (n(2)*r2(3) - n(3)*r2(2));
    Inconnue1(12,11) = - (n(3)*r2(1) - n(1)*r2(3)); %-(w2xr2).n
    Inconnue1(12,12) = - (n(1)*r2(2) - n(2)*r2(1));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Calcul de la solution
    Solution = inv(Inconnue1)*Inconnue2;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Normalisation de toutes les vitesses angulaires des ellipsoides.
    % Ceci éviterai d'avoir de très grande vitesses.
    
    % norme1 = sqrt(Vitesse(:,1).^2 + Vitesse(:,2).^2 +Vitesse(:,3).^2);
    norme2 = sqrt(Vitesse(:,4).^2 + Vitesse(:,5).^2 +Vitesse(:,6).^2);
    
    % Vitesse(:,1:3) = sqrt(3)*Vitesse(:,1:3) ./ repmat(norme1,1,3);
    Vitesse(:,4:6) = 0.1*sqrt(3)*Vitesse(:,4:6) ./ repmat(norme2,1,3);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Mise à jour de la matrice Vitesse
    Vitesse(p,1:3)   = Solution(1:3);
    Vitesse(p,4:6)   = Solution(7:9);
    Vitesse(q,1:3)   = Solution(4:6);
    Vitesse(q,4:6)   = Solution(10:12);
    
    % Copie des vecteurs vitesse pour les partenaires de l'ellipsoide 1
    for j=1:sum(partenaire_1~=0)
        for k=1:size(Identite,1)
            if Identite(k) == partenaire_1(j)
                Vitesse(k,:)= Vitesse(p,:);
                break
            end
        end
    end
    
    % Copie des vecteurs vitesse pour les partenaires de la sphere 1
    for j=1:sum(partenaire_2~=0)
        for k=1:size(Identite,1)
            if Identite(k) == partenaire_2(j)
                Vitesse(k,:)= Vitesse(q,:);
                break
            end
        end
    end
    
    
end

% Copie de l'ensemble de cellule
nouveau_evenement = evenement;

% Insertion de la matrice Vitesse dans l'ensemble de cellule
nouveau_evenement{5} = Vitesse;


