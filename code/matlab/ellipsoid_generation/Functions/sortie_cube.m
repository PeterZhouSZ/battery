function [Temps_cube, check] = sortie_cube(evenement,nb_Ellipsoides,check)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Temps_cube] = sortie_cube(evenement,nb_Ellipsoides)
%
%% Description :
%  Cette fonction permet de calculer le temps de la prochaine sortie du
%  cube par un ellipsoide
%
%% Variable(s) d'entrée :
%  evenement     : ensemble de cellule contenant l'ensemble des ellipsoidess
%  nb_Ellipsoides: Nombre d'ellipsoides

%% Variable de sortie:
%  Temps_cube    : Ensemble de cellule contenant le prochain temps de
%                  sortie du cube, l'ellipsoide impliquée et l'endroit par où
%                  il est sorti du cube.
%                  ************************exemple:************************
%                  [0 1 0 0 0 0] signifie qu'il est sorti par la face
%                  x_sup du cube; [0 0 0 0 1 0] signifie qu'il est sorti par la
%                  face z_inf du cube.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

options= optimset('Display','off','TolFun',1e-15,'TolX',1e-15,'Maxiter',inf,'MaxFunevals',inf);

% Extraction des matrices de l'ensemble de cellule.
Identite = evenement{1}(1:nb_Ellipsoides,:);
Rayon = evenement{3}(1:nb_Ellipsoides,:);
Position = evenement{4}(1:nb_Ellipsoides,:);
Vitesse = evenement{5}(1:nb_Ellipsoides,:);
Code = evenement{6}(1:nb_Ellipsoides,:);


% Verifier si la sphère englobante touchera la face du cube. Si
% oui, le point de départ sera le temps de collision entre la
% sphère et la face. Sinon, il est certain qu'il n'y aura pas de
% collision entre l'ellipsoide et la face du cube.

% Initialisation de la matrice dt0 [Nx6]
dt0 = zeros(nb_Ellipsoides,6);

% Détermination du rayon maximal de la sphère qui englobe l'ellipsoide
Rayon_max = repmat(max(Rayon(:,1:3),[],2),1,3);
taux_max = repmat(max(Rayon(:,4:6),[],2),1,3);

%Collision entre les BS et les faces négatives
dt0_1 = (Position(:,1:3) - Rayon_max)./(taux_max - Vitesse(:,1:3));
%Collision entre les BS et les faces posities
dt0_2 = (1 - Position(:,1:3) - Rayon_max)./(taux_max + Vitesse(:,1:3));

% Si les BS touchent déjà les faces, alors dt0 = 0.
dt0_1(Position(:,1:3) <= Rayon_max) = 0;
dt0_2(1 - Position(:,1:3) <= Rayon_max) = 0;

% Regroupement des deux matrices
dt0(:,[1 3 5]) = dt0_1;
dt0(:,[2 4 6]) = dt0_2;

% Si un temps est négatif, dt0 = inf
dt0(dt0 < 0) = inf;



% Calcul pour chaque sphère, les 6 temps nécéssaire pour sortir
% respectivement des 6 faces du cube.
% Une sphère sort d'une face du cube lorsque la distance entre son centre
% et et cette face est inférieure à son rayon.

% Extraction des matrices de l'ensemble de cellule check
temps_ellipsoides_2 = check{2,1};
ellipsoides_check_2 = check{2,2};

% Identification des ellipsoides supplémentaires à checker. C'est ceux dont
% le dt0 est défini et ou le temps était inf.
%ellipsoides_check_2(isinf(dt0)==0 & isinf(temps_ellipsoides_2)==1) = 1;

% Initialisation du temps minimum
tmin = realmax;

% Marge
Marge = 1e-6;

for i = 1:nb_Ellipsoides
    
    % Extraction des vecteur pertinents à l'ellipsoide i
    Rayon_A = Rayon(i,:);
    Position_A = Position(i,:);
    Vitesse_A = Vitesse(i,:);
    
    
    %% Face x_inf
    
    if Code(i,1) == 0 && Code(i,2) == 0
        
        if ellipsoides_check_2(i,1) == 1
            
            if dt0(i,1) <= tmin 
                
                % Recherche numérique du temps de collision
                [dt,valeur] = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,1,0),sqrt(dt0(i,1)),options);
                
                % Vérification qu'il s'agit bien d'un temps de collision
                if abs(valeur) < Marge
                    temps_ellipsoides_2(i,1) = dt^2;
                else
                    temps_ellipsoides_2(i,1) = inf;
                end
            else
                temps_ellipsoides_2(i,1) = inf; %dt0(i,1);
            end
            
        end
        
    else
        temps_ellipsoides_2(i,1) = inf;
    end
    
    % Mise à jour du temps minimum
    tmin = min(tmin,temps_ellipsoides_2(i,1));
    
    
    
    %% Face x_sup
    
    if Code(i,1) == 0 && Code(i,2) == 0
        
        if ellipsoides_check_2(i,2) == 1
            
            if dt0(i,2) <= tmin 
                
                % Recherche numérique du temps de collision
                [dt,valeur] = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,1,1),sqrt(dt0(i,2)),options);
                
                % Vérification qu'il s'agit bien d'un temps de collision
                if abs(valeur) < Marge
                    temps_ellipsoides_2(i,2) = dt^2;
                else
                    temps_ellipsoides_2(i,2) = inf;
                end
            else
                temps_ellipsoides_2(i,2) = inf; %dt0(i,2);
            end
            
        end
        
    else
        temps_ellipsoides_2(i,2) = inf;
    end
    
    % Mise à jour du temps minimum
    tmin = min(tmin,temps_ellipsoides_2(i,2));
    
    
    
    %% Face y_inf
    
    if Code(i,3) == 0 && Code(i,4) == 0
        
        if ellipsoides_check_2(i,3) == 1
            
            if dt0(i,3) <= tmin 
                
                % Recherche numérique du temps de collision
                [dt,valeur] = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,2,0),sqrt(dt0(i,3)),options);
                
                % Vérification qu'il s'agit bien d'un temps de collision
                if abs(valeur) < Marge
                    temps_ellipsoides_2(i,3) = dt^2;
                else
                    temps_ellipsoides_2(i,3) = inf;
                end
            else
                temps_ellipsoides_2(i,3) = inf; %dt0(i,3);
            end
            
        end
        
    else
        temps_ellipsoides_2(i,3) = inf;
    end
    
    % Mise à jour du temps minimum
    tmin = min(tmin,temps_ellipsoides_2(i,3));
    
        
    
    %% Face y_sup
    
    if Code(i,3) == 0 && Code(i,4) == 0
        
        if ellipsoides_check_2(i,4) == 1
            
            if dt0(i,4) <= tmin 
                
                % Recherche numérique du temps de collision
                [dt,valeur] = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,2,1),sqrt(dt0(i,4)),options);
                
                % Vérification qu'il s'agit bien d'un temps de collision
                if abs(valeur) < Marge
                    temps_ellipsoides_2(i,4) = dt^2;
                else
                    temps_ellipsoides_2(i,4) = inf;
                end
            else
                temps_ellipsoides_2(i,4) = inf; %dt0(i,4);
            end
            
        end
        
    else
        temps_ellipsoides_2(i,4) = inf;
    end
    
    % Mise à jour du temps minimum
    tmin = min(tmin,temps_ellipsoides_2(i,4));
    
    
    
    %% Face z_inf
    
    if Code(i,5) == 0 && Code(i,6) == 0
        
        if ellipsoides_check_2(i,5) == 1
                
            if dt0(i,5) <= tmin 
                
                % Recherche numérique du temps de collision
                [dt,valeur] = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,3,0),sqrt(dt0(i,5)),options);
                
                % Vérification qu'il s'agit bien d'un temps de collision
                if abs(valeur) < Marge
                    temps_ellipsoides_2(i,5) = dt^2;
                else
                    temps_ellipsoides_2(i,5) = inf;
                end
            else
                temps_ellipsoides_2(i,5) = inf; %dt0(i,5);
            end
        end
        
    else
        temps_ellipsoides_2(i,5) = inf;
    end
    
    % Mise à jour du temps minimum
    tmin = min(tmin,temps_ellipsoides_2(i,5));
    
    
    
    %% Face z_sup
    
    if Code(i,5) == 0 && Code(i,6) == 0
        
        if ellipsoides_check_2(i,6) == 1

            if dt0(i,6) <= tmin 
                
                % Recherche numérique du temps de collision
                [dt,valeur] = fzero(@(dt) minimum(dt,Rayon_A,Position_A, Vitesse_A,3,1),sqrt(dt0(i,6)),options);
                
                % Vérification qu'il s'agit bien d'un temps de collision
                if abs(valeur) < Marge
                    temps_ellipsoides_2(i,6) = dt^2;
                else
                    temps_ellipsoides_2(i,6) = inf;
                end
            else
                temps_ellipsoides_2(i,6) = inf; %dt0(i,6);
            end
            
        end
        
    else
        temps_ellipsoides_2(i,6) = inf;
    end
    
    % Mise à jour du temps minimum
    tmin = min(tmin,temps_ellipsoides_2(i,6));
    
    
    
end


% Recherche de la position du minimum
[min1, indice1] = min(temps_ellipsoides_2);
[min2, indice2] = min(min1);
i = indice1(indice2);
j = indice2;
tmin = min2;

% Création de l'ensemble de cellule Temps_cube
Temps_cube = cell(1,3);
Temps_cube{1} = tmin;
Temps_cube{2} = Identite(i);
Temps_cube{3} = zeros(1,6);
Temps_cube{3}(j) = 1;

% Update de l'ensemble de cellule check
check{2,1} = temps_ellipsoides_2;
check{2,2} = ellipsoides_check_2;

% % Création de l'ensemble de cellule Temps_cube
% Temps_cube =cell(floor(size(temps_ellipsoides_2,1)/2),3);
%
% % Vérification s'il existe plusieurs temps minimals
% m = 0;
% for j = 1:size(temps_ellipsoides_2,1)
%
%     % Vérfication si 2 ellipsoides quittent plusieurs face en même
%     % temps
%     if abs(min(temps_ellipsoides_2(j,:)) - tmin) <=1e-15
%
%         m=m+1;
%         Temps_cube{m,1} = tmin;
%         Temps_cube{m,2} = Identite(j);
%         Temps_cube{m,3} = zeros(1,6);
%
%         % Vérification si un ellipsoide quitte un seteur à différentes faces
%         for k=1:size(temps_ellipsoides_2,2)
%             if abs(temps_ellipsoides_2(j,k) -tmin) <=1e-15
%                 Temps_cube{m,3}(k) = 1;
%             end
%         end
%
%     end
%
% end
%
% % Elimination des lignes non utlisés dans l'ensemble de cellule
% % Temps_cube
% Temps_cube = Temps_cube(1:m,:);

