function [Temps_collision, check] = interaction(evenement,t_cube,NNL,check)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [Temps_collision] = interaction(evenement)
%
%% Description :
%  Cette fonction permet de calculer le temps de la prochaine collision les
%  numéros des ellipsoides qui y sont impliquées ainsi que les coordonnées
%  du point de contact.
%
%% Variable(s) d'entrée
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides
%  t_cube   : Prochain temps de collision entre un ellipsoide et un cube
%  NNL      : Si NNL = 0, la technique NNL n'est pas utilisé pour les
%             collisions. Si NNL = 1, la technique est utilisée
%% Variable de sortie:
%  Temps_collision: Le prochain temps de collision, les ellipsoides
%                   qui y sont impliquées et les coordonnées du point de
%                   contact : Temps_collision = [t, i, j, [x, y, z]]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

%% Calcul des collisions entre les sphères qui englobent les ellispoides.

Rayon = evenement{3};
Position = evenement{4};
Vitesse = evenement{5};

% Calcul des matrices rx, ry et rz où chaque composante (i,j) est la
% différence de position entre les sphères i et j

% Extraction de la matrice position
rx = Position(:,1);
ry = Position(:,2);
rz = Position(:,3);

% Répétition des colonne de chaque matrice
rx = repmat(rx,1,size(Position,1));
ry = repmat(ry,1,size(Position,1));
rz = repmat(rz,1,size(Position,1));

% Obtention des matrices symétriques
rx = rx - rx';
ry = ry - ry';
rz = rz - rz';

% Calcul des matrices vx, vy et vz où chaque composante (i,j) est la
% différence de vitesse entre les sphères i et j

% Extraction de la matrice vitesse
vx = Vitesse(:,1);
vy = Vitesse(:,2);
vz = Vitesse(:,3);

% Répétition des colonne de chaque matrice
vx = repmat(vx,1,size(Vitesse,1));
vy = repmat(vy,1,size(Vitesse,1));
vz = repmat(vz,1,size(Vitesse,1));

% Obtention des matrices symétriques
vx = vx - vx';
vy = vy - vy';
vz = vz - vz';

% Calcul de la matrice symmétrique des taux de croissance des rayons où
% chaque composante (i,j) est la somme des taux de croissance des sphères i
% et j. Pour cela, il faut choisir le taux de croissance maximal pour
% chaque ellipsoide

a = evenement{3}(:,4:6);
a = max(a,[],2);
a = repmat(a,1,size(a,1));
a = a + a';

% Calcul des paramètres A, B et C
Temps = evenement{2}(1,1);

A = (vx.^2 + vy.^2 + vz.^2) - (a.^2);
B = (rx.*vx + ry.*vy + rz.*vz) - ( (a.^2)*Temps );
C = (rx.^2 + ry.^2 + rz.^2) - ( (a.^2)*Temps^2 );

% Calcul de la matrice symmétrique des temps de collision.
t_spheres = (-B - sqrt((B.^2)-(A.*C)) )./A;
% Si les sphères ne se toucheront jamais
t_spheres( (B>0 & A>=0) | (((B.^2)-(A.*C)) < 0) ) = inf;
% Si les sphères se touchent déjà
t_spheres( (rx.^2 + ry.^2 + rz.^2) < (a.*Temps).^2 ) = 0;
% Element de la diagonale
t_spheres(1:length(t_spheres)+1:end) = inf;

%% Calcul des temps de collisions entre les ellipsoides susceptible de se
%% toucher, c'est à dire lorsque les sphères se touchent déjà ou bien vont
%% se toucher bientôt. En d'autres termes lorsque t_spheres ~= inf.

options= optimset('Display','off','TolFun',1e-15,'TolX',1e-15,'Maxiter',inf,'MaxFunevals',inf);

% Extraction des matrices de l'ensemble de cellule check
temps_ellipsoides_1 = check{1,1};
ellipsoides_check_1 = check{1,2};
%ellipsoides_check_1(isinf(t_spheres)==0 & isinf(temps_ellipsoides_1)==1) = 1;

% Initialisation du temps minimum
tmin  = t_cube;

% Cutoff mu_cut pour construire les NNL
mu_cut = 1.2;

% Marge
Marge = 1e-1;

for i = 1:size(Rayon,1)-1
    
    Rayon_A = Rayon(i,:);
    Position_A = Position(i,:);
    Vitesse_A = Vitesse(i,:);
    
    for j = i+1:size(Rayon,1)
        
        % Extraction des vecteur pertinents aux ellipsoides i et j
        
        Rayon_B = Rayon(j,:);
        Position_B = Position(j,:);
        Vitesse_B = Vitesse(j,:);
        
        if ellipsoides_check_1(i,j) == 1
            
            % Verifier si les NNL se touchent
            if t_spheres(i,j) < tmin  && (NNL == 0 || verifier_collision(mu_cut*Rayon_A,mu_cut*Rayon_B,Position_A,Position_B) == 1)
                
                % Recherche numérique du temps de collision
                %[dt,valeur] = fsolve(@(dt) difference(dt,Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B),sqrt(t_spheres(i,j)),options);
                [dt,valeur] = fzero(@(dt) polynome(dt,Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B,1),sqrt(t_spheres(i,j)),options);
                
                % Vérification qu'il s'agit bien d'un temps de collision
                sr = polynome(dt,Rayon_A,Rayon_B,Position_A,Position_B,Vitesse_A,Vitesse_B,3);
                if (sr(1) > 0 && sr(3) > 0 && sr(4) < 0 && abs(sr(5)) < Marge) || (sr(1) > 0 && sr(2) < 0 && abs(sr(3)) < Marge && abs(sr(5)) < Marge)
                    temps_ellipsoides_1(i,j) = dt^2;
                else
                    temps_ellipsoides_1(i,j) = inf;
                end
                
            else
                temps_ellipsoides_1(i,j) = inf;
            end
            
        end
        
        % Mise à jour du temps minimum
        tmin = min(tmin,temps_ellipsoides_1(i,j));
        
    end
end

% Formation de la matrice symétrique temps_ellipsoides_1
temps_ellipsoides_1(tril(temps_ellipsoides_1)>0) = 0; 
temps_ellipsoides_1 = temps_ellipsoides_1 + temps_ellipsoides_1';
temps_ellipsoides_1(1:length(temps_ellipsoides_1)+1:end) = inf;
% Pour éviter de trouver des collisions qui ont déjà eu lieu
temps_ellipsoides_1(temps_ellipsoides_1 < 1e-10) = inf;

% Calcul du temps minimum et des ellipsoides impliquées
% tmin = min(min(temps_ellipsoides_1));
[min1, indice1] = min(temps_ellipsoides_1);
[min2, indice2] = min(min1);
i = indice1(indice2);
j = indice2;
tmin = min2;

% Constrution de la matrice Temps_collision
if tmin ~= inf
    
    %    % Initialisation de la matrice Temps_collision
    %    Temps_collision = zeros(5,6);
    %
    %    % Vérifier s'il y a eu plusieurs collisions simultanée,
    %    m = 0;
    %     for i = 1:length(temps_ellipsoides_1)-1
    %         for j = i+1:length(temps_ellipsoides_1)
    %
    %             if abs(temps_ellipsoides_1(i,j) -tmin) <=1e-15 && (sum(Identite(i) == Partenaire(p,:)) == 0 || sum(Identite(i) == Partenaire(q,:)) == 0)
    %                 m = m+1;
    
    % Calcul du point de contact
    % Extraction des vecteur pertinents aux ellipsoides i et j qui sont
    % rentrés en collision
    Rayon_A = Rayon(i,:);
    Rayon_B = Rayon(j,:);
    Position_A = Position(i,:);
    Position_B = Position(j,:);
    Vitesse_A = Vitesse(i,:);
    Vitesse_B = Vitesse(j,:);
    % Calcul de A et B au tmeps de collision
    A = matrice(tmin,Rayon_A,Position_A,Vitesse_A);
    B = matrice(tmin,Rayon_B,Position_B,Vitesse_B);
    % Calcul des vecteur propre de A^-1*B
    [P,D] = eig(B,A);
    D = real(D);
    % Recherche du vecteur propre qui correspond à la plus petite racine
    % (i.e les racine négatifs
    [min3, indice3] = min(min(D));
    vecteur = real(P(:,indice3));
    
    Temps_collision= [tmin,evenement{1}(i),evenement{1}(j) vecteur(1:3)'/vecteur(4)];
    
    %
    %             end
    %
    %         end
    %     end
    %
    %     % Elimination des lignes non utlisés dans la matrice Temps_collision
    %     Temps_collision = Temps_collision(1:m,:);
    %
else
    Temps_collision=[];
end

% Update de l'ensemble de cellule check
check{1,1} = temps_ellipsoides_1;
check{1,2} = ellipsoides_check_1;

