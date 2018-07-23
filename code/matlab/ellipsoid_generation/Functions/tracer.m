function [] = tracer(evenement)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function tracer(evenement)
%
%% Description :
%  Cette fonction permet de tracer l'ensemble des ellipsoides dans le cube
%
%% Variable(s) d'entrée :
%  evenement: ensemble de cellule contenant l'ensemble des ellipsoides

%% Variable de sortie:
% Pas de variable de sortie
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation

% Extraction des matrices de l'ensemble de cellule.
Rayon = evenement{3};
Position = evenement{4};

% Traçage du cube.

hold on

plot3([0,1],[0,0],[0,0]);
plot3([0,1],[1,1],[0,0]);
plot3([0,1],[1,1],[1,1]);
plot3([0,1],[0,0],[1,1]);

plot3([1,1],[1,0],[0,0]);
plot3([1,1],[1,0],[1,1]);
plot3([0,0],[1,0],[1,1]);
plot3([0,0],[1,0],[0,0]);

plot3([1,1],[0,0],[0,1]);
plot3([1,1],[1,1],[0,1]);
plot3([0,0],[0,0],[0,1]);
plot3([0,0],[1,1],[0,1]);

% Traçage des ellipsoides

n=50;

for i=1:size(Rayon,1)
  
    % Extraction des vecteur pertinents aux ellipsoides i et j
     Rayon_A = Rayon(i,:);
     Position_A = Position(i,:);
           
    [X,Y,Z] = ellipsoid(0,0,0,Rayon_A(1),Rayon_A(2),Rayon_A(3),n);
    
    % Calcul de la matrice caractéristique et la matrice de rotation 
    [A,R] = matrice_static(Rayon_A,Position_A);
     
    % Calcul de toutes les quadratiques résultants de l'intersection de
    % l'ellipsoide (i) avec les 6 faces du cube
    [A1, D1, F1] = quadratique_plan(A,1,0);
    [A2, D2, F2] = quadratique_plan(A,1,1);
    [A3, D3, F3] = quadratique_plan(A,2,0);
    [A4, D4, F4] = quadratique_plan(A,2,1);
    [A5, D5, F5] = quadratique_plan(A,3,0);
    [A6, D6, F6] = quadratique_plan(A,3,1);
    
    % Chaque point (Xi,Yi,Zi) généré avec la commande ellipsoid est tourné
    % par la matrice de rotation R
    for j = 1:n+1
        for k = 1:n+1
            temp = R*[X(j,k) Y(j,k) Z(j,k)]';
            X(j,k) = temp(1);
            Y(j,k) = temp(2);
            Z(j,k) = temp(3);
            
            % Translation de chaque point de l'ellipsoide
            X(j,k) = X(j,k) + Position(i,1);
            Y(j,k) = Y(j,k) + Position(i,2);
            Z(j,k) = Z(j,k) + Position(i,3);
            
            % On veut couper les parties des ellipsoides qui sortent du
            % cube par les 6 faces
            
            %% Face x négatif 
            if X(j,k) < 0
                X(j,k) = 0;
                Vecteur = [Y(j,k) Z(j,k)]';
                % Voir si le point est à l'extérieur de l'ellipse. Si oui,
                % il faut le ramener à la circonférence de l'ellipse
                if Vecteur'*A1*Vecteur + D1'*Vecteur + F1 > 0
                    % Recherche des points d'intersection entre l'ellipse
                    % et la droite reliant le point au centre de l'ellipse
                    [Point1 Point2] = ellipse_droite(A1,D1,F1,Vecteur);
                    % Recherche du point d'intersection le plus près du
                    % point. Les 2 distances sont calculés 
                    d1 = (Point1(1) - Vecteur(1))^2 + (Point1(2) - Vecteur(2))^2;
                    d2 = (Point2(1) - Vecteur(1))^2 + (Point2(2) - Vecteur(2))^2;
                    if d1 < d2
                        % Le point est plus proche du premier point d'intersection
                        Y(j,k) = Point1(1);
                        Z(j,k) = Point1(2);
                    else
                        % Le point est plus proche du second point d'intersection
                        Y(j,k) = Point2(1);
                        Z(j,k) = Point2(2);
                    end 
                end
            end
            
            %% Face x positif 
            if X(j,k) > 1
                X(j,k) = 1;
                Vecteur = [Y(j,k) Z(j,k)]';
                % Voir si le point est à l'extérieur de l'ellipse. Si oui,
                % il faut le ramener à la circonférence de l'ellipse
                if Vecteur'*A2*Vecteur + D2'*Vecteur + F2 > 0
                    % Recherche des points d'intersection entre l'ellipse
                    % et la droite reliant le point au centre de l'ellipse
                    [Point1 Point2] = ellipse_droite(A2,D2,F2,Vecteur);
                    % Recherche du point d'intersection le plus près du
                    % point. Les 2 distances sont calculés
                    d1 = (Point1(1) - Vecteur(1))^2 + (Point1(2) - Vecteur(2))^2;
                    d2 = (Point2(1) - Vecteur(1))^2 + (Point2(2) - Vecteur(2))^2;
                    if d1 < d2
                        % Le point est plus proche du premier point d'intersection
                        Y(j,k) = Point1(1);
                        Z(j,k) = Point1(2);
                    else
                        % Le point est plus proche du second point d'intersection
                        Y(j,k) = Point2(1);
                        Z(j,k) = Point2(2);
                    end 
                end
            end
            
            
            %% Face y négatif 
            if Y(j,k) < 0
                Y(j,k) = 0;
                Vecteur = [X(j,k) Z(j,k)]';
                % Voir si le point est à l'extérieur de l'ellipse. Si oui,
                % il faut le ramener à la circonférence de l'ellipse
                if Vecteur'*A3*Vecteur + D3'*Vecteur + F3 > 0
                    % Recherche des points d'intersection entre l'ellipse
                    % et la droite reliant le point au centre de l'ellipse
                    [Point1 Point2] = ellipse_droite(A3,D3,F3,Vecteur);
                    % Recherche du point d'intersection le plus près du
                    % point. Les 2 distances sont calculés
                    d1 = (Point1(1) - Vecteur(1))^2 + (Point1(2) - Vecteur(2))^2;
                    d2 = (Point2(1) - Vecteur(1))^2 + (Point2(2) - Vecteur(2))^2;
                    if d1 < d2
                        % Le point est plus proche du premier point d'intersection
                        X(j,k) = Point1(1);
                        Z(j,k) = Point1(2);
                    else
                        % Le point est plus proche du second point d'intersection
                        X(j,k) = Point2(1);
                        Z(j,k) = Point2(2);
                    end 
                end
            end
   
            
            %% Face y positif
            if Y(j,k) > 1
                Y(j,k) = 1;
                Vecteur = [X(j,k) Z(j,k)]';
                % Voir si le point est à l'extérieur de l'ellipse. Si oui,
                % il faut le ramener à la circonférence de l'ellipse
                if Vecteur'*A4*Vecteur + D4'*Vecteur + F4 > 0
                    % Recherche des points d'intersection entre l'ellipse
                    % et la droite reliant le point au centre de l'ellipse
                    [Point1 Point2] = ellipse_droite(A4,D4,F4,Vecteur);
                    % Recherche du point d'intersection le plus près du
                    % point. Les 2 distances sont calculés
                    d1 = (Point1(1) - Vecteur(1))^2 + (Point1(2) - Vecteur(2))^2;
                    d2 = (Point2(1) - Vecteur(1))^2 + (Point2(2) - Vecteur(2))^2;
                    if d1 < d2
                        % Le point est plus proche du premier point d'intersection
                        X(j,k) = Point1(1);
                        Z(j,k) = Point1(2);
                    else
                        % Le point est plus proche du second point d'intersection
                        X(j,k) = Point2(1);
                        Z(j,k) = Point2(2);
                    end 
                end
            end   
            
            
            %% Face z négatif 
            if Z(j,k) < 0
                Z(j,k) = 0;
                Vecteur = [X(j,k) Y(j,k)]';
                % Voir si le point est à l'extérieur de l'ellipse. Si oui,
                % il faut le ramener à la circonférence de l'ellipse
                if Vecteur'*A5*Vecteur + D5'*Vecteur + F5 > 0
                    % Recherche des points d'intersection entre l'ellipse
                    % et la droite reliant le point au centre de l'ellipse
                    [Point1 Point2] = ellipse_droite(A5,D5,F5,Vecteur);
                    % Recherche du point d'intersection le plus près du
                    % point. Les 2 distances sont calculés
                    d1 = (Point1(1) - Vecteur(1))^2 + (Point1(2) - Vecteur(2))^2;
                    d2 = (Point2(1) - Vecteur(1))^2 + (Point2(2) - Vecteur(2))^2;
                    if d1 < d2
                        % Le point est plus proche du premier point d'intersection
                        X(j,k) = Point1(1);
                        Y(j,k) = Point1(2);
                    else
                        % Le point est plus proche du second point d'intersection
                        X(j,k) = Point2(1);
                        Y(j,k) = Point2(2);
                    end 
                end
            end
            
            
            
            %% Face z positif
            if Z(j,k) > 1
                Z(j,k) = 1;
                Vecteur = [X(j,k) Y(j,k)]';
                % Voir si le point est à l'extérieur de l'ellipse. Si oui,
                % il faut le ramener à la circonférence de l'ellipse
                if Vecteur'*A6*Vecteur + D6'*Vecteur + F6 > 0
                    % Recherche des points d'intersection entre l'ellipse
                    % et la droite reliant le point au centre de l'ellipse
                    [Point1 Point2] = ellipse_droite(A6,D6,F6,Vecteur);
                    % Recherche du point d'intersection le plus près du
                    % point. Les 2 distances sont calculés
                    d1 = (Point1(1) - Vecteur(1))^2 + (Point1(2) - Vecteur(2))^2;
                    d2 = (Point2(1) - Vecteur(1))^2 + (Point2(2) - Vecteur(2))^2;
                    if d1 < d2
                        % Le point est plus proche du premier point d'intersection
                        X(j,k) = Point1(1);
                        Y(j,k) = Point1(2);
                    else
                        % Le point est plus proche du second point d'intersection
                        X(j,k) = Point2(1);
                        Y(j,k) = Point2(2);
                    end 
                end
            end
            
            
            
         end
     end

    % Traçage de l'ellipsoide
    surf(real(X),real(Y),real(Z),'EdgeColor','none','LineStyle','none','FaceColor','red')
    
end

%title('Random Generation of Ellipsoid');
xlabel('x');
ylabel('y');
zlabel('z')
%axis ([-0.5,1.5,-0.5,1.5,-0.5,1.5,-Inf,Inf])
axis ([-0.2,1.2,-0.2,1.2,-0.2,1.2,-Inf,Inf]);
camlight left;
lighting gouraud;
view([-65,10]);

hold off