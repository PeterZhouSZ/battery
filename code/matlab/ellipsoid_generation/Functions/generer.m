clc
clear all
warning off

rng shuffle

%matlabpool open 8

%Saisie des données
nb_Ellipsoides = 30;
R1 = 5;
R2 = 5;
fraction = 0.3;

% Nombre de microstructure à générer
n = 40;

for m = 1:n
    
    fprintf('Generation of Microstructure no %i\n',m)
    warning off
    % rng(sum(1e6*clock)+1e3*m)
    
    % Calcul du rayon final maximal des ellipsoides
    rayon_final = ( (fraction/nb_Ellipsoides)*R1*R2*3/(4*pi) )^(1/3);
    
    % Initialisation    
    [A] = initialiser(nb_Ellipsoides,R1,R2);
    
    nb_Iterations = 0;
    Fraction_volumique = 0;
    
    % Créatin des matrices temps_elipsoides et ellipsoides_check
    check{1,1} = zeros(nb_Ellipsoides,nb_Ellipsoides);
    check{1,2} = ones(nb_Ellipsoides,nb_Ellipsoides);
    check{2,1} = zeros(nb_Ellipsoides,6);
    check{2,2} = ones(nb_Ellipsoides,6);
        
    while (Fraction_volumique - fraction) < 0
        
        nb_Iterations = nb_Iterations + 1;
        
        % Copie de l'ensemble de cellule
        B = A;
        
        % Calcul du prochain évènement
        [Temps_prochain, check] = prochain_evenement(A,nb_Ellipsoides,check);
        
        % Mise à jour de tous les ellipsoides
        A = avancer(A,Temps_prochain{1});
        
        % Vérification et resimulation
        [A,Temps_prochain,check] = verification(A,B,nb_Ellipsoides,Temps_prochain,check);
        
        % Mise à jour de l'évènement
        A = update_evenement(A, Temps_prochain, nb_Ellipsoides);
        
        % Vérifier si des ellipsoides sont complétement à l'intérieur ou à
        % l'extérieur du cube
        A = interne_externe(A,nb_Ellipsoides);
        
        % Mise à jour des ellipsoides à checker (check)
        check = check_update(check,A,B,Temps_prochain);
        
        % Calcul de la fraction volumique
        Fraction_volumique = fraction_volumique(A,nb_Ellipsoides);
        
    end
    
    % Calcul du temps restant et avancement des ellipsoides
    tps = ( rayon_final - B{3}(1,1) )/B{3}(1,4);
    A = avancer(B,tps);
    
    % Fraction volumique finale
    Fraction_volumique = fraction_volumique(A,nb_Ellipsoides);
    
    % Tracage des ellipsoides dans le volume élémentaire
    % fprintf('Plotting the ellipsoids\n')
    %tracer(A);
    
    % Enregistrement des données
    fichier = ['N=' num2str(nb_Ellipsoides) '_R1=' num2str(R1) '_R2=' num2str(R2) '_f=' num2str(100*fraction) '_' num2str(m)];
    mysave(fichier,A,R1,R2,nb_Ellipsoides,fraction)
    

end

% matlabpool close
