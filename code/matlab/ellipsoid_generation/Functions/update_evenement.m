function [nouveau_evenement] = update_evenement(evenement, Temps_prochain, nb_Ellipsoides)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fonction : [nouveau_evenement] = update_evenement(evenement,Temps_prochain, nb_Ellipsoides)
%
%% Description : 
%  Cette fonction permet de mettre à jour l'ensemble de cellule evenement
%  suite à la réalisation de plusieurs événements simultanée
%
%% Variable(s) d'entrée :
%  evenement         : ensemble de cellule contenant l'ensemble des ellipsoides
%  Temps_prochain    : ensemble de cellule le tupe d'évènement ainsi que le
%                      ou les ellipsoides impliqués
%  nb_Ellipsoides    : Nombre d'ellipsoides
%
%% Variable de sortie:
%  nouveau_evenement : ensemble de cellule contenant l'ensemble des
%  ellipsoides mis à jour. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Implémentation
    
    if isequal(Temps_prochain{4}, 'collision') == 1
        
        nouveau_evenement = apres_collision(evenement,Temps_prochain{2},Temps_prochain{3},Temps_prochain{5});
               
    elseif isequal(Temps_prochain{4}, 'cube') == 1
        
        nouveau_evenement = apres_cube(evenement,Temps_prochain{2},Temps_prochain{3},nb_Ellipsoides);
        
    end
    


