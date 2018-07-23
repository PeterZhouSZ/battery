clear all
close all
clc
warning off
rng shuffle
cd Functions

% Input Datas (with validation)
nb_Ellipsoides = 100;
fraction = 0.1;

R1_arr = [1 1 8];
R2_arr = [5 2 9];
R3_arr = 0.0:0.1:1;

[m_r n_r] = size(R1_arr);
[m_r3 n_r3] = size(R3_arr);

dataset_folder = 'D:\Battery_code\ShapeDataset\N100VF01\';

for iPosR = 1:1:n_r
    for iPosR3 =1:1:n_r3        
      
        close all
        clc 
     
        R1 = R1_arr(iPosR);
        R2 = R2_arr(iPosR);
        MD_val = R3_arr(iPosR3);
        
        generate_volume(dataset_folder, nb_Ellipsoides, fraction, R1, R2, MD_val)
    end
end
    
%%






