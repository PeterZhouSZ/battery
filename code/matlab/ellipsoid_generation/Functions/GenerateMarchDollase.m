function [ angles ] = GenerateMarchDollase( nb_Ellipsoides, r_param )

if (r_param == 0)
    angles = ones(nb_Ellipsoides,1);
else
    theta = 0:1:90;
    nominator = sind(theta);
    denominator = (r_param.*r_param.*cosd(theta).*cosd(theta) + (1./r_param).*sind(theta).*sind(theta)).^(3/2);
    prob = nominator./denominator;
    angles = sin(2*degtorad(randpdf(prob, theta, [nb_Ellipsoides, 1])) - 1);
end

end

