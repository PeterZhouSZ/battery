function area = ellipsoid_surface_area(a,b,c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function area = ellipsoid_surface_area(a,b,c)
%
%% Description : 
%  Quickly calculates the surface area of an ellipsoid.
%  When none of the radii of the ellipsoid are equal, uses a rough
%  approximation of the actual surface area.
%  (written by Ezra Davis)
%
%% Input variable(s) :
%  a, b, and c are all radii of the ellipsoid.
%
%% Output variables:
%  area: surface area of the ellipsoid
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Many of the equations are simply from Wikipedia:
% https://en.wikipedia.org/wiki/Ellipsoid

if(a==b && b == c) % If it's a sphere, it is very easy to calculate.
    area = 4*pi*a^2;
elseif(a==b || b == c || a == c)
    % We can use (exact) prolate or oblate spheroid calculations:
    if(a==b)
        b = c;
    elseif(a==c)
        % b = b;a=a
    elseif(b==c)
        b = a;
        a = c;
    end
    
    % Version from wikipedia:
    % Intended for an oblate spheroid: (b < a)
    e = sqrt(1-b^2/a^2); % Note that this is imaginary if b > a, but that works out in the next calculation.
    area = 2*pi*a^2 * (1+atanh(e)*(1-e^2)/e);

%     alpha = acos(a/b);
%     
%     if(a > b) % Prolate spheroid
%         area = 2*pi * (a^2 + a*b*alpha / sin(alpha))
%     else % Oblate spheroid
%         area = 2*pi*(a^2 + (b^2 / sin(alpha)) * log((1+sin(alpha))/cos(alpha)))
%     end
else
    % We are using an approximation of the actual surface area, which is
    % probably good enough.
    % Knud Thomsen's Approximation:
    % (the actual formula contains an integral, and this one will have at most 1.061% relative error)
    % http://www.web-formulas.com/Math_Formulas/Geometry_Surface_of_Ellipsoid.aspx
    p = 1.6075;
    area = 4*pi * ((a^p*b^p + a^p*c^p + b^p*c^p)/3)^(1/p);
end

end