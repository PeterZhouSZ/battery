function [ u ] = diffusion_dirichlet( b,u )
	% Algorithm parameters
	v1 = 2; % Number of pre smoothing steps.
	v2 = 2; % Number of post smoothing steps.
	cl = 5; % Coarsest level
	n_it = 5;

	% Stencil
	L = zeros(3,3,3);
	L(:,:,1) = [ 0 -1 0; -1 -2 -1; 0 -1 0];
	L(:,:,2) = [-1 -2 -1; -2 24 -2; -1 -2 -1];
	L(:,:,3) = [ 0 -1 0; -1 -2 -1; 0 -1 0];
	L = (1/6)*L;

	% Iteration loop
	for i = 1:n_it
		[u] = mgcyc_dir( u, b, L, v1, v2, cl);
	end
end


function [ u ] = mgcyc_dir(u, f, L, v1, v2, cl )
	[m n o] = size(u);
	
	% Boundary values
	Ux = u([1 m],:,:);
	Uy = u(:,[1 n],:);
	Uz = u(:,:,[1 o]);
	
	%% Pre-smoothing
	L_sm = L;
	L_sm(2,2,2) = 0;
	for j = 1:v1
		u = 1/L(2,2,2)*(f - convn(u,L_sm,'same'));
		
		% Dirichlet boundary conditions
		u([1 m],:,:) = Ux;
		u(:,[1 n],:) = Uy;
		u(:,:,[1 o]) = Uz;
	end

	%% Compute the residual
	df = f - convn(u,L,'same');
	
	%% Restriction
	% Define restriction operator
	KER = zeros(3,3,3);
	KER(:,:,1) = 1/28*[0 1 0;1 2 1;0 1 0];
	KER(:,:,2) = 1/28*[1 2 1;2 4 2;1 2 1];
	KER(:,:,3) = 1/28*[0 1 0;1 2 1;0 1 0];
	df = convn(df,KER,'same');
	df = 4*df(1:2:end,1:2:end,1:2:end);
	vc = gzeros(size(df));
	
	if (ceil(numel(u).^(1/3)) > cl) % Continue coaresning
		[vc] = mgcyc_dir(vc, df, L, v1, v2, cl );
	else % Solve the restriction equation.
		n_it = 3;
		[mc nc oc] = size(vc);
		for i = 1:n_it
			vc = 1/L(2,2,2)*(df - convn(vc,L_sm,'same'));
			% Dirichlet boundary conditions
			vc([1 mc],:,:) = 0;
			vc(:,[1 nc],:) = 0;
			vc(:,:,[1 oc]) = 0;
		end
	end

	clear df

	%% Interpolation
	vf = gzeros(size(u));
	vf(1:2:end ,1:2:end ,1:2:end ) = vc;
	clear vc

	% Define interpolation operator
	KER(:,:,1) = 1/8*[1 2 1;2 4 2;1 2 1];
	KER(:,:,2) = 1/8*[2 4 2;4 8 4;2 4 2];
	KER(:,:,3) = 1/8*[1 2 1;2 4 2;1 2 1];
	vf = convn(vf,KER,'same');

	%% Compute the corrected approximation
	u = u + vf;

	%% Post-smoothing
	for j = 1:v2
		u = 1/L(2,2,2)*(f - convn(u,L_sm,'same'));
		% Dirichlet boundary conditions
		u([1 m],:,:) = Ux;
		u(:,[1 n],:) = Uy;
		u(:,:,[1 o]) = Uz;
	end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerics
n = 8; % Grid size
nx = 2^n + 2; % Nr of grid points
n = 8; % Grid size
ny = 2^n + 2; % Nr of grid points
n = 6; % Grid size
nz = 2^n + 1; % Nr of grid points
h = 1/(nz-1); % Spatial step length
nt = 2000;
x = linspace( 0,(nx-1)/(nz-1) ,nx );
y = linspace( 0,(ny-1)/(nz-1) ,ny );
z = linspace( 0,1 ,nz );
LX = max(X(:)) - min(X(:)) - h;
LY = max(Y(:)) - min(Y(:)) - h;

%% Physics
dt = 5e-5;
Ra = 5e2;

%% Initialize
PRE = zeros(nx,ny,nz);
RHS_PRE = zeros(size(PRE));
V = zeros(size(PRE));
TEMP = (1-Z);
TEMP(2:end-1,2:end-1,2:end-1) = TEMP(2:end-1,2:end-1,2:end-1) + .1*rand(nx-2,ny-2,nz-2)-.05;

% Periodic boundary conditions
TEMP(1,:,:) = TEMP(end-1,:,:);
TEMP(:,1,:) = TEMP(:,end-1,:);

%% Stencils velocity
[GCOORD, ELEM2NODE] = generate_mesh_8_brick(x,y,z);
[Y X Z] = meshgrid(y,x,z);
[Kzs] = thermal3d_brick_mat(ELEM2NODE, GCOORD);
clear ELEM2NODE GCOORD

KERvx = zeros(3,3,3);
KERvx(1,2,2) = -1/(2*h);
KERvx(3,2,2) = 1/(2*h);

KERvy = zeros(3,3,3);
KERvy(2,1,2) = -1/(2*h);
KERvy(2,3,2) = 1/(2*h);

KERvz = zeros(3,3,3);
KERvz(2,2,1) = -1/(2*h);
KERvz(2,2,3) = 1/(2*h);

L = 1+(dt/(6*h^2))*24;

L_temp = zeros(3,3,3);
L_temp(:,:,1) = [ 0 -1 0; -1 -2 -1; 0 -1 0];
L_temp(:,:,2) = [-1 -2 -1; -2 24 -2; -1 -2 -1];
L_temp(:,:,3) = [ 0 -1 0; -1 -2 -1; 0 -1 0];
L_temp = (1/(6*h^2))*L_temp;
L_temp(2,2,2) = 0;

[KzsaT KzsaT_bot KzsaT_top] = Stencil_brick_3d(Kzs');

%% Boundary conditions
% Temperature (dirichlet)
TEMPz = TEMP(:,:,[1 nz]);

%% Allocate memory on the GPU
TEMP = gsingle(TEMP);
PRE = gsingle(PRE );
gforce(TEMP);
gforce(PRE );
%% Time loop
for i = 1:nt
	i
	%% Right hand side pressure
	TEMP(end ,:,:) = TEMP(2,:,:);
	TEMP(:,end ,:) = TEMP(:,2,:);
	DTDz = convn(TEMP,KzsaT,'same');

	% Periodic boundary conditions
	DTDz(1 ,:,:) = DTDz(end-1,:,:);
	DTDz(:,1 ,:) = DTDz(:,end-1,:);
	DTDz(end,:,:) = DTDz(2 ,:,:);
	DTDz(:,end,:) = DTDz(:,2 ,:);
	tmp = conv2(TEMP(:,:,1 ), KzsaT_top(:,:,2),'same') +
	conv2(TEMP(:,:,2 ), KzsaT_top(:,:,3),'same');
	DTDz(:,:,1 ) = tmp;
	tmp = conv2(TEMP(:,:,end-1), KzsaT_bot(:,:,1),'same') +
	conv2(TEMP(:,:,end), KzsaT_bot(:,:,2),'same');
	DTDz(:,:,end) = tmp;
	DTDz(1 ,:,1 ) = DTDz(end-1,:,1 );
	DTDz(:,1 ,1 ) = DTDz(:,end-1,1 );
	DTDz(1 ,:,end) = DTDz(end-1,:,end);
	DTDz(:,1 ,end) = DTDz(:,end-1,end);

	%% Pressure solver
	PRE = pressure_solver(-Ra/h*DTDz,0*PRE );

	%% Velocity field
	PRE(:,end,:) = PRE(:,2 ,:);
	PRE(end,:,:) = PRE(2 ,:,:);

	% x-direction
	% ---------------------------------------------------------------------
	V = -convn(PRE,KERvx,'same');

	% Periodic boundary conditions
	V(1 ,:,:) = V(end-1,:,:);
	V(:,1 ,:) = V(:,end-1,:)
	xadv = X-V*dt;
	% y-direction
	% ---------------------------------------------------------------------
	V = -convn(PRE,KERvy,'same');
	% Periodic boundary conditions
	V(1 ,:,:) = V(end-1,:,:);
	V(:,1 ,:) = V(:,end-1,:);
	yadv = Y-V*dt;
	% z-direction
	% ---------------------------------------------------------------------
	V = -(convn(PRE,KERvz,'same') - Ra*TEMP);
	% Dirichlet boundary conditions
	V(:,:,1 ) = 0;
	V(:,:,end) = 0;
	% Periodic boundary conditions
	V(1 ,:,:) = V(end-1,:,:);
	V(:,1 ,:) = V(:,end-1,:);
	zadv = Z-V*dt;
	TEMP_OLD = TEMP;

	%% Heat diffusion
	for j = 1:50
		TEMP(:,end,:) = TEMP(:,2,:);
		TEMP(end,:,:) = TEMP(2,:,:);
		TEMP = 1/L*(TEMP_OLD - dt*convn(TEMP,L_temp,'same'));

		% Dirichlet boundary conditions		
		TEMP(:,:,[1 nz]) = TEMPz;

		% Periodic boundary conditions
		TEMP(1,:,:) = TEMP(end-1,:,:);
		TEMP(:,1,:) = TEMP(:,end-1,:);
		TEMP(:,end,:) = TEMP(:,2,:);
		TEMP(end,:,:) = TEMP(2,:,:);
	end

	%% transfere data to the CPU
	TEMP = single(TEMP);
	xadv = single(xadv(1:end-1,1:end-1,:));
	yadv = single(yadv(1:end-1,1:end-1,:));
	zadv = single(zadv(1:end-1,1:end-1,:));

	%% Advection
	% Periodic boundary conditions
	xadv(xadv<min(X(:)) ) = xadv(xadv<min(X(:)) ) + LX;
	xadv(xadv>max(X(:))-h) = xadv(xadv>max(X(:))-h) - LX;
	yadv(yadv<min(Y(:)) ) = yadv(yadv<min(Y(:)) ) + LY;
	yadv(yadv>max(Y(:))-h) = yadv(yadv>max(Y(:))-h) - LY;
	TEMP(1:end-1,1:end-1,:) = interp3(Y,X,Z,TEMP,yadv,xadv,zadv,'linear');

	%% Save temperature
	if (mod(i,30) == 0)
		filename = ['temperature' num2str(i) '_257.mat'];
		%tmp_temp = TEMP(1:end-1,1:end-1,:);
		save(filename,'TEMP', 'V');
	end

	TEMP = gsingle(TEMP);
end