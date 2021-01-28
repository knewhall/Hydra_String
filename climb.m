function [x,y,e1,exitflag,lambs] = climb(x_min,y_min,x_perturb,y_perturb,sigma,n_images)
%Climbs a single string when one sphere doesn't move

%   INPUT:
%     xmin and ymin contain the positions at a minimum in Np x 1 arrays
%     xsad and ysad contain the positions of perturbed particles in Np x 1 arrays
%     sigma contains radii information in an Np x Np matrix
%     n_images is the number of images on the string 
%   OUTPUT:
%     x and y contain the entire MEP string in Np x n_images arrays with the
%         x(1) being the x positions at the minimum and x(n_images) is the x
%         position at the saddle
%     exitflag is 1 if a saddle was converged to
%     lambs contains the eigenvalues of the hessian at the "saddle"
    
exitflag = 0;   %Changes to 1 if convergence reached
Np = length(x_min(:,1));   %Number of particles
nstepmax = 1e5;    %%Maximum number of time steps
% nn = 1e0;   %Frequency of plotting
dt = 1e-3;  %Time-step
amp = 2;    %nu parameter in the climbing string ODE
diff_tol1 = 1e-8;   %Parameter used as stopping criterion

%Preallocate matrices
eyesigma = eye(Np,Np).*sigma;
e1 = zeros(1,n_images);
dVx = zeros(Np,n_images);
dVy = zeros(Np,n_images);

%Initialize string with the specified number of images
g1 = linspace(0,1,n_images);
x = x_min*ones(1,n_images)+(x_perturb-x_min)*g1;
y = y_min*ones(1,n_images)+(y_perturb-y_min)*g1;

%Gradient descent for each image
for j = 1:n_images
    Dx = mod(x(:,j),1)*ones(1,Np)-ones(Np,1)*mod(x(:,j),1)';
    Dy = mod(y(:,j),1)*ones(1,Np)-ones(Np,1)*mod(y(:,j),1)';
    Dx = Dx.*and(Dx<0.5,Dx>-0.5) + (Dx-1).*(Dx>=0.5) + (1+Dx).*(Dx<=-0.5);
    Dy = Dy.*and(Dy<0.5,Dy>-0.5) + (Dy-1).*(Dy>=0.5) + (1+Dy).*(Dy<=-0.5);
    Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
    tmp = Dxy./sigma-1;
    h1 = (Dxy<=sigma).*tmp./Dxy./sigma;
    e1(j) = 0.5*sum(sum((Dxy<=sigma).*tmp.^2));
    dVx(:,j) = (h1.*Dx)*ones(Np,1);
    dVy(:,j) = (h1.*Dy)*ones(Np,1);
end

for nstep = 1:nstepmax
    
    %Compute force on climbing image
    
    %Unit tangent vector to string at the last point  [tx ty]
    txy = sqrt(sum((x(:,n_images)-x(:,n_images-1)).^2+(y(:,n_images)-y(:,n_images-1)).^2));  % dist between last 2 string elements
    tx = (x(:,n_images)-x(:,n_images-1))/txy;
    ty = (y(:,n_images)-y(:,n_images-1))/txy;
    dVxy = dVx(:,n_images)'*tx+dVy(:,n_images)'*ty;
    
    x0 = x; %Save previous string
    y0 = y;
    
    %Gradient descent the intermediate images
    x(2:Np,2:n_images-1) = x(2:Np,2:n_images-1) - dt*dVx(2:Np,2:n_images-1);
    y(2:Np,2:n_images-1) = y(2:Np,2:n_images-1) - dt*dVy(2:Np,2:n_images-1);
    
    %Move the climbing image
    x(2:Np,n_images) = x(2:Np,n_images) - dt*dVx(2:Np,n_images) + amp*dt*dVxy*tx(2:Np);
    y(2:Np,n_images) = y(2:Np,n_images) - dt*dVy(2:Np,n_images) + amp*dt*dVxy*ty(2:Np);
    
    %-----------------------------------
    %Reinterpola1te along string
    dx = x(:,2:n_images) - x(:,1:n_images-1);
    dy = y(:,2:n_images) - y(:,1:n_images-1);
    
    dd = sum(dx.^2+dy.^2);
    dd = sqrt([0 dd]);
    
    ll = cumsum(dd);
    ll = ll/ll(n_images);
    x = interp1q(ll',x',g1')';
    y = interp1q(ll',y',g1')';
    %-----------------------------------
    
    %Calculate the energy of every image along the string and gradient vector
    for j = 1:n_images
        Dx = mod(x(:,j),1)*ones(1,Np)-ones(Np,1)*mod(x(:,j),1)';
        Dy = mod(y(:,j),1)*ones(1,Np)-ones(Np,1)*mod(y(:,j),1)';
        Dx = Dx.*and(Dx<0.5,Dx>-0.5) + (Dx-1).*(Dx>=0.5) + (1+Dx).*(Dx<=-0.5);
        Dy = Dy.*and(Dy<0.5,Dy>-0.5) + (Dy-1).*(Dy>=0.5) + (1+Dy).*(Dy<=-0.5);
        Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
        tmp = Dxy./sigma-1;
        h1 = (Dxy<=sigma).*tmp./Dxy./sigma;
        e1(j) = 0.5*sum(sum((Dxy<=sigma).*tmp.^2));
        dVx(:,j) = (h1.*Dx)*ones(Np,1);
        dVy(:,j) = (h1.*Dy)*ones(Np,1);
    end
    
    %Check monotonicity. We want the string to always be monotonically
    %increasing
    de1 = e1(2:n_images)-e1(1:n_images-1);

    %Find all indices where monotonicity fails (if any) then take the first instance of non-monotonicity
    nn1 = find(de1<0); nn1=nn1(find(nn1>2,1));   
    
    if (~isempty(nn1) && nn1>2) %If non-monotonic, cut string and reinterpolate
%        fprintf('string was truncated at iteration %d to length %d\n',nstep,nn1);
        
        x = x(:,1:nn1);
        y = y(:,1:nn1);
        
        dx = x(:,2:nn1) - x(:,1:nn1-1);
        dy = y(:,2:nn1) - y(:,1:nn1-1);
        
        dd = sum(dx.^2+dy.^2);
        dd = sqrt([0 dd]);
        
        ll = cumsum(dd);
        ll = ll/ll(nn1);
        x = interp1q(ll',x',g1')';
        y = interp1q(ll',y',g1')';
        
        for j = 1:n_images
            Dx = mod(x(:,j),1)*ones(1,Np)-ones(Np,1)*mod(x(:,j),1)';
            Dy = mod(y(:,j),1)*ones(1,Np)-ones(Np,1)*mod(y(:,j),1)';
            Dx = Dx.*and(Dx<0.5,Dx>-0.5) + (Dx-1).*(Dx>=0.5) + (1+Dx).*(Dx<=-0.5);
            Dy = Dy.*and(Dy<0.5,Dy>-0.5) + (Dy-1).*(Dy>=0.5) + (1+Dy).*(Dy<=-0.5);
            Dxy = sqrt(Dx.^2 + Dy.^2)+eye(Np,Np).*sigma;
            h1 = (Dxy<=sigma).*(Dxy./sigma-1)./Dxy./sigma;
            e1(j) = 0.5*sum(sum((Dxy<=sigma).*(1-Dxy./sigma).^2));
            dVx(:,j) = (h1.*Dx)*ones(Np,1);
            dVy(:,j) = (h1.*Dy)*ones(Np,1);
        end
        
        continue
        
    end
    
%-------------------------------------------------------------------    
% This creates plots of the energy as the string evolves. This can be
% useful when debugging this function.
%
%     if mod(nstep,nn)==0
%         figure(11);
%         
%         dx = x(:,2:n1) - x(:,1:n1-1);
%         dy = y(:,2:n1) - y(:,1:n1-1);
%         
%         dd = sum(dx.^2+dy.^2);
%         ll = cumsum(sqrt([0 dd]));
%         
%         % plot(dd,e1,'ro')
%         % plot(ll,e1,'ro')
%         title(sprintf('step: %g',nstep))
%         drawnow
%     end
%-------------------------------------------------------------------

    %Check stopping criteria
      
        errX = max(abs(x-x0));
        errY = max(abs(y-y0));

      if max([errX errY]) < diff_tol1
        exitflag = 1;
        break
      end
      
end

%Check eigenvalues

%Hessian Matrix
hxx = (Dxy<sigma).*( (Dxy./sigma-1)./Dxy + Dx.^2./Dxy.^3 )./sigma;
hyy = (Dxy<sigma).*( (Dxy./sigma-1)./Dxy + Dy.^2./Dxy.^3 )./sigma;
hxy = (Dxy<sigma).*Dx.*Dy./Dxy.^3./sigma;

H0 = [diag(hxx*ones(Np,1))-hxx diag(hxy*ones(Np,1))-hxy;...
    diag(hxy*ones(Np,1))-hxy diag(hyy*ones(Np,1))-hyy];

if (any(isnan(H0(:))) || any(isinf(H0(:))))
    lambs = 1;
else
    lambs = eig(H0);
end

if exitflag && (sum(lambs < -1e-10)~= 1)    %not a single positive eigenvalue i.e. not sad
    exitflag = 0;
    
elseif exitflag && x(1,1) == 0  %something broke so exitflag is 0
    exitflag = 0;
    
elseif exitflag && ((sum(lambs < -1e-10)) == 1)    %converged and single positive eigenvalue means sad
    exitflag = 1;
    
end

