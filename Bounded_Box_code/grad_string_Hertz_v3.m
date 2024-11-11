% gradient descend the string

function [x,y,e1,exitflag] = grad_string_Hertz_v3(x,y,Np,sigma,n1)
exitflag = 0;

% number of images along the string (try from  n1 = 3 up to n1 = 1e4)
% n1 = 30;

% number of steps of steepest descent
% nstepmax = 1e5;
nstepmax = 1e6;
% nsave = 1e2;
% nn = nstepmax/nsave;
nn = 1e3;

% time-step (limited by the ODE step but independent of n1)
dt = 1e-1;

% parameter used as stopping criterion (not always used)
% diff_tol1 = 1e-8;
% diff_tol1 = 1e-6;
diff_tol1 = 1e-7;

% for wall forces
r = diag(sigma)/2*ones(1,n1);  % radius of each particle, for each string

% precompute
eyesigma = eye(Np,Np).*sigma;
e1 = zeros(1,n1);
dVx = zeros(Np,n1);
dVy = zeros(Np,n1);
kap = 4/9;

% initialize string
g1 = linspace(0,1,n1);

% redistribute along string
dx = x(:,2:end) - x(:,1:end-1);
dy = y(:,2:end) - y(:,1:end-1);

dd = sum(dx.^2+dy.^2);
dd = sqrt([0 dd]);

ll = cumsum(dd);
ll = ll/ll(end);
x = interp1q(ll',x',g1')';
y = interp1q(ll',y',g1')';

% gradient descent for each image except last, but energy for all
for j = 1:n1
    % pairwise particle force
    Dx = x(:,j)*ones(1,Np)-ones(Np,1)*x(:,j)';
    Dy = y(:,j)*ones(1,Np)-ones(Np,1)*y(:,j)';
    Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
    tmp = sigma-Dxy; tmp2 = (sigma-Dxy).^(5/4);
    Cij = Dxy<sigma;
    h1 = Cij.*tmp2./Dxy;
    e1(j) = kap*sum(sum(Cij.*tmp2.*tmp));
    
    dVx(:,j) = -(h1.*Dx)*ones(Np,1);
    dVy(:,j) = -(h1.*Dy)*ones(Np,1);
end

% --- wall force
Dx = x; % dist from left wall
h1 = (Dx<=r).*(r - Dx);
dWx0 = -(h1);  % force in x-direction
e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );

Dy = y; % dist from bottom wall
h1 = (Dy<=r).*(r - Dy);
dWy0 = -(h1);  % force in y-direction
e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );

Dx = 1-x; % dist from right wall
h1 = (Dx<=r).*(r - Dx);
dWx1 = (h1);  % force in neg x-direction
e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );

Dy = 1-y; % dist from top wall
h1 = (Dy<=r).*(r - Dy);
dWy1 = (h1);  % force in neg y-direction
e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );

dVx = dVx + dWx0 + dWx1;
dVy = dVy + dWy0 + dWy1;

for nstep = 1:nstepmax
    
    x0 = x;
    y0 = y;
    
    x = x - dt*dVx;
    y = y - dt*dVy;
    
    % redistribute along string
    dx = x(:,2:n1) - x(:,1:n1-1);
    dy = y(:,2:n1) - y(:,1:n1-1);
    
    dd = sum(dx.^2+dy.^2);
    dd = sqrt([0 dd]);
    
    ll = cumsum(dd);
    ll = ll/ll(n1);
    x = interp1q(ll',x',g1')';
    y = interp1q(ll',y',g1')';
    
    for j = 1:n1
        % pairwise particle force
        Dx = x(:,j)*ones(1,Np)-ones(Np,1)*x(:,j)';
        Dy = y(:,j)*ones(1,Np)-ones(Np,1)*y(:,j)';
        Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
        tmp = sigma-Dxy; tmp2 = (sigma-Dxy).^(5/4);
        Cij = Dxy<sigma;
        h1 = Cij.*tmp2./Dxy;
        e1(j) = kap*sum(sum(Cij.*tmp2.*tmp));
        
        dVx(:,j) = -(h1.*Dx)*ones(Np,1);
        dVy(:,j) = -(h1.*Dy)*ones(Np,1);
    end
    
    % --- wall force
    Dx = x; % dist from left wall
    h1 = (Dx<=r).*(r - Dx);
    dWx0 = -(h1);  % force in x-direction
    e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );
    
    Dy = y; % dist from bottom wall
    h1 = (Dy<=r).*(r - Dy);
    dWy0 = -(h1);  % force in y-direction
    e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );
    
    Dx = 1-x; % dist from right wall
    h1 = (Dx<=r).*(r - Dx);
    dWx1 = (h1);  % force in neg x-direction
    e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );
    
    Dy = 1-y; % dist from top wall
    h1 = (Dy<=r).*(r - Dy);
    dWy1 = (h1);  % force in neg y-direction
    e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );
    
    dVx = dVx + dWx0 + dWx1;
    dVy = dVy + dWy0 + dWy1;
    
    % check stopping criteria
    errX = max(abs(x-x0));
    errY = max(abs(y-y0));
    %
    %      if max( [errX errY] ) < diff_tol1
    %          exitflag = 1;
    %          break
    %      end
    tmpx = dVx(:,n1) ;
    tmpy = dVy(:,n1) ;
    errdV = max(abs([tmpx; tmpy]));
    if max([errX errY errdV]) < diff_tol1
        exitflag = 1;
        break
    end
    
end
% disp(nstep)
end

