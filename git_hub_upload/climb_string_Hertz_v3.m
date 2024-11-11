% function [saddle,exitflag,lambs] = climb_string(xmin,ymin,xsad,ysad,Np,sigma)
% climb up the wall forces too

function [x,y,e1,exitflag] = climb_string_Hertz_v3(xmin,ymin,xsad,ysad,Np,sigma,n1)
exitflag = 0;

% number of images along the string (try from  n1 = 3 up to n1 = 1e4)
% n1 = 30;

% number of steps of steepest descent
nstepmax = 5e5;
% nsave = 1e2;
% nn = nstepmax/nsave;
nn = 1e3;

% time-step (limited by the ODE step but independent of n1)
dt = 1e-1;

% amplitude of climbing point
% amp=1.5;
amp = 2;

% parameter used as stopping criterion (not always used)
% diff_tol1 = 1e-8;
diff_tol1 = 1e-6;

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
x = xmin*ones(1,n1)+(xsad-xmin)*g1;
y = ymin*ones(1,n1)+(ysad-ymin)*g1;

% gradient descent for each image except last, but energy for all
for j = 1:n1
    % pairwise particle force
    Dx = x(:,j)*ones(1,Np)-ones(Np,1)*x(:,j)';
    Dy = y(:,j)*ones(1,Np)-ones(Np,1)*y(:,j)';
    Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
    tmp = sigma-Dxy; tmp2 = (sigma-Dxy).^(5/4);
    Cij = Dxy<=sigma;
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
    
    % unit tangent vector to string at the last point
    txy = sqrt(sum((x(:,n1)-x(:,n1-1)).^2+(y(:,n1)-y(:,n1-1)).^2));  % dist between last 2 string elements
    tx = (x(:,n1)-x(:,n1-1))/txy;
    ty = (y(:,n1)-y(:,n1-1))/txy;
    % only climb with particle forces, don't climb the wall (??)
    dVxy = dVx(:,n1)'*tx+dVy(:,n1)'*ty;
    
    x0 = x;
    y0 = y;
    
    x(:,1:n1-1) = x(:,1:n1-1) - dt*dVx(:,1:n1-1);
    y(:,1:n1-1) = y(:,1:n1-1) - dt*dVy(:,1:n1-1);
    
    % last point, moves with inverted direction
    x(:,n1) = x(:,n1) - dt*dVx(:,n1) + amp*dt*dVxy*tx;
    y(:,n1) = y(:,n1) - dt*dVy(:,n1) + amp*dt*dVxy*ty;
    
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
    
    % check for non-monotone
    de1 = e1(2:n1)-e1(1:n1-1);
    nn1 = find(de1<-1e-9); nn1=nn1(find(nn1>2,1));   % find all, then take the first greater than 2
    if (~isempty(nn1) && nn1>2)
        % fprintf('string was truncated at iteration %d to length %d\n',nstep,nn1);
        
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
        dWx0 = -(h1.*Dx);  % force in x-direction
        e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );
        
        Dy = y; % dist from bottom wall
        h1 = (Dy<=r).*(r - Dy);
        dWy0 = -(h1.*Dy);  % force in y-direction
        e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );
        
        Dx = 1-x; % dist from right wall
        h1 = (Dx<=r).*(r - Dx);
        dWx1 = (h1.*Dx);  % force in neg x-direction
        e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );
        
        Dy = 1-y; % dist from top wall
        h1 = (Dy<=r).*(r - Dy);
        dWy1 = (h1.*Dy);  % force in neg y-direction
        e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );
        
        dVx = dVx + dWx0 + dWx1;
        dVy = dVy + dWy0 + dWy1;
    end
    
    % check stopping criteria
    errX = max(abs(x-x0));
    errY = max(abs(y-y0));

    tmpx = dVx(:,n1) ;
    tmpy = dVy(:,n1) ;
    errdV = max(abs([tmpx; tmpy]));
    if max([errX errY errdV]) < diff_tol1
        exitflag = 1;
        break
    end
    
end % end time-stepping loop

if exitflag==1
    % check for one negative eigenvalue
    Hm = Hessian_Hertz(x(:,n1),y(:,n1),sigma);
    lamb = eig(Hm);
    if sum( lamb < -1e-12 ) ~= 1
        exitflag=2;
    end
end

% disp(nstep)
end

