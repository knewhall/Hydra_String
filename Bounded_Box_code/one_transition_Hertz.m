function [xb,yb,e1]=one_transition_Hertz(x,y,sigma)

Np = size(x,1); % number of particles
n1 = size(x,2); % number of images

kap = 4/9;
eyesigma = eye(Np,Np).*sigma;
r = diag(sigma)/2;
e1 = zeros(1,n1);

for j = 1:n1
    % pairwise particle force
    Dx = x(:,j)*ones(1,Np)-ones(Np,1)*x(:,j)';
    Dy = y(:,j)*ones(1,Np)-ones(Np,1)*y(:,j)';
    Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
    tmp = sigma-Dxy; tmp2 = (sigma-Dxy).^(5/4);
    Cij = Dxy<sigma;
    e1(j) = kap*sum(sum(Cij.*tmp2.*tmp));
end

% --- wall energy
Dx = x; % dist from left wall
e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );

Dy = y; % dist from bottom wall
e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );

Dx = 1-x; % dist from right wall
e1 = e1 + 0.5*sum( (Dx<=r).*(r - Dx).^2 );

Dy = 1-y; % dist from top wall
e1 = e1 + 0.5*sum( (Dy<=r).*(r - Dy).^2 );

% --- determine if one transition, otherwise truncate
[~,indM] = max(e1);

de1 = e1(indM+1:n1)-e1(indM:n1-1);
nn1 = find(de1>1e-9,1);   % find up-hill after max (saddle pt)
if (~isempty(nn1) && nn1>2)
    % if non-monotone down, truncate
    x = x(:,1:indM+nn1-1);
    y = y(:,1:indM+nn1-1);
    
    dx = x(:,2:nn1) - x(:,1:nn1-1);
    dy = y(:,2:nn1) - y(:,1:nn1-1);
    
    dd = sum(dx.^2+dy.^2);
    dd = sqrt([0 dd]);
    
    ll = cumsum(dd);
    ll = ll/ll(nn1);
    g1 = linspace(0,1,n1);
    x = interp1q(ll',x',g1')';
    y = interp1q(ll',y',g1')';
    
    % evolve one more time
    [xb,yb,e1,~] = grad_string_Hertz_v3(x,y,Np,sigma,n1);
else
    xb = x;
    yb = y;
end