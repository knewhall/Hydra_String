function [x,y,e1,dd,exitflag] = descend(x,y,sigma,n_images)
%Relaxes a single string when one sphere doesn't move

%   INPUT:
%     x and y are n x 2 arrays containing position data of the particles at
%           a saddle point and a point just before the saddle.
%     sigma contains the radii information in an n x n array, sigma[i,j] = R_i + R_j
%     
%   OUTPUT:
%     x and y on the output are strings connecting a saddle to a minimum
%     e1 is the energy of the system along the string
%     dd is the euclidean distance between string images
%     exitflag is 1 when the string converged to a minimum, 2 if the string
%       ran away, and 0 otherwise

    nstepmax = 1e5; %Maximum number of steps for gradient descent
    %nn = 1e1;   %Frequency of plotting
    dt = 1e-3;  %Time-step 
    diff_tol1 = 1e-8;   %Parameter used as stopping criterion
    exitflag = 0;   %Changes to 1 if convergence reached    
    Np = length(x(:,1));   %Number of particles

    %--------------------------------------
    %Interpolate the images along the string
    g1 = linspace(0,1,n_images);

    dx = diff(x,1,2);
    dy = diff(y,1,2);

    dd = sum(dx.^2+dy.^2);
    dd = sqrt([0 dd]);

    ll = cumsum(dd);
    ll = ll/ll(end);
    x = interp1q(ll',x',g1')';
    y = interp1q(ll',y',g1')';
    %-------------------------------------

    for nstep = 1:nstepmax

        x0 = x; %Store previous string
        y0 = y;

        if isnan(x(1))
            exitflag = 2;   %Sometimes string run away. If this happens exitflag = 2
            break
        end

        %Gradient descent for each image except first but, energy for all
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

        x(2:Np,2:n_images) = x(2:Np,2:n_images) - dt*dVx(2:Np,2:n_images);    %Gradient descend images
        y(2:Np,2:n_images) = y(2:Np,2:n_images) - dt*dVy(2:Np,2:n_images);    %But not first

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
    %----------------------------------    

    % This creates plots of the energy as the string evolves. This can be
    % useful when debugging this function.
    %
    %     if mod(nstep,nn)==0
    %         figure(11);
    %         
    %         dx = x - x(:,1)*ones(1,n_images);
    %         dy = y - y(:,1)*ones(1,n_images);
    %         
    %         dd = sum(dx.^2+dy.^2);
    %         ll = cumsum(sqrt([0 dd]));
    %         
    %         plot(dd,e1,'ro')
    %         plot(ll,e1,'ro')
    %         title(sprintf('step: %g',nstep))
    %         drawnow
    %     end

        %Exit condition
        errX = max(abs(x-x0));  %Hoow much have the images moves in this timestep
        errY = max(abs(y-y0));    
        errdV = max(abs([dVx(:,n_images);dVy(:,n_images)])); %How much has the energy changes

        if max([errX errY errdV]) < diff_tol1
            exitflag = 1;   %If converged, exitflag = 1 and break
            break
        end

    end

    % Hessian Matrix
    hxx = (Dxy<sigma).*( (Dxy./sigma-1)./Dxy + Dx.^2./Dxy.^3 )./sigma;
    hyy = (Dxy<sigma).*( (Dxy./sigma-1)./Dxy + Dy.^2./Dxy.^3 )./sigma;
    hxy = (Dxy<sigma).*Dx.*Dy./Dxy.^3./sigma;

    H0 = [diag(hxx*ones(Np,1))-hxx diag(hxy*ones(Np,1))-hxy;...
        diag(hxy*ones(Np,1))-hxy diag(hyy*ones(Np,1))-hyy];

    if (any(isnan(H0(:))) || any(isinf(H0(:))))
        lambs = 1;  %If the hessian is weird, lambs = 1
    else
        lambs = eig(H0);    %If not weird, calculate eigenvalues
    end

    if exitflag && ((sum(lambs > 1e-10)) == length(H0(:,1)))    %converged and all eigenvalues are positive i.e. min
        exitflag = 1;
    end
        
end