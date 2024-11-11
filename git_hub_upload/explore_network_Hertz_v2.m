% Network Building Script

myCluster = parcluster('local');
myCluster.NumWorkers = 32;
saveProfile(myCluster);
% oldprofile = parallel.defaultClusterProfile(myCluster);

parpool(32,'IdleTimeout', Inf)

load('run_mins.mat','x_min_all','y_min_all','sigma','Np')
k = 8; % this appears jammed and energy 5e-6
xa = x_min_all(:,k);
ya = y_min_all(:,k);

% xa and ya are the minimum to start from
% sigma is the matrix of particle separation distances
% Np is the number of particles
% k is the index for naming the save file

fln = 'full_network_run5.mat';

n_min = 0;      % number of minimum that have been explored
% n_min_max = 150;  % total number of minimum to explore
n_min_max = 75;

n1 = 30;        % number of images along the string
nloops = 128;    % number of times to climb string

% tol_sad = 1e-6; % used everywhere to determine "same"
tol_sad = 1e-4;

% --- variables for storing everything
x_min_unique = xa;  % list of unique min, numbered for edge list
y_min_unique = ya;
x_sad_unique = [];  % list of unique sad, numbered for edge list
y_sad_unique = [];
edge_list = [];         % save network min sad min edgeweight
x_string_list = [];     % save string same order as edge_list
y_string_list = [];
cnt = 1;                % count length of edge_list

x_min_new = xa;         % list of minima to explore
y_min_new = ya;         % start off with the first minimum

while( ~isempty(x_min_new) && (n_min<n_min_max) )
    xa = x_min_new(:,1);
    ya = y_min_new(:,1);
    
    % which point is this?
    Nd = size(x_min_unique,2);
    dx = max( abs(xa*ones(1,Nd) - x_min_unique) );
    dy = max( abs(ya*ones(1,Nd) - y_min_unique) );
    
    ind_minA = find( (dx<tol_sad) & (dy<tol_sad) );
    % remove this minimum from list since will now explore
    if size(x_min_new,2)>1
        x_min_new = x_min_new(:,2:end);
        y_min_new = y_min_new(:,2:end);
    else
        x_min_new = [];
        y_min_new = [];
    end
    
    % --- climb to find a bunch of saddles
    x_strings = zeros(Np,n1,nloops);
    y_strings = zeros(Np,n1,nloops);
    erg = zeros(1,nloops);        % energy of the saddle point
    e_flags = zeros(1,nloops);
    
    tic
    fprintf('climbing: ');
    parfor k=1:nloops
        % for k=1:nloops
        % --- randomly move one particle --- %
        xb = xa;
        yb = ya;
        
        alpha=0.04; % less than 1/2 rad of small, should stay inside box
        ind = 1 + ceil((Np-1)*rand);    % not the first particle
        th = 2*pi*rand;
        xb(ind)=xa(ind)+alpha*cos(th);
        yb(ind)=ya(ind)+alpha*sin(th);
        
        % ---------------------------------- %
        
        [xs,ys,e1,exitflag] = climb_string_Hertz_v3(xa,ya,xb,yb,Np,sigma,n1);
        
        x_strings(:,:,k) = xs;
        y_strings(:,:,k) = ys;
        erg(k) = e1(n1);        % saddle point energy
        e_flags(k) = exitflag;  % 1 if successfully converged
    end
    toc
    
    % --- create unique list of new saddles and minima
    tic
    fprintf('processing: ');
    % exitflag = 0 did not converge
    % exitflag = 2 converged, but not index 1 saddle
    for k=find(e_flags==1)
        % for k=1:nloops
        
        x = x_strings(:,:,k);
        y = y_strings(:,:,k);
        
        x_saddle = x(:,n1);
        y_saddle = y(:,n1);
        
        % only compare rattle-less particles
        keep = notRattlers_v2(x_saddle,y_saddle,sigma);
        
        Nd = size(x_sad_unique,2);
        if Nd>=1
            % take the max distance for each point already in saddle
            % only compare non-rattling particles
            dx = x_saddle(keep)*ones(1,Nd) - x_sad_unique(keep,:);
            dy = y_saddle(keep)*ones(1,Nd) - y_sad_unique(keep,:);
        end
        
        if  all( sum(dx.^2 + dy.^2) > tol_sad ) || Nd==0 || isempty(keep)
            x_sad_unique = [x_sad_unique x_saddle];
            y_sad_unique = [y_sad_unique y_saddle];
            ind_sad = size(x_sad_unique,2);
            
            % --- find new min -- unit tangent vector to string at the last point
            
            txy = sqrt(sum((x(:,n1)-x(:,n1-1)).^2+(y(:,n1)-y(:,n1-1)).^2));
            % dist between last 2 string elements
            tx = (x(:,n1)-x(:,n1-1))/txy;
            ty = (y(:,n1)-y(:,n1-1))/txy;
            
            x(:,n1) = x(:,n1) + 0.01*tx;
            y(:,n1) = y(:,n1) + 0.01*ty;
            
            [x2,y2,e1,exitflag] = grad_string_Hertz_v3(x,y,Np,sigma,2*n1);
            % x2 and y2 different number images from x and y
            
            % check that didn't skip a transition, otherwise truncate
            [x2,y2,e1] = one_transition_Hertz(x2,y2,sigma);
            
            x_string_list(:,:,cnt) = x2;
            y_string_list(:,:,cnt) = y2;
            
            % --- need to compare min to total minimum to list, see if new
            % only compare rattle-less particles
            keep = notRattlers_v2(x2(:,end),y2(:,end),sigma);
            Nd = size(x_min_unique,2);
            dx = x2(keep,end)*ones(1,Nd) - x_min_unique(keep,:);
            dy = y2(keep,end)*ones(1,Nd) - y_min_unique(keep,:);
            ind_minB = find( sum(dx.^2 + dy.^2) < tol_sad, 1 );
            if isempty(ind_minB) || isempty(keep)
                % new unique minimum found
                x_min_unique = [x_min_unique x2(:,end)];
                y_min_unique = [y_min_unique y2(:,end)];
                
                ind_minB = size(x_min_unique,2);
                
                if ~isempty(keep)
                    % only add to unexplored min if truely minima and jammed
                    Hm = Hessian_Hertz(x2(keep,end),y2(keep,end),sigma(keep,keep));
                    lambm = eig(Hm);
                    % un-jammed will have eigenvalue of zero
                    if all( lambm > 1e-12 )
                        x_min_new = [x_min_new x2(:,end)];
                        y_min_new = [y_min_new y2(:,end)];
                    end
                end
            end
            % --- add to network
            edge_list = [edge_list; ind_minA ind_sad ind_minB e1(1) erg(k) e1(end)];
            
            cnt = cnt+1;    % increase counter, added to edge list
        end % end if new saddle found
        
    end % end looping through all found saddles
    toc
    
    save(fln);
    
    n_min = n_min + 1;  % increase counter, have explored one min
end % expore one minimum

