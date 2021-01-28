%-------------------------------------------------------------------------%
%   This is the Hydra String (HSM) as proposed in the paper by Moakler and
%   Newhall and available at arXiv:2012.09974
%
%   The HSM begins at a minimum of a function and extends climbing strings (as
%   described by Vanden-Eijnden) to find saddle points on the ridge of the
%   originating minimum. From these saddles, new minima are found and the
%   process continues by extending new climbing strings. Like a Hydra, the
%   strings branch out enumerating saddle points and minima and the minimum
%   energy paths that connect them. The HSM also creates a connected bi
%   partite graph that maps out the potential energy surface.
%
%   We have devised this method to explore the potential energy landscape of
%   granular materials to better understand their dynamics when they
%   experience shears and perturbations. However, this method can be applied
%   to any non-convex function and we welcome its application to other fields
%
%   The provided data describe a system of 24 bi-disperse particles in a
%   periodic boundary of unit length. The first particle is kept fixed to break
%   the symmetry caused by the periodic boundary 
%   The user defined parameters that follow reflect those found in the paper 
%   where we describe how to pick these parameters for different systems.
%
%If you use this method please cite our paper and if you have questions or
%comments about the code, we can be reached at cmoakler@unc.edu and
%knewhall@unc.edu
%
%-------------------------------------------------------------------------%

tol_sad = 1e-2;     %Tolerance to distinguish degenerate saddles and minima
roundth = 5;        %Round to roundth decimal
n_images = 10;      %Number of images along the string
nstrings = 4;      %Number of strings to extend from each minimum
ext_dist=0.16;      %Distance to initially extend string from a minimum

% parpool([1 48])     %Initialize Matlab parpool. 
rng(1)              %Set RNG seed for debugging purposes

%
%Load in parameters for the granular system. Includes particles positions
%and particle radii. The sample data is for a periodic system in a unit box
%
load('xy_24ParticleData.mat','xy','R1','R2','erg_min_unique');
x_min_new = round(xy(:,1),roundth);         %First minimum to be explored
y_min_new = round(xy(:,2),roundth);
erg_min_new = erg_min_unique(1);            %Energy of the first minimum

%This is a bidisperse system so there are equal numbers of small and large
%particles
Np=size(x_min_new,1);       %Number of particles in the system
Np1=12;                     %Number of larger particles
Np2=12;                     %Number of smaller particles
sigma = [2*R1*ones(Np1,Np1) (R1+R2)*ones(Np1,Np2); ...  %Generate the matrix
            (R1+R2)*ones(Np2,Np1) 2*R2*ones(Np2,Np2)];  %of particle radii
        
% --- variables for storing everything
x_sad_unique = [];      %List of unique saddles, numbered for edge list
y_sad_unique = [];
erg_sad_unique = [];    %Unique saddle energy list
x_min_unique = x_min_new;  %List of unique min, numbered for edge list
y_min_unique = y_min_new;
erg_min_unique = erg_min_new;        %Energy of unique min, for plotting
edge_list = [];         %Save network [min_images_ind sad_ind min2_ind min_images_erg sad_erg min2_erg]
x_string_min_sad = [];     %Strings between minima and saddles
y_string_min_sad = [];
x_string_sad_min = [];     %Strings between saddles and minima
y_string_sad_min = [];
cnt = 1;                %Count length of edge_list

n_min = 0;      %Number of minimum that have been explored
n_min_max = 10;  %Total number of minimum to explore

while( ~isempty(x_min_new) && (n_min<n_min_max) )

    %Minimum to be explored take from list of new minima
    x_min = x_min_new(:,1);
    y_min = y_min_new(:,1);

    %Find the index of this minimum on the minimum list
    Nd = size(x_min_unique,2);  %Number of unique minima already found
    dx = max( abs(x_min*ones(1,Nd) - x_min_unique) );      %Compare to list of minima
    dy = max( abs(y_min*ones(1,Nd) - y_min_unique) );

    ind_minA = find( (dx<tol_sad) & (dy<tol_sad) );     %Find index it corresponds to

    %Remove this minimum from list since we will now explore it
    if size(x_min_new,2)>1
        x_min_new = x_min_new(:,2:end);
        y_min_new = y_min_new(:,2:end);
    else
        x_min_new = [];
        y_min_new = [];
    end

    %Initialize the matrices for the parallel loop
    x_strings = zeros(nstrings,Np,n_images);
    y_strings = zeros(nstrings,Np,n_images);
    erg = zeros(1,nstrings);        %Energy of the saddle point
    erg_min = zeros(1,nstrings);    % Energy of the minimum explored
    e_flags = zeros(1,nstrings);    %Output flag 1=converged and 1 neg eigval
    
tic   %Time the parallel loop
parfor k=1:nstrings

    %------------------------------------%
    %Randomly perturb one particle
    x_perturb = x_min;    %Copy particle locations
    y_perturb = y_min;
    
    ind = 1 + ceil((Np-1)*rand);        %Choose random particle but, not the first one
    theta = 2*pi*rand;                  %Angle to perturb in
    x_perturb(ind) = x_min(ind) + ext_dist*cos(theta);     %Perturb it in x and y
    y_perturb(ind) = y_min(ind) + ext_dist*sin(theta);    
    % ---------------------------------- %
    
    %Climb the string
    [x,y,e1,exitflag,lambs] = climb(x_min,y_min,x_perturb,y_perturb,sigma,n_images);
    
    x_strings(k,:,:) = mod(x,1);   %Add string to list of strings
    y_strings(k,:,:) = mod(y,1);
    erg(k) = e1(n_images);        %Add Energy to 
    erg_min(k) = e1(1);    %Energy of the minimum explored
    e_flags(k) = exitflag;  %Add exit flag (should be 1 if properly converged)
    
end

erg_min = erg_min(1);   %Only one minimum was explored so they all have the same energy

%Select out those strings that converged to a saddle with 1 neg eigval
%Create unique list of new saddles

saddle_ind_parfor = find(e_flags==1);   %Find index of saddles that properly converged

x_strings = x_strings(saddle_ind_parfor,:,:);   %Select those strings out
y_strings = y_strings(saddle_ind_parfor,:,:);

%Find unique saddles found in previous loop
[~,iaxy,icx] = uniquetol([x_strings(:,:,end),y_strings(:,:,end)],tol_sad,'ByRows',true);   

x_strings = x_strings(iaxy,:,:);     %Remove repeated saddles
y_strings = y_strings(iaxy,:,:);
erg = erg(iaxy);

x_strings(x_strings(:,1,1) == 0,:,:) = [];  %Remove strings that are all 0s
y_strings(y_strings(:,1,1) == 0,:,:) = [];
erg(erg == 0) = [];

    %How many new saddles did we find?
    if isempty(x_sad_unique)
        num_sads = 0;
    else
        num_sads = length(x_sad_unique(1,1,:));
    end

fprintf('\nSaddles: ');     %Print it
toc

tic   %Time climbing strings

%This loop can also be done in parallel and these are the loop variables if
%you wish to do so. We have found this to be less efficient in our
%application

x_string_list_temp = zeros(length(iaxy),Np,n_images);     % temp variables for the loop
y_string_list_temp = zeros(length(iaxy),Np,n_images);
x_min_unique_temp = zeros(length(iaxy),Np);
y_min_unique_temp = zeros(length(iaxy),Np);
x_sad_unique_temp = zeros(length(iaxy),Np);
y_sad_unique_temp = zeros(length(iaxy),Np);
erg_min_unique_temp = zeros(length(iaxy),1);
erg_sad_unique_temp = zeros(length(iaxy),1);
edge_list_temp = zeros(length(iaxy),6);

%-----------------------
%Descending strings loop
%-----------------------
for k = 1:length(iaxy)

    %Descend from each new saddle towards a new minimum    
    x = x_strings(k,:,:);
    y = y_strings(k,:,:);
    
    x = squeeze(x); %Remove extra matrix dimension
    y = squeeze(y);

    x_saddle = mod(round(x(:,n_images),roundth),1);   %Round and mod 
    y_saddle = mod(round(y(:,n_images),roundth),1);
    
    %Compare saddle to the entire list of saddles and remove repeats
    Nd = size(x_sad_unique,2); %Number of saddles already found
    dx=0;dy=0;  %clear dx;dy
    
    if Nd>=1
        %Take the max distance for each point already in saddle
        dx = max( abs(x_saddle*ones(1,Nd) - x_sad_unique) );
        dy = max( abs(y_saddle*ones(1,Nd) - y_sad_unique) );
    end
    
    %If the min of ALL distances is above tolerance, new unique
    if ((all((dx>tol_sad) & (dy>tol_sad))) && (x_saddle(1) ~= 0) ) || Nd==0
        x_sad_unique_temp(k,:) = x_saddle;
        y_sad_unique_temp(k,:) = y_saddle;
        erg_sad_unique_temp(k) = erg(k);
        
        %Add string to list
        x_string_min_sad = cat(3, x_string_min_sad, permute(x_strings(k,:,:), [2 3 1]));
        y_string_min_sad = cat(3, y_string_min_sad, permute(y_strings(k,:,:), [2 3 1]));

        %Find unit tangent vector to string at the last point to direct
        %descending string
        txy = sqrt(sum((x(:,n_images)-x(:,n_images-1)).^2+(y(:,n_images)-y(:,n_images-1)).^2));  % dist between last 2 string elements
        tx = (x(:,n_images)-x(:,n_images-1))/txy;
        ty = (y(:,n_images)-y(:,n_images-1))/txy;
        
        %Take saddle point
        x = x(:,n_images);
        y = y(:,n_images);

        %Perturb saddle point away from minimum    
        x(:,n_images+1) = x + 0.01*tx;
        y(:,n_images+1) = y + 0.01*ty;
        
        %Descend the string to new minimum
        [x2,y2,e1,dd,exitflag] = descend(x,y,sigma,n_images);
        
        x2 = mod(round(x2,roundth),1);  %Mod and round
        y2 = mod(round(y2,roundth),1);
        
        x_min_unique_temp(k,:) = x2(:,end);     %Add minimum to temp list
        y_min_unique_temp(k,:) = y2(:,end);
        erg_min_unique_temp(k) = e1(end);
        
        x_string_list_temp(k,:,:) = x2;     %Add string to temp list
        y_string_list_temp(k,:,:) = y2;
              
        %Add to temp network
        edge_list_temp(k,:) = [ind_minA 0 0 erg_min erg(k) e1(end)];
    end
    
end

    %Postprocess strings
    %Determine the index of the saddle points for the dge list

    if size(x_sad_unique_temp,1) > 0
        %Remove strings that have broken
        x_sad_unique_temp(x_sad_unique_temp(:,1,1) == 0,:,:) = [];
        y_sad_unique_temp(y_sad_unique_temp(:,1,1) == 0,:,:) = [];
        edge_list_temp(edge_list_temp(:,4) == 0,:) = [];
        erg_sad_unique_temp(erg_sad_unique_temp == 0) = [];

        %Append remaining saddles
        x_sad_unique = [x_sad_unique , x_sad_unique_temp(:,:,end)'];
        y_sad_unique = [y_sad_unique , y_sad_unique_temp(:,:,end)'];

        %Index of new saddles appended to edge list
        edge_list_temp(:,2) = size(edge_list,1) + (1:size(edge_list_temp,1));
    end
        
    %Remove strings that have broken
    if size(x_min_unique_temp,1) > 0
        x_min_unique_temp(x_min_unique_temp(:,1,1) == 0,:,:) = [];
        y_min_unique_temp(y_min_unique_temp(:,1,1) == 0,:,:) = [];
    end
    
    %Determine index of final minimum in remaining strings
    ind_temp = 1;
    
    for k = 1:size(x_min_unique_temp,1)
        
        %Determine if the minimum is unique on the global list
        Nd = size(x_min_unique,2);

        dx = max( abs(x_min_unique_temp(k,:)'*ones(1,Nd) - x_min_unique) );
        dy = max( abs(y_min_unique_temp(k,:)'*ones(1,Nd) - y_min_unique) );

        ind_min = find( (dx<tol_sad) & (dy<tol_sad) );
    
            if isempty(ind_min) %Minimum is unique

                x_min_unique(:,end+1) = x_min_unique_temp(k,:);        %Add min to list
                y_min_unique(:,end+1) = y_min_unique_temp(k,:);

                x_min_new = [x_min_new , x_min_unique_temp(k,:)'];       %Add min to new min list
                y_min_new = [y_min_new , y_min_unique_temp(k,:)'];

                x_string_sad_min(:,:,end+1) = x_string_list_temp(k,:,:);   %Add string to sad min list
                y_string_sad_min(:,:,end+1) = y_string_list_temp(k,:,:);


                edge_list_temp(k,3) = Nd + 1;           %Assign proper index to new min

                edge_list(end+1,:) = edge_list_temp(k,:);	%Add edge to network

            elseif all(size(ind_min) == [1 1])  %Minimum is not unique

                edge_list_temp(k,3) = ind_min;           %Assign proper index to new min

                if ~isequal(unique([edge_list ;edge_list_temp(k,:)],'rows','stable'), edge_list) 

                    x_string_sad_min(:,:,end+1) = x_string_list_temp(k,:,:);   %Add string to list
                    y_string_sad_min(:,:,end+1) = y_string_list_temp(k,:,:);

                    edge_list(end+1,:) = edge_list_temp(k,:);               %Add edge to network

                end

            else
                        
            end
    end
        
fprintf('NewMins: ');
toc   %Time descending strings

n_min = n_min + 1;  %We have explored one minimum

save(['run_' date]) %Save the workspace
disp('saved')
end % end while loop to explore minimum

save(['run_' date])
















