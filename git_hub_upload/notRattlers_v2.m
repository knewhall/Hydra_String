function keep = notRattlers_v2(x,y,sigma)
% should recursively-remove

Np = size(x,1);
eyesigma = eye(Np,Np).*sigma;
    
keep = [1:Np]'; % start by keeping all
keep_one = ones(size(x));
rat = [];   % no rattlers
flag = true;

while flag && ~isempty(keep)

    % pairwise particle force
    Dx = x*ones(1,Np)-ones(Np,1)*x';
    Dy = y*ones(1,Np)-ones(Np,1)*y';
    Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
    Cij = Dxy<sigma;    % puts zero on the diagonal
    
    Cij(rat,:) = 0; % remove rattler contacts
    Cij(:,rat) = 0;
    
    r = diag(sigma/2);
    
    % non-rattlers >=3 contacts including walls
    numC = sum(Cij,2) + (x<r).*keep_one + (y<r).*keep_one ...
        + ( (1-x)<r ).*keep_one + ( (1-y)<r ).*keep_one;
    
    tmp = find(numC >= 3);
    if length(tmp)==length(keep)
        % nothing changed, exit
        flag = false;
    else
        keep = tmp;
        rat = find(numC < 3);
        keep_one(rat) = 0;
    end
end