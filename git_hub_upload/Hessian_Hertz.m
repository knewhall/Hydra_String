function Hess = Hessian_Hertz(x,y,sigma)
% this should be derivatives of the energy function
% ignore the added - for F = -grad V

Np = length(x);
eyesigma = eye(Np,Np).*sigma;
r = diag(sigma)/2;

Dx = x*ones(1,Np)-ones(Np,1)*x';
Dy = y*ones(1,Np)-ones(Np,1)*y';
Dxy = sqrt(Dx.^2 + Dy.^2)+eyesigma;
tmp2 = (sigma-Dxy).^(5/4);
tmp3 = (sigma-Dxy).^(1/4);
Cij = Dxy<sigma;    % puts zero on the diagonal

hxii = 5/4*tmp3.*Dx.^2./Dxy.^2 + tmp2.*( Dx.^2./Dxy.^3 - 1./Dxy );
hxii = sum(Cij.*hxii,2) + (x<r) + ((1-x)<r);

hxij = -5/4*tmp3.*Dx.^2./Dxy.^2 - tmp2.*( Dx.^2./Dxy.^3 - 1./Dxy );
hxij = Cij.*hxij;

hyii = 5/4*tmp3.*Dy.^2./Dxy.^2 + tmp2.*( Dy.^2./Dxy.^3 - 1./Dxy );
hyii = sum(Cij.*hyii,2) + (y<r) + ((1-y)<r);

hyij = -5/4*tmp3.*Dy.^2./Dxy.^2 - tmp2.*( Dy.^2./Dxy.^3 - 1./Dxy );
hyij = Cij.*hyij;

hxyii = 5/4*tmp3.*Dx.*Dy./Dxy.^2 + tmp2.*Dx.*Dy./Dxy.^3 ;
hxyii = sum(Cij.*hxyii,2);

hxyij = -5/4*tmp3.*Dx.*Dy./Dxy.^2 - tmp2.*Dx.*Dy./Dxy.^3 ;
hxyij = Cij.*hxyij;

Hess = [ diag(hxii)+hxij diag(hxyii)+hxyij;...
    diag(hxyii)+hxyij diag(hyii)+hyij ];

end