f = [2 5 3; 1 4 1];
h = [1 -1; 1 1]; % flip of h = [1 1; -1 1]
hflip = [1 1; -1 1];
g = conv2(a,b);
g_same = conv2(a,b,'same');

g_fft = ifft2(fft2(f,3,4).*fft2(h,3,4));

% gen doubly block Toeplitz matrices
H1 = [1 0 0; 
     -1 1 0; 
     0 -1 1;
     0 0 -1;];
H2 = [1 0 0;
      1 1 0;
      0 1 1;
      0 0 1;];
H3 = [0 0 0;
      0 0 0;
      0 0 0;
      0 0 0;];
H = [H1 H3;
     H2 H1;
     H3 H2;];
ft = f';
g2 = H*ft(:);
g2 = reshape(g2,[4 3]).';  
% this is wired,because default matlab reshape use column major...

% compare H'Hf and its fft version
HtH = H'*H;
HtHf = HtH*ft(:);

Fh = fft2(h,2,3);
eigHtH = abs(Fh).^2;
Ff = fft2(f,2,3);
HtHf2 = ifft2(eigHtH.*Ff);

eigHtH2 = reshape(eig(HtH),[3 2]).';  
HtHf3 = ifft2(eigHtH2.*Ff);