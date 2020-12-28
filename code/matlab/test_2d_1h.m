f = [2 5 3; 1 4 1;];
h = [1 -1]; 
ht = h';
g_fft = ifft2(fft2(f).*fft2(h,2,3));
ht = circshift(rot90(h,2),[1 1]);
g_fft2 = ifft2(fft2(circshift(f,[0, -1])).*fft2(rot90(h,2),2,3));

H1 = [1 0 -1;
      -1 1 0;
      0 -1 1;];
H2 = [0 0 0;
      0 0 0;
      0 0 0;];
H = [H1 H2;
     H2 H1;];

ft = f';
g2 = H*ft(:);
g2 = reshape(g2,[3 2]).';

g3 = H'*ft(:);
g3 = reshape(g3,[3 2]).';