f = [2 5 3; 1 4 1;];
h = [1 -1; 1 1]; 
% g = conv2(f,h,'same');
g_fft = ifft2(fft2(f).*fft2(h,2,3));

g_fft2 = ifft2(fft2(circshift(f,[-1,-1])).*fft2(rot90(h,2),2,3));

H1 = [1 0 -1;
      -1 1 0;
      0 -1 1;];
H2 = [1 0 1;
      1 1 0;
      0 1 1;];
H = [H1 H2;
     H2 H1;];

ft = f';
g2 = H*ft(:);
g2 = reshape(g2,[3 2]).';

g3 = H'*ft(:);
g3 = reshape(g3,[3 2]).';

left = H' * H * ft(:);
eigHtH = abs(fft2(h,2,3)).^2;
right = ifft2(eigHtH.*fft2(f));