clear all;close all;clc;

%nxn array
n = 256;

%make convolution kernel
[t1 t2] = meshgrid(1:n,1:n);
k = fft2(exp(-t1/50).*exp(-t2/75));

%circular convolution fft-based function
convk = @(u) ifftn(fftn(k).*fftn(u));

%the transpose of circular convolution, time-reversed complex-conjugated
%shifted-by-one kernel.
kt = rot90(k,2);
kt = conj(kt);
kt = circshift(kt,[1 1]);
convkt = @(u) ifftn(fftn(kt).*fftn(u));

%test that they are the same thing
x = rand(size(k)) + 1j*randn(size(k));
y = randn(size(k)) + 1j*rand(size(k));

%check the output error
fprintf('The dot product test : %d \n', sum(sum(conj(convk(x)).*y)) - sum(sum(conj(x).*convkt(y))) );