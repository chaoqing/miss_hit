clear
clc
rng default

t = 1:10;
x = randn(size(t))';
ts = linspace(-5,15,2^9);
[Ts,T] = ndgrid(ts,t);
y = sinc(Ts - T)*x;

f = my_fft(y);

disp([y, f]);
%plot(y, 'o')
%plot(f, 'o')