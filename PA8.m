% Michael Nwokobia
% PA 8 
% 101044216

Is = 0.01e-12; 
Ib = 0.1e-12; 
Vb = 1.3; 
Gp = 0.1; 

V = linspace(-1.95, 0.7, 200); 
I = (Is .* (exp(V .* 1.2/0.025) - 1)) + (Gp .* V) - (Ib .* (exp((-1.2/0.025) .* (V + Vb))));

Inoise = ((randn(1, 200))) .* 0.2 .* I;

% POLYNOMIAL FITTING:
fit4 = polyfit(V, I, 4); 
fit4noise = polyfit(V, Inoise, 4); 
fit8 = polyfit(V, I, 8); 
fit8noise = polyfit(V, Inoise, 8); 

figure(1)
plot(V, I)
title('V vs. I')
xlabel('V (V)')
ylabel('I (A)')
hold on
plot(V, polyval(fit4, V))
hold off

figure(2)
plot(V, Inoise)
title('V vs. I (with experimental noise)')
xlabel('V (V)')
ylabel('I (A)')
hold on
plot(V, polyval(fit4noise, V))
hold off

figure(3)
semilogy(V, I)
title('V vs. log(I)')
xlabel('V (V)')
ylabel('log(I) (A)')
hold on
plot(V, polyval(fit8, V))
hold off

figure(4)
semilogy(V, Inoise)
title('V vs. log(I) with Experimental Noise')
xlabel('V (V)')
ylabel('log(I) (A)')
hold on
plot(V, polyval(fit8noise, V))
hold off

% NONLINEAR CURVE FITTING:
fo = fittype('A.*(exp(1.2.*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V',I',fo);
If = ff(V);
figure(5);
plot(V, If);
title('V vs. I (fitted parameters: Is, Ib)')
xlabel('V (V)')
ylabel('I (A)')

fo = fittype('A.*(exp(1.2.*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V',I',fo);
If = ff(V);
figure(6);
plot(V, If);
title('V vs. I (fitted parameters: Is, Ib, Gp)')
xlabel('V (V)')
ylabel('I (A)')

fo = fittype('A.*(exp(1.2.*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff = fit(V',I',fo);
If = ff(V);
figure(7);
plot(V, If);
title('V vs. I (fitted parameters: Is, Ib, Gp, Vb)')
xlabel('V (V)')
ylabel('I (A)')

% NEURAL NET FITTING:
inputs = V.';
targets = I.'; 
hiddenLayerSize = 10; 
net = fitnet(hiddenLayerSize); 
net.divideParam.trainRatio = 70/100; 
net.divideParam.valRatio = 15/100; 
net.divideParam.testRatio = 15/100; 
[net, tr] = train(net, inputs, targets); 
outputs = net(inputs); 
errors = gsubtract(outputs, targets); 
performance = perform(net, targets, outputs) 
view(net)
Inn = outputs