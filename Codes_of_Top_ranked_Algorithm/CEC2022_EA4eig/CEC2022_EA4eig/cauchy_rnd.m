function  y = cauchy_rnd(x0, gamma)
% nah. cis. z Cauchyho rozdeleni a parametry x0, gamma
y = x0 + gamma * tan (pi * (rand(1) - 1 / 2));
