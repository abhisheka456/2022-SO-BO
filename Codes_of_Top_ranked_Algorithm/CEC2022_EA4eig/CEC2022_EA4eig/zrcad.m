% mirroring, Perturbation y into <a,b>
function result = zrcad(y,a,b)
zrc = find(y<a|y>b);
for i = zrc
	while (y(i)<a(i) || y(i)>b(i))
		if y(i) > b(i)
		    y(i) = 2*b(i)-y(i);
		elseif y(i) < a(i)
		    y(i) = 2*a(i)-y(i);
		end
	end
end
result=y;