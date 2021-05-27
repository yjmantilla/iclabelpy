
myepsx = 0.6;
myepsy = 0.6;
x = [0 0.5 -0.5 -0.25];
y = [0 0.5 -0.5 -0.25];
v = [1 5 3 0];
sz = numel(x);
x = reshape(x,sz,1);
y = reshape(y,sz,1);
v = reshape(v,sz,1);

xyv = matlab.internal.math.mergesimpts([x, y, v], [myepsy, myepsx, Inf], 'average');
%%

x = [3 2 1 0];
y = [0 1 2 3];
v = [1 5 3 0];
sz = numel(x);
x = reshape(x,sz,1);
y = reshape(y,sz,1);
v = reshape(v,sz,1);
xyv = matlab.internal.math.mergesimpts([x, y, v], [myepsy, myepsx, Inf], 'average');

%%
myepsx = 0.6;
myepsy = 0.6;
x = [0.5 0 -0.5 -0.25];
y = [0.5 0 -0.5 -0.25];
v = [5 1 3 0];
sz = numel(x);
x = reshape(x,sz,1);
y = reshape(y,sz,1);
v = reshape(v,sz,1);
data = [x, y, v];
xyv = matlab.internal.math.mergesimpts(data, [myepsy, myepsx, Inf], 'average');
