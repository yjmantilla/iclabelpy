import numpy as np
from iclabelpy.utils import mergesimpts

#%%
x = [0, 0.5, -0.5,-0.25]
y = [0, 0.5, -0.5,-0.25]
v = [1, 5 ,3,0]
tols = [0.6,0.6,np.inf]
data = np.stack((x, y,v), axis=-1)
us = mergesimpts(data,tols)

result = np.array([[-0.250000000000000,-0.250000000000000,1.33333333333333],
[0.500000000000000,0.500000000000000,5]])

assert np.all(np.isclose(result,us))
#%%
tols = [0.6,0.6,np.inf]
x = [3, 2, 1, 0];
y = [0, 1, 2, 3];
v = [1, 5, 3, 0];
data = np.stack((x, y,v), axis=-1)
result =np.array( [[0,3,0],
[1,2,3],
[2,1,5],
[3,0,1]])
us = mergesimpts(data,tols)
assert np.all(np.isclose(result,us))

#%%

tols = [0.6,0.6,np.inf]
x = [0.5 ,0 ,-0.5 ,-0.25]
y = [0.5 ,0 ,-0.5 ,-0.25]
v = [5   ,1 ,   3 ,    0]
data = np.stack((x,y,v), axis=-1)
result = np.array([[-0.250000000000000,-0.250000000000000,1.33333333333333],
[0.500000000000000,0.500000000000000,5]])
us = mergesimpts(data,tols)
assert np.all(np.isclose(result,us))

## https://stackoverflow.com/questions/1988535/return-unique-element-with-a-tolerance?noredirect=1&lq=1

#%%
"""
>> x = [1; 1.1; 1.05];             % elements need not be sorted
>> builtin('_mergesimpts',x,eps)   % but the output is sorted
ans =
    1.0000
    1.0500
    1.1000
"""
eps = 2.2204e-16
tols = [eps]
x = [1 ,1.1 ,1.05]
data = np.stack((x,), axis=-1)

result = np.array([[1],
[1.05000000000000],
[1.10000000000000]])
us = mergesimpts(data,tols)
assert np.all(np.isclose(result,us))

#%%
"""
>> builtin('_mergesimpts',x,0.1,'first')
ans =
    1.0000  % first of [1, 1.05] since abs(1 - 1.05) < 0.1
    1.1000
"""

tols = [0.1]
x = [1 ,1.1 ,1.05]
data = np.stack((x,), axis=-1)

result = np.array([[1],
[1.10000000000000]])
us = mergesimpts(data,tols,'first')
assert np.all(np.isclose(result,us))


#%%
"""
builtin('_mergesimpts',x,0.1,'average')
ans =
    1.0250  % average of [1, 1.05]
    1.1000
"""

tols = [0.1]
x = [1 ,1.1 ,1.05]
data = np.stack((x,), axis=-1)

result = np.array([[1.02500000000000],
[1.10000000000000]])
us = mergesimpts(data,tols,'average')
assert np.all(np.isclose(result,us))

#%%
"""
>> builtin('_mergesimpts',x,0.2,'average')
ans =
    1.0500  % average of [1, 1.1, 1.05]
"""
tols = [0.2]
x = [1 ,1.1 ,1.05]
data = np.stack((x,), axis=-1)

result = np.array([[1.05000000000000]])
us = mergesimpts(data,tols,'average')
assert np.all(np.isclose(result,us))

#%%
"""
>> x = [1 2; 1.06 2; 1.1 2; 1.1 2.03]
x =
    1.0000    2.0000
    1.0600    2.0000
    1.1000    2.0000
    1.1000    2.0300

All 2D points unique to machine precision:

>> xMerged = builtin('_mergesimpts',x,[eps eps],'first')
xMerged =
    1.0000    2.0000
    1.0600    2.0000
    1.1000    2.0000
    1.1000    2.0300
"""
data = np.array([[1,2],[1.06, 2], [1.1, 2],[1.1, 2.03]])
eps = 2.2204e-16
tols = [eps,eps]

result = np.array([[1,2],
[1.06000000000000,2],
[1.10000000000000,2],
[1.10000000000000,2.03000000000000]])

us = mergesimpts(data,tols,'first')
assert np.all(np.isclose(result,us))

#%%

"""
Merge based on second dimension tolerance:

>> xMerged = builtin('_mergesimpts',x,[eps 0.1],'first')
xMerged =
    1.0000    2.0000
    1.0600    2.0000
    1.1000    2.0000   % first of rows 3 and 4
"""

data = np.array([[1,2],[1.06, 2], [1.1, 2],[1.1, 2.03]])
eps = 2.2204e-16
tols = [eps,0.1]

result = np.array([[1,2],
[1.06000000000000,2],
[1.10000000000000,2]])

us = mergesimpts(data,tols,'first')
assert np.all(np.isclose(result,us))

#%%
"""
>> xMerged = builtin('_mergesimpts',x,[eps 0.1],'average')
xMerged =
    1.0000    2.0000
    1.0600    2.0000
    1.1000    2.0150   % average of rows 3 and 4
"""
data = np.array([[1,2],[1.06, 2], [1.1, 2],[1.1, 2.03]])
eps = 2.2204e-16
tols = [eps,0.1]

result = np.array([[1,2],
[1.06000000000000,2],
[1.10000000000000,2.01500000000000]])

us = mergesimpts(data,tols,'average')
assert np.all(np.isclose(result,us))

#%%
"""
Merge based on first dimension tolerance:

>> xMerged = builtin('_mergesimpts',x,[0.2 eps],'average')
xMerged =
    1.0533    2.0000   % average of rows 1 to 3
    1.1000    2.0300
"""
data = np.array([[1,2],[1.06, 2], [1.1, 2],[1.1, 2.03]])
eps = 2.2204e-16
tols = [0.2,eps]

result = np.array([[1.05333333333333,2],
[1.10000000000000,2.03000000000000]])

us = mergesimpts(data,tols,'average')
assert np.all(np.isclose(result,us))

#%%
"""
>> xMerged = builtin('_mergesimpts',x,[0.05 eps],'average')
xMerged =
    1.0000    2.0000
    1.0800    2.0000   % average of rows 2 and 3
    1.1000    2.0300   % row 4 not merged because of second dimension
"""
data = np.array([[1,2],[1.06, 2], [1.1, 2],[1.1, 2.03]])
eps = 2.2204e-16
tols = [0.05,eps]

result = np.array([[1,2],
[1.08000000000000,2],
[1.10000000000000,2.03000000000000]])

us = mergesimpts(data,tols,'average')
assert np.all(np.isclose(result,us))

#%%
"""
Merge based on both dimensions:

>> xMerged = builtin('_mergesimpts',x,[0.05 .1],'average')
xMerged =
    1.0000    2.0000
    1.0867    2.0100   % average of rows 2 to 4
"""

data = np.array([[1,2],[1.06, 2], [1.1, 2],[1.1, 2.03]])
eps = 2.2204e-16
tols = [0.05,0.1]

result = np.array([[1,2],
[1.08666666666667,2.01000000000000]])

us = mergesimpts(data,tols,'average')
assert np.all(np.isclose(result,us))

