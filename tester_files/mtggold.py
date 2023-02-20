import numpy as np
from matplotlib import pyplot as plt


# Goal is to graph curve distributions for colors and color pairs

# setting color curve distributions
w = np.array([ 4.5, 5.0, 4.0, 2.0, 0.5, 1.0 ])
u = np.array([ 3.0, 5.0, 4.0, 2.0, 2.0, 1.0 ])
b = np.array([ 4.5, 5.0, 2.5, 3.0, 2.0, 0.0 ])
r = np.array([ 4.0, 4.5, 3.5, 4.0, 1.0, 0.0 ])
g = np.array([ 3.0, 4.5, 4.0, 2.0, 1.5, 2.0 ])

# creature distributions per color
# hybrid noncreatures are ub equipment cmc 1,
# br removal spell cmc 3,
# and ur looter cmc 2
wc = np.array([ 1.5, 4.0, 3.0, 2.0, 0.5, 1.0 ])
uc = np.array([ 0.5, 1.5, 2.0, 2.0, 1.0, 1.0 ])
bc = np.array([ 2.0, 1.0, 2.0, 2.0, 2.0, 0.0 ])
rc = np.array([ 1.0, 2.0, 3.5, 2.0, 1.0, 0.0 ])
gc = np.array([ 1.0, 1.5, 3.0, 2.0, 1.5, 2.0 ])
# color noncreature distributions
wnc = np.subtract(w, wc)
unc = np.subtract(u, uc)
bnc = np.subtract(b, bc)
rnc = np.subtract(r, rc)
gnc = np.subtract(g, gc)


# Setting color pair curve distributions
wu = np.add(w, u)
ub = np.add(u, b)
br = np.add(b, r)
rg = np.add(r, g)
gw = np.add(g, w)
wb = np.add(w, b)
ur = np.add(u, r)
bg = np.add(b, g)
rw = np.add(r, w)
gu = np.add(g, u)


# X set = m21 common and uncommon creature distributions
# uncommon creatres are at half rate from commons
Xwc = np.array([ 2.5, 4.0, 4.0, 2.5, 2.5, 0.0 ])
Xuc = np.array([ 1.0, 2.5, 3.0, 2.5, 1.5, 2.0 ])
Xbc = np.array([ 1.5, 4.0, 3.0, 2.5, 1.0, 1.5 ])
Xrc = np.array([ 1.0, 4.0, 2.5, 2.0, 1.5, 1.5 ])
Xgc = np.array([ 1.5, 2.5, 4.0, 3.0, 2.0, 2.0 ])



# cmc array -- lists 1 to 6+ cmc
cmc = np.array([ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ])

# plotting
colorConcat = np.array([ w, u, b, r, g ])
colorConcatc = np.array([ wc, uc, bc, rc, gc ])
colorConcatcX = np.array([ Xwc, Xuc, Xbc, Xrc, Xgc ])
labelConcat = ['white', 'blue', 'black', 'red', 'green']
colorList = ['c', 'b', 'k', 'r', 'g']
for i in range(5):
    labelColor = labelConcat[i]
    color = colorList[i]
    plt.plot(cmc, colorConcatc[i], color, label = labelColor)

plt.xlabel('cmc')
plt.ylabel('frequency')
plt.title('Mono Color')
plt.legend(loc='best')
plt.show()


#fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#ax1.plot(cmc, colorConcatc[i], color, label = labelColor)
#ax1.set_title('Custom vs ZNR')
#ax2.plot(cmc, colorConcatc[i], color, label = labelColor)


# plotting custom vs znr





colorConcat = np.array([ w, u, b, r, g ])
colorConcatc = np.array([ wc, uc, bc, rc, gc ])
colorConcatcX = np.array([ Xwc, Xuc, Xbc, Xrc, Xgc ])
labelConcat = ['white', 'blue', 'black', 'red', 'green']
colorList = ['c', 'b', 'k', 'r', 'g']
fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
for i in range(5):
    labelColor = labelConcat[i]
    color = colorList[i]
    
    ax1.plot(cmc, colorConcatc[i], color, label = labelColor)
    ax2.plot(cmc, colorConcatcX[i], color, label = labelColor)

ax1.set_xlabel('cmc')
ax1.set_ylabel('custom frequency')
ax2.set_ylabel('actual frequency')
ax1.set_title('Custom vs Actual')
plt.legend(loc='best')
plt.show()



