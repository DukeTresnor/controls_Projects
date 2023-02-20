import numpy as np
from matplotlib import pyplot as plt 



# goal is to simulate the rate of profit of a simplified business



# sales are the total amount of products sold times the price per product. 
# These values are based on market demand for those products -- we'll assume that all products that are produced
# through this production process are sold.
# prodnum is number of products sold
# prodprice is price per product (sold)

# sales = prodnum * prodprice

# the cost for production is formed from constant capital and variable capital
# costs for production include the cost of raw materials that workers add their labor value to,
# the maintenence costs for instruments, machines, tools, etc. that degrade over time,
# and the cost of labor power. The cost of labor power is determined by the amount of money
# needed for the worker to be able to exchange for necessities - ie commodities that sustain
# and propagate the worker's labor power. This cost of labor power is expressed in wages.
# these wages are nominal wages, as opposed to real wages and relative wages, which are different.
# decay is the total cost for maintenence of tools -- it's a fraction of the actual value of each of these tools.
# For example, a production line that included a machine costing 100 with a lifespan of 5 years would incur maintenence costs
# of 100 / 5 = 20 per production period (quarter, year, etc.). If the production line added another tool that cost 300 with
# a lifespan of 3 years, the total maintenence cost for production would now be (100 / 5) + (300 / 3) = 120 per period.

# rawmat and wages are separated from decay as concepts because they must be paid each time production occurs, and so it's useful to keep them separate

# costs = rawmat + wages + decay

# something to keep in mind is that the value of wages is at a minimum determined by the total exchange value of commodities that sustain and propagate labor power,
# but it's total amount is actually this minimum plus some amount that's based on the level of class struggle going on. Thus wages are heavily related to p, the profits
# that are pocketed by the capitalist, and the value of the relative wages is determined by political factors.
# To help focus this mathematical model, then, it might be more appropriate to loop wages and profit into a single value
# this single value is the total amount of value that has been valorized / added to the raw materials of the product through the expenditure of labor power in useful labor.
# the amount of this is purely related to the amount of useful time spent producing products. We'll use addval?
# addval = p + wages


# profit is the amount of sales made minus any costs

# p = sales - costs

# using the formation of addval, this changes the profit equation slightly.
# p = sales - rawmat - wages - decay
# p + wages = sales - rawmat - decay

# addval = sales - rawmat - decay

# this isn't exactly right either, the amount of value added is purely based on the amount of useful labor performed on raw materials.
# more realistically i think, the amount of value that's added is based on the number of hours spend doing useful labor, but in the simulation
# of the production, selling, buying, etc. of this business, the numerical value of addval is going to be equivalent to the above equation
# (since we're assuming all of the produced products are sold).
# essentially this concept of value added is contained within the sum of products, and through the complete selling of them the value of addval is
# determined. So addval is the numerical representation of the value added through useful labor done under the assumption that all of the products
# in the production line will be sold. 

# rate of profit is the amount of profit per investment, ie profit / investment

# rp = p / invest


# ok let's simulate some 100 years.
simt = 100

# costs
decay = np.zeros(simt)
rawmat = np.zeros(simt)
wages = np.zeros(simt)
# revenue or sales
prodnum = np.zeros(simt)
prodprice = np.zeros(simt)



#for i in range(simt):

    # decay fill

    # rawmat fill

    # wage fill

    # prodnum fill

    # prodpice fill

sales = prodnum * prodprice
costs = rawmat + wages + decay
p = sales - costs
invest = costs
rp = p / invest

# plotting
time = np.arange(1, simt+1)
plt.title("Rate of Profit vs time for 100 years")
plt.ylabel("Rate of Profit rp")
plt.xlabel("Time in years")
plt.plot(time, rp)
plt.show()