### This function takes in a tuple? of latitude, longitude, and altitude coordinates over time 
### and returns a list of parameters for a spline function for one moving obstacle.

#import interop - this will eventually be needed when integrating with interop server
import numpy as np 

#Initialize moving obstacle historical data - currently for 4 time steps
#Currently dummy data to check if it can make a linear fit
obst1 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T
#Initializes coefficient matrix for spline function
m, n = obst1.shape
thetas1 = np.empty(shape=(m,n) )


r1 = np.ones(n)
r2 = np.array(range(1, n+1))
r3 = np.power(r2, 2)
r4 = np.power(r2, 3)
T = np.vstack([r1, r2, r3, r4])

#solve for coefficients of cubic spline function using least squares 
#Phi(t)=a+bt+ct^2+dt^3
#first row is coefficients for x direction, 2nd for y, 3rd for z
thetas1 = (np.linalg.pinv(T.T) @ obst1.T).T
print(thetas1)