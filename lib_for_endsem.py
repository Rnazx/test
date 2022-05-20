from numpy import *
import numpy as np

##################################################################
##################################################################
#############CODE FOR GAUSSIAN QUADRATURE#########################
##################################################################
# Recursive generation of the Legendre polynomial of order n
def Legendre(n,x):
	x=np.array(x)
	if (n==0):
		return x*0+1.0
	elif (n==1):
		return x
	else:
		return ((2.0*n-1.0)*x*Legendre(n-1,x)-(n-1)*Legendre(n-2,x))/n
 
##################################################################
# Derivative of the Legendre polynomials
def DLegendre(n,x):
	x=np.array(x)
	if (n==0):
		return x*0
	elif (n==1):
		return x*0+1.0
	else:
		return (n/(x**2-1.0))*(x*Legendre(n,x)-Legendre(n-1,x))
##################################################################
# Roots of the polynomial obtained using Newton-Raphson method
def LegendreRoots(polyorder,tolerance=1e-20):
        if polyorder<2:
            err=1 # bad polyorder no roots can be found
        else : 
            roots=[]
            # The polynomials are alternately even and odd functions. So we evaluate only half the number of roots. 
            for i in range(1,int(polyorder/2) +1):
                x = np.cos(np.pi*(i-0.25)/(polyorder+0.5))
                error=10*tolerance
                iters=0
                while (error>tolerance) and (iters<1000):
                    dx=-Legendre(polyorder,x)/DLegendre(polyorder,x)
                    x=x+dx
                    iters=iters+1
                    error=abs(dx)
                roots.append(x)
            # Use symmetry to get the other roots
            roots=np.asarray(roots)
            if polyorder%2==0:
                roots=np.concatenate( (-1.0*roots, roots[::-1]) )
            else:
                roots=np.concatenate( (-1.0*roots, [0.0], roots[::-1]) )
            err=0 # successfully determined roots
        return [roots, err]
##################################################################
# Weight coefficients
def GaussLegendreWeights(polyorder):
	W=[]
	[xis,err]=LegendreRoots(polyorder)
	if err==0:
		W=2.0/( (1.0-xis**2)*(DLegendre(polyorder,xis)**2) )
		err=0
	else:
		err=1 # could not determine roots - so no weights
	return [W, xis, err]
##################################################################
# The integral value 
# func 		: the integrand
# a, b 		: lower and upper limits of the integral
# polyorder 	: order of the Legendre polynomial to be used
#
def GaussLegendreQuadrature(func, polyorder, a, b):
	[Ws,xs, err]= GaussLegendreWeights(polyorder)
	if err==0:
		ans=(b-a)*0.5*sum( Ws*func( (b-a)*0.5*xs+ (b+a)*0.5 ) )
	else: 
		# (in case of error)
		err=1
		ans=None
	return [ans,err]
##################################################################
# The integrand - change as required
def func(x):
	return 2*x
##################################################################
# 
 
# order=5
# [Ws,xs,err]=GaussLegendreWeights(order)
# if err==0:
# 	print("Order    : ", order)
# 	print("Roots    : ", xs)
# 	print("Weights  : ", Ws)
# else:
# 	print("Roots/Weights evaluation failed")
 
# # Integrating the function
# [ans,err]=GaussLegendreQuadrature(func , order, -3,3)
# if err==0:
# 	print( "Integral : ", ans)
# else:
# 	print("Integral evaluation failed")

##################################################################
##################################################################
#############CODE FOR LU and CHOLESKY#############################
##################################################################
def forward_backward(L, U, b):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        sum = 0
        for j in range(i):
            sum += L[i, j] * y[j]
        y[i] = (b[i] - sum) / L[i,i]

    for i in reversed(range(n)):
        sum = 0
        for j in range(i + 1, n):
            sum += U[i, j] * x[j]
        x[i] = (y[i] - sum) / U[i, i]
    return x


def partial_pivot(A, b):
    count = 0  
    n = len(A)
    for i in range(n - 1):
        if abs(A[i][i]) < 1e-10:
            for j in range(i + 1, n):
                if abs(A[j][i]) > abs(A[i][i]):
                    A[j], A[i] = (A[i], A[j], )  
                    count += 1
                    b[j], b[i] = ( b[i], b[j],)  
    return A, b, count

def crout(A):
    n = len(A)

    U = np.zeros((n,n))
    L = np.zeros((n,n))

    for i in range(len(A)):
        L[i][i] = 1

    for j in range(len(A)):
        for i in range(len(A)):
            sum = 0
            for k in range(i):
                sum += L[i][k] * U[k][j]
            if i == j:
                U[i][j] = A[i][j] - sum
            elif i > j:
                L[i][j] = (A[i][j] - sum) / U[j][j]
            else:
                U[i][j] = A[i][j] - sum

    return L, U

# solving x with crout's lu decomposition
def solvex_lu(A, b):
    partial_pivot(A, b)
    L, U = crout(A)
    x = forward_backward(L, U, b)
    return x


def cholesky(A, b):
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        sum1 = 0
        for j in range(i):
            sum1 += L[i,j]**2
        L[i,i] = np.sqrt(A[i,i] - sum1)

        for j in range(i+1, n):
            sum2 = 0
            for k in range(i):
                sum2 += L[i,k]*L[j,k]
            L[j,i] = 1/L[i,i] * (A[i,j] - sum2)

    x = forward_backward(L, L.T, b)
    return x

##################################################################
##################################################################
#############CODE FOR POLYNOMIAL FIT AND CHEBYSHEV FIT############
##################################################################
def poly_fit(X, Y, d = 1):
    
    n = len(X)
    p = d + 1  
    A = np.zeros((p, p))  
    b = np.zeros(p)  

    for i in range(p):
        for j in range(p):
            sum = 0
            for k in range(n):
                sum += X[k] ** (i + j)

            A[i, j] = sum

    for i in range(p):
        sum = 0
        for k in range(n):
            sum += X[k] ** i * Y[k]

        b[i] = sum

    x = solvex_lu(A, b)
    return x


def cheby_poly(x, order):
    if order == 0: return 1
    elif order == 1: return 2 * x - 1
    elif order == 2: return 8 * x**2 - 8 * x + 1
    elif order == 3: return 32 * x**3 - 48 * x**2 + 18 * x - 1


def cheby_fit(X, Y, d = 3):
    n = len(X)
    p = d + 1
    A = np.zeros((p, p))
    b = np.zeros(p)

    for i in range(p):
        for j in range(p):
            sum = 0
            for k in range(n):
                sum += cheby_poly(X[k], j) * cheby_poly(X[k], i)
            A[i, j] = sum

    for i in range(p):
        sum = 0
        for k in range(n):
            sum += cheby_poly(X[k], i) * Y[k]
        b[i] = sum

    x = solvex_lu(A, b)
    return x


##################################################################
##################################################################
#############CODE FOR RANDOM WALK############
##################################################################
# def f(x):
#     return np.sqrt(1-x**2)
  
# N=500000

# def pi_rand(a, m, N,seed1,seed2):
#     X = mlcg(a, m,seed1 , N)/m
#     Y = mlcg(a, m, seed2, N)/m
#     count=0
#     for i in range(N):
#         if(X[i]**2 +Y[i]**2) <=1:
#             count+=1
#     return count*(float(1/N))
  
# def pi_mcinteg(a, m, N, func, seed):
#     xrand = mlcg( a, m,seed, N)/m
#     sum = 0
#     for i in range(N):
#         sum += func(xrand[i])
#     return 1/ float(N) * sum


def mlcg(a, m, seed, N):
    x = seed
    r = []
    for i in range(N):
        x = (a * x) % m
        r.append(x)

    return np.array(r)

# f = open('ass2_fit.txt', 'r')
# data = np.genfromtxt(f, delimiter='')
# f.close()

# plt.scatter(X, Y, c = 'g', label="Datapoints")
# plt.plot(x, y, "r", label="poly-fit")
# plt.plot(x, y_c, "y", label="cheby-fit")

# plt.legend()
# plt.show()