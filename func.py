import math
import numpy as np

def sphere(a):
    if type(a) is str:
        if a == 'max':
            return 1
        elif a == 'min':
            return -1
    else:
        f_x = np.sum(a*a)
        return f_x

def rastrigin(a):
    if type(a) is str:
        if a == 'max':
            return 5
        elif a == 'min':
            return -5
    else:
        f_x = 10 * len(a)
        for i in range(len(a)):
            f_x += (a[i] * a[i]) - 10 * math.cos(2 * math.pi * a[i])
        return f_x

def ackley(a):
    if type(a) is str:
        if a == 'max':
            return 32.768
        elif a == 'min':
            return -32.768
    else:
        tmp = 0
        for i in range(len(a)):
            tmp += math.cos(2 * math.pi * a[i])
        f_x = 20 + math.e - 20 * math.exp(-0.2 * math.sqrt(np.sum(a*a) / len(a))) - math.exp(tmp / len(a))
        return f_x

def rosenbrock(a):
    if type(a) is str:
        if a == 'max':
            return 5
        elif a == 'min':
            return -5
    else:
        f_x = 0
        for i in range(len(a)-1):
            f_x += 100 * (a[i+1] - a[i] * a[i]) * (a[i+1] - a[i] * a[i]) + (a[i] - 1) * (a[i] - 1)
        return f_x

def beale(a):
    if type(a) is str:
        if a == 'max':
            return 4.5
        elif a == 'min':
            return -4.5
    else:
        f_x = (1.5-a[0]+a[0]*a[1])*(1.5-a[0]+a[0]*a[1])+(2.25-a[0]+a[0]*a[1]*a[1])*(2.25-a[0]+a[0]*a[1]*a[1])+(2.625-a[0]+a[0]*a[1]*a[1]*a[1])*(2.625-a[0]+a[0]*a[1]*a[1]*a[1])
        return f_x

def goldstein(a):
    if type(a) is str:
        if a == 'max':
            return 2
        elif a == 'min':
            return -2
    else:
        f_x = (1+(a[0]+a[1]+1)*(a[0]+a[1]+1)*(19-14*a[0]+3*a[0]*a[0]-14*a[1]+6*a[0]*a[1]+3*a[1]*a[1]))*(30+(2*a[0]-3*a[1])*(2*a[0]-3*a[1])*(18-32*a[0]+12*a[0]*a[0]+48*a[1]-36*a[0]*a[1]+27*a[1]*a[1]))
        return f_x

def booth(a):
    if type(a) is str:
        if a == 'max':
            return 10
        elif a == 'min':
            return -10
    else:
        f_x = (a[0]+2*a[1]-7)*(a[0]+2*a[1]-7)+(2*a[0]+a[1]-5)*(2*a[0]+a[1]-5)
        return f_x

def easom(a):
    if type(a) is str:
        if a == 'max':
            return 100
        elif a == 'min':
            return -100
    else:
        f_x = -math.cos(a[0])*math.cos(a[1])*math.exp(-(a[0]-math.pi)*(a[0]-math.pi)-(a[1]-math.pi)*(a[1]-math.pi))
        return f_x

def matyas(a):
    if type(a) is str:
        if a == 'max':
            return 10
        elif a == 'min':
            return -10
    else:
        f_x = 0.26*(a[0]*a[0]+a[1]*a[1])-0.48*a[0]*a[1]
        return f_x

def levi(a):
    if type(a) is str:
        if a == 'max':
            return 10
        elif a == 'min':
            return -10
    else:
        x = a[0]
        y = a[1]
        f_x = math.sin(3*math.pi*x)*math.sin(3*math.pi*x)+(x-1)*(x-1)*(1+math.sin(3*math.pi*y)*math.sin(3*math.pi*y))+(y-1)*(y-1)*(1+math.sin(2*math.pi*y)*math.sin(2*math.pi*y))
        return f_x

def hump(a):
    if type(a) is str:
        if a == 'max':
            return 5
        elif a == 'min':
            return -5
    else:
        x = a[0]
        y = a[1]
        f_x = 2*x*x-1.05*x*x*x*x+(x*x*x*x*x*x)/6+x*y+y*y
        return f_x