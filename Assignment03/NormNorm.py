


# Alex Maisel
# acm277



import matplotlib.pyplot as plt
from scipy import *
from scipy import stats, integrate

import numpy

# For unit test assertions:
import numpy.testing as npt

from matplotlib.pyplot import *
plt.ion()  # interactive mode on





#4.1



mu = 1
sigma = 5
N = 100
d = stats.norm.rvs(mu,sigma,N)

mu0 = 2
w0 = 4

prior = stats.norm(mu0,w0)

w = sqrt(numpy.var(d))/sqrt(N)


B = (w**2)/(w**2 + w0**2)

dbar = mean(d)

mutilde = dbar + B*(mu0-dbar)
wtilde = w*sqrt(1-B)

posterior = stats.norm(mutilde,wtilde)

a = linspace(-2.5,2.5, 200)



plot(a,posterior.pdf(a),lw = 2,label = "Post")

xlabel(r'$\mu$')
ylabel('PDF')








#4.2

privals = prior.pdf(a)

# based on propto in lec 8 slide 10
likevals = exp(-N*(a - dbar)**2/(2*numpy.var(d)))


prilike = privals*likevals



def trapqueen(arr,alphas):

    total = 0

    for i in range(len(alphas)-1):
        ma = arr[i+1]
        mi = arr[i]

        curr = (alphas[i+1] - alphas[i])*((ma+mi)/2)

        total += curr
    return total


marg = trapqueen(prilike,a)


plot(a,posterior.pdf(a),color = 'pink',lw=3)
plot(a,prilike/marg,linestyle = 'dashed',lw=3,color = 'black')
xlabel('mu')
ylabel('PDF')




#4.3

def test_trapz():
    """
    Test trapezoid rule
    """
    npt.assert_almost_equal(numpy.trapz(prilike, a),trapqueen(prilike,a))


def test_overlap():
    """
    Test that the two Gaussian distro's from 4.1 and 4.3 overlap
    """
    npt.assert_allclose(posterior.pdf(a),prilike/marg,.02)


test_trapz()

test_overlap()
