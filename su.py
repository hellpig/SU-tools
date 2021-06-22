# -*- coding: utf-8 -*-
#
# import this Python-3 file via:
#   from su import *
#
# reload via:
#   from importlib import reload
#   import su
#   reload(su)
#   from su import *
# Only the last two lines need to be run if reloading the next time
#
# (c) 2021 Bradley Knockel


import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from matplotlib import animation




def suw(L):
# Finds dimension of irreducible representations of SU(n>1) using Weyl
# dimension formula. That is, this code finds the dimension of the space
# that is spanned by all weights. For an n-by-n irrep, this dimension
# is n.
#
# L = [n1,n2,...] is Dynkin coefficients of highest weight
#   (non-negative integers only)
#
# If dimension is larger than a trillion, this code may be inaccurate
# because it rounds at the end to compensate for round-off errors caused by
# division of double-precision numbers. If the d is greater than 10^15,
# there is much reason to worry because you are beyond the limits of
# double-precision numbers.

  r = len(L)  #SU(r+1)

  D = np.zeros(r)
  for i in range(r):
    ii = i+1
    LL = np.zeros((ii, r-i), dtype=int)
    for j in range(ii):
      LL[j] = L[j:r-i+j]
    D[i] = np.prod( 1.0 * (np.sum(LL, axis=0) + ii) / ii )

  return int(round(np.prod(D)) + 0.5)  # add 0.5 to take care of floating point errors








def suf(L):
# Makes list of weights of in irreducible representation for SU(n>1).
# This list will have info about how many simple roots (alphas) must be
#     subtracted from L.
# This list will also have weight dimension appended on.
# The first row will be [L,zeros(size(L)),1] since L is the first weight to
#     be found, no simple roots must be subtracted from L to get L, and the
#     highest weight always has dimension 1.
#
# L = [n1,n2,...] is Dynkin coefficients of highest weight
#   (non-negative integers only)
#
# L having many elements or L having large elements will make this code
# run a long time in spite of effort to speed things up.

  r = len(L)  #SU(r+1)

  #make Cartan matrix
  A = np.diag(2*np.ones(r, dtype=int)) + np.diag(-1*np.ones(r-1, dtype=int), k=1) + np.diag(-1*np.ones(r-1, dtype=int), k=-1)

  #b is defined as the minimum number of "branches" needed to find the weight
  #from the highest weight, where a branch is a string of weights
  b = 0

  #l will grow by gaining rows as more weights are found, and the last
  #column will eventually be removed and replaced with dimension info
  l = np.array( L + [0]*r + [b] , ndmin = 2)
  #columns of l with weight info are 0:r
  #columns of l with alphas-subtracted info are r:2*r


  #find all weights
  alphas = np.identity(r, dtype=int)  #alphas[i] is the ith simple root in the alpha basis
  GO = sum(L)  #if and only if GO is large will an IF statement speed things up
  while True:
    lSmall = l[b==l[:,-1],:] # l reduced to weights that belong to previous b value
    n = lSmall.shape[0]  # number of weights in reduced l
    if n==0:
        break
    b += 1
    for m in range(n):
        ll = l.shape[0]
        lNow = lSmall[m]
        temp = []
        for i in range(r):  # i looks at the alpha_i string
            if GO>10 and np.any(np.all(npm.repmat(lNow[r:2*r]-alphas[i],ll,1)==l[:,r:2*r], axis=1)):
                continue
            dynk = lNow[i]  #Dynkin coefficient
            if dynk>0:
                for j in range(1,dynk+1):   # j is length of alpha_i string
                    c = lNow[0:r] - j*A[i]  #candidate for new weight
                    if np.any(np.all(npm.repmat(c,ll,1)==l[:,0:r],axis=1)):
                        continue            #continue if c is already a known weight
                    temp.append( np.hstack((c, lNow[r:2*r]+j*alphas[i], b )) )
        if len(temp)>0:
            l = np.vstack((l,temp))

  #get l looking nice
  l[:,-1] = 0   #set last column of l to zero (get rid of branch info)
  ll = l.shape[0]
  i = np.argsort( np.sum(l[:,r:2*r], axis=1) )
  l = l[i]   #sort l by level of weight

  #use Freudenthal's formula to add weight-space-dimension info to l
  #note: positive roots in basis of simple roots are anything with a single
  #      group of adjacent 1's. If r=3, all positive roots are
  #      [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], and [1,1,1].
  l[0,-1] = 1
  d = np.sum( (npm.repmat(L,ll,1)+l[:,0:r]+2*np.ones((ll,r), dtype=int)) * l[:,r:2*r] , axis=1)  #d[i] is the integer denominator
  for i in range(1,ll):

    aa = npm.repmat(l[i,r:2*r],i,1) - l[0:i,r:2*r] #aa[j] are the simple roots that need to be added
    aa = 1.0*aa / np.linalg.norm(aa, np.inf, axis=1)[:,None]  #candidate for positive root that has at least one 1.0
    bb = np.all( np.logical_or(aa==0.0, aa==1.0), axis=1 )
    aa = aa.astype(int)
    summ = np.linalg.norm(np.diff(aa, axis=1), 1, axis=1)

    n = 0  #initialize numerator
    for j in range(i):
        if bb[j]:
            if  (aa[j][0]!=1 and summ[j]==2) or summ[j]<2:
                n += l[j,-1]*sum( l[j,0:r]*aa[j] )
    l[i,-1] = 2*n // d[i]

  return l





### I make helper functions to expose calculations to hadrons()
#  x = su2helper(l)
#  (x,y) = su3helper(l)
#  su4helper(l, ax)
#  su5helper(l, ax)


def su2helper(l):
# get x coordinates from results of suf()
  alpha1 = 1.0  #chosen to be positive
  n1 = alpha1 / 2
  x = l[:,0] * n1
  return x


def su3helper(l):
# get x and y coordinates from results of suf()
  alpha1 = np.array([1.0, 0.0])  #chosen to be in the positive x direction
  alpha2 = np.array([-1.0/2, np.sqrt(3.0)/2])  #chosen to have alpha2[1]>0
  n1 = (2*alpha1 + 1*alpha2) / 3
  n2 = (1*alpha1 + 2*alpha2) / 3
  x = l[:,0]*n1[0] + l[:,1]*n2[0]
  y = l[:,0]*n1[1] + l[:,1]*n2[1]
  return (x,y)


def su4helper(l, ax):
# plot3D the coordinates that result from suf()

  alpha1 = np.array([1.0, 0.0, 0.0])  #chosen to be in the positive x direction
  alpha2 = np.array([-1.0/2, np.sqrt(3.0)/2, 0])  #chosen to be in xy-plane and have a2(1)>0
  alpha3 = np.array([0.0, -1.0/np.sqrt(3.0), -np.sqrt(2.0/3)])  #chosen to have a3(2)<0
  n1 = (3*alpha1 + 2*alpha2 + 1*alpha3) / 4
  n2 = (2*alpha1 + 4*alpha2 + 2*alpha3) / 4
  n3 = (1*alpha1 + 2*alpha2 + 3*alpha3) / 4
  x = l[:,0]*n1[0] + l[:,1]*n2[0] + l[:,2]*n3[0]
  y = l[:,0]*n1[1] + l[:,1]*n2[1] + l[:,2]*n3[1]
  z = l[:,0]*n1[2] + l[:,1]*n2[2] + l[:,2]*n3[2]

  ax.plot3D(x,y,z,'*')
  w = 'ox+o'  # feel free to add to or otherwise modify w!
  for i in range(len(w)):
    a = np.argwhere(l[:,-1] > i+1.5)[:,0]
    l = l[a,:]
    x = x[a]
    y = y[a]
    z = z[a]
    ax.plot3D(x,y,z,w[i], markersize = (i+1)*10 )


def su5helper(l, ax):
# plot3D a slice of coordinates that result from suf()

  alpha1 = np.array([1.0, 0.0, 0.0, float("nan")])  #chosen to be in the positive x direction
  alpha2 = np.array([-1.0/2, np.sqrt(3.0)/2, 0, float("nan")])  #chosen to be in xy-plane and have a2(1)>0
  alpha3 = np.array([0.0, -1.0/np.sqrt(3.0), -np.sqrt(2.0/3), float("nan")])  #chosen to have a3(2)<0
  alpha4 = np.array([0.0, 0.0, np.sqrt(6.0)/4, float("nan")])
  n1 = (4*alpha1 + 3*alpha2 + 2*alpha3 + 1*alpha4)/5
  n2 = (3*alpha1 + 6*alpha2 + 4*alpha3 + 2*alpha4)/5
  n3 = (2*alpha1 + 4*alpha2 + 6*alpha3 + 3*alpha4)/5
  n4 = (1*alpha1 + 2*alpha2 + 3*alpha3 + 4*alpha4)/5
  x = l[:,0]*n1[0] + l[:,1]*n2[0] + l[:,2]*n3[0] + l[:,3]*n4[0]
  y = l[:,0]*n1[1] + l[:,1]*n2[1] + l[:,2]*n3[1] + l[:,3]*n4[1]
  z = l[:,0]*n1[2] + l[:,1]*n2[2] + l[:,2]*n3[2] + l[:,3]*n4[2]

  ax.plot3D(x,y,z,'*')
  w = 'ox+o'  # feel free to add to or otherwise modify w!
  for i in range(len(w)):
    a = np.argwhere(l[:,-1] > i+1.5)[:,0]
    l = l[a,:]
    x = x[a]
    y = y[a]
    z = z[a]
    ax.plot3D(x,y,z,w[i], markersize = (i+1)*10 )


# A su6helper() would follow the same pattern with...
#   alpha5 = np.array([0.0, 0.0, 0.0, float("nan"), float("nan")])
#   n1 = (5*alpha1 + 4*alpha2 + 3*alpha3 + 2*alpha4 + 1*alpha5)/6
#   n2 = (4*alpha1 + 8*alpha2 + 6*alpha3 + 4*alpha4 + 2*alpha5)/6
#   n3 = (3*alpha1 + 6*alpha2 + 9*alpha3 + 6*alpha4 + 3*alpha5)/6
#   n4 = (2*alpha1 + 4*alpha2 + 6*alpha3 + 8*alpha4 + 4*alpha5)/6
#   n5 = (1*alpha1 + 2*alpha2 + 3*alpha3 + 4*alpha4 + 5*alpha5)/6




def su2(L):
# This code finds and plots all weights given a highest weight of an
# irreducible representation of SU(2).
#
# L is Dynkin coefficient of highest weight (non-negative integer only)

  try:
    len(L)
  except:
    L = [L]

  #display the input
  r = 1
  if len(L)!=r:
    print('L must be a ' + str(r) + '-element vector')
    return
  print('L = ' + str(L))

  #find all weights and their dimensions
  l = suf(L)

  #I now use Weyl dimension formula to get dimension of representation L
  dimension = suw(L)
  print('dimension = ' + str(dimension))
  if dimension != sum(l[:,-1]):
    print('Error: Weyl dimension formula does not agree with # of weights found!')
    return


  ## plot the weights (I take the lengths of the simple roots to be 1)

  x = su2helper(l)
  y = x * 0

  fig = plt.figure()
  plt.plot(x,y,'*')
  plt.xlabel('x')
  fig.suptitle(u'SU(2) with Λ = ' + str(L))
  v = plt.axis()
  plt.axis([v[0]-2,v[1]+2,v[2],v[3]])
  plt.show()







def su3(L):
# This code finds and plots all weights given a highest weight of an
# irreducible representation of SU(3).
#
# L = [n1,n2] is Dynkin coefficients of highest weight
#   (non-negative integers only)

  #display the input
  r = 2
  if len(L)!=r:
    print('L must be a ' + str(r) + '-element vector')
    return
  print('L = ' + str(L))

  #find all weights and their dimensions
  l = suf(L)

  #I now use Weyl dimension formula to get dimension of representation L
  dimension = suw(L)
  print('dimension = ' + str(dimension))
  if dimension != sum(l[:,-1]):
    print('Error: Weyl dimension formula does not agree with # of weights found!')
    return


  ## plot the weights (I take the lengths of the simple roots to be 1)

  (x,y) = su3helper(l)

  fig = plt.figure()
  plt.plot(x,y,'*')
  plt.xlabel('x')
  plt.ylabel('y')
  fig.suptitle(u'SU(3) with Λ = ' + str(L))
  plt.axis('equal')
  v = plt.axis()
  plt.axis([v[0]-1,v[1]+1,v[2]-1,v[3]+1])

  w = 'ox+o'  # feel free to add to or otherwise modify w!
  for i in range(len(w)):
    a = np.argwhere(l[:,-1] > i+1.5)[:,0]
    l = l[a,:]
    x = x[a]
    y = y[a]
    plt.plot(x,y,w[i], markersize = (i+1)*10)

  plt.show()








def su4(L):
# This code finds and plots all weights given a highest weight of an
# irreducible representation of SU(4).
#
# L = [n1,n2,n3] is Dynkin coefficients of highest weight
#    (non-negative integers only)

  #display the input
  r = 3
  if len(L)!=r:
    print('L must be a ' + str(r) + '-element vector')
    return
  print('L = ' + str(L))

  #find all weights and their dimensions
  l = suf(L)

  #I now use Weyl dimension formula to get dimension of representation L
  dimension = suw(L)
  print('dimension = ' + str(dimension))
  if dimension != sum(l[:,-1]):
    print('Error: Weyl dimension formula does not agree with # of weights found!')
    return


  ## plot the weights (I take the lengths of the simple roots to be 1)

  fig = plt.figure()
  ax = plt.axes(projection="3d")
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  fig.suptitle(u'SU(4) with Λ = ' + str(L))

  su4helper(l, ax)

  plt.show()








def su5(L):
# This code finds and plots all weights given a highest weight of an
# irreducible representation of SU(5).
#
# L = [n1,n2,n3,n4] is Dynkin coefficients of highest weight
#    (non-negative integers only)

  #display the input
  r = 4
  if len(L)!=r:
    print('L must be a ' + str(r) + '-element vector')
    return
  print('L = ' + str(L))

  #find all weights and their dimensions
  l = suf(L)

  #I now use Weyl dimension formula to get dimension of representation L
  dimension = suw(L)
  print('dimension = ' + str(dimension))
  if dimension != sum(l[:,-1]):
    print('Error: Weyl dimension formula does not agree with # of weights found!')
    return


  ## plot the weights (I take the lengths of the simple roots to be 1)

  i = 0
  while True:
    a = np.argwhere(l[:,-2] == i)[:,0]
    if a.size == 0:
        break

    fig = plt.figure(i)
    plt.clf()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.suptitle(u'SU(5) with Λ = ' + str(L) + u'. This plot looks at Λ - k1 α1 - k2 α2 - k3 α3 - ' + str(i) + u' α4')
    i += 1

    su5helper(np.array(l[a,:]), ax)

  plt.show()










def suy(*argv):
# Finds all irreps within the tensor product of SU(N>1) irreps:
#     SU(N) x SU(N) x ...
#     where all N's are the same.
# Each row of the output will be an irrep.
#
# suy(L1,L2,...) or suy(L1,L2,...,-1):
# Li=[n1,n2,...,nr] is Dynkin coefficients of highest weight (non-negative
#     integers only, and r of all L's must be the same)
# Make the last input be -1 if you want the dimension info to be output.
#
# If the Young diagrams associated with the L's have lots of boxes or if
#     there are many L's, this code can take a very long time.


  # check to see if last input is -1
  inputs = len(argv)
  printDim = False
  if inputs == 0:
      print('too few inputs')
      return
  if argv[-1] == -1:
      inputs -= 1
      printDim = True
  if inputs == 0:
      print('too few inputs')
      return
    

  # make sure all L's have the same r, find their dimension, and sort
  r = len(argv[0])           # SU(r+1)
  D = np.zeros(inputs, dtype=int)
  n = np.zeros(inputs, dtype=int)       # number of boxes in Young diagram
  L = np.zeros((inputs,r), dtype=int)   # L[i,:] is the ith L
  for i in range(inputs):
    if len(argv[i])!=r:
        print('inputs must be correct length')
        return
    L[i,:] = argv[i]
    D[i] = suw(argv[i])
    n[i] = np.sum( L[i,:] * np.arange(1, r+1) )
  Di = np.prod(D)            # total dimension (initial calculation)
  i = np.flip(np.argsort(n)) # indices to sort n in descending order
  L = L[i,:]    # L is sorted to make the code run as fast as possible

  # take care of simplest cases
  if inputs == 1 or r == 0:
      LL = argv[0]
      if printDim:
          LL.append(D[0])
      return np.array(LL, dtype=int)

  # the Young diagram for ith L has Y[i,j] blocks in the jth row
  Y = np.cumsum(np.flip(L, axis=1), axis=1)
  Y = np.flip(Y, axis=1)

  # initialize YY, where YY[i,:] will represent a Young diagram
  # YY[i,k1] represents the blocks in the r+1 rows of diagram
  # YY[i,k2] has to do with which rows have received new blocks
  # YY[i,-1] is a way of keeping track of which diagrams came from the same
  #    parent diagram
  YY = np.concatenate((Y[0,:], [0], np.zeros(1+r, dtype=int), [0]))
  YY = np.reshape(YY, (1,-1))
  k1 = np.arange(r+1)
  k2 = np.arange(r+1, 2*(r+1))

  # Of above variables, the following will now be the only used variables:
  #    Y[], YY[], k1[], k2[], L[], r, inputs, Di

  ######## find irreps! ########
  I = np.identity(r+1, dtype=int)
  for N in range(1,inputs):
    if not np.any( L[N,:]!=0 ):
      rows=0
    else:
      rows = np.argwhere( L[N,:]!=0 )[-1,0] + 1
    history = [0]*rows      # will contain info for previous rows
    for i in range(rows):   # all rows of the Young diagram that will be multiplying all diagrams in YY

        # add blocks
        for j in range(Y[N,i]):   # all columns of current row
            temp = np.array([], dtype=int).reshape(0,YY.shape[1])
            for k in range(YY.shape[0]):   # all prior Young diagrams
                for l in np.argwhere( np.diff( np.concatenate(([-1],YY[k,k1])) ) != 0 )[:,0]:  # all rows of a Young diagram that can accept another block
                    temp = np.concatenate(( temp, np.concatenate(( YY[k,k1]+I[l,:], YY[k,k2]+I[l,:], [YY[k,-1]] )).reshape(1,-1) ), axis=0)
            YY = np.unique(temp, axis=0)   # remove repeated diagrams then update YY

        # remove diagrams with multiple entries in the same column
        d = []  # rows of YY to be deleted
        for j in range(YY.shape[0]):   # all Young diagrams
            c = np.array([], dtype=int)   # columns that have received blocks
            for k in range(r+1):   # all rows
                c = np.concatenate(( c, np.arange(YY[j,k] + 1 - YY[j,k+r+1], YY[j,k]+1) ))
            if np.any(np.diff(np.sort(c))==0):
                d.append(j)
        YY = np.delete(YY, d, axis=0)

        # update history
        history[i] = np.concatenate(( np.zeros((YY.shape[0],1), dtype=int), np.cumsum(YY[:,k2[0:r]], axis=1)), axis=1)   # info about how many blocks have been added above a given row
        for j in range(i):    # previous histories must grow in size
            history[j] = history[j][YY[:,-1],:]

        # relabel rows of YY
        YY[:,-1] = np.arange(YY.shape[0])

        # remove some diagrams using history
        d=[]  # rows of YY to be deleted
        for j in range(YY.shape[0]):  # all Young diagrams
            # clever (magical!) code
            for k in range(i):        # previous rows of the Young diagram that is multiplying all the others
                if np.any(history[k][j,:]-history[i][j,:]-YY[j,k2] < 0):
                    d.append(j)
                    break
        YY = np.delete(YY, d, axis=0)

        #prepare YY for next iteration
        YY[:,k2] = 0

    # remove columns of Young diagram with r+1 blocks
    YY -= np.concatenate(( npm.repmat(YY[:,r].reshape(-1,1), 1, r+1), np.zeros((YY.shape[0], r+2), dtype=int) ), axis=1)

  # get rid of the no-longer-needed extra columns of YY
  YY = YY[:, 0:r]

  # convert Young diagrams to highest weights
  LL = -np.diff( np.hstack((YY, np.zeros((YY.shape[0],1), dtype=int))) )

  # find the dimension of all irreps in LL to check
  D = np.zeros(LL.shape[0], dtype=int)
  for i in range(LL.shape[0]):
    D[i] = suw(LL[i,:])
  Df = np.sum(D)
  if printDim:
    LL = np.hstack(( LL, D.reshape(-1,1) ))
  if Di != Df:
    print('Warning: Final dimension does not equal initial dimension!')
    print('Dimension_initial =', Di)
    print('Dimension_final =', Df)

  return LL









def hadrons(string):
# Generate animation of flavor-state multiplets for hadrons.
# Close the video by pushing the "x" in corner.
# Dimension of weight will be marked by *, o, x, +, and o.
#
# hadrons('ns'):
# n = 2, 3, 4, 5, or 6 to choose between SU(n)
# s = b or m to choose between baryons or mesons (not required for SU(2))


  def hadr(s):
  # to make the figure

      scaling = 2   #sets size of figure

      fig = plt.figure()
      fig.canvas.manager.set_window_title(s)

      p = fig.get_size_inches()
      fig.set_size_inches( scaling*p[0], scaling*p[1] )

      return fig




  def had(fig, h):
  # to make the video

      e=15        # elevation in degrees
      delay = 1   # per frame (in milliseconds)

      def animate(frame):
          for k in h:
              k.view_init(e, frame % 360)
          return h

      _ = animation.FuncAnimation(fig, animate, frames=360000000, interval=delay, blit=True)
      plt.show()





  if string in ['2b','2m','2M','2B','2',2]:
        N=6   #Number of plots

        fig = hadr('SU(2) Hadrons')  #create and setup the figure

        #data needed later
        h = [0]*N               #initialize the list of axis handles of the N plots
        bb=[1,3,5,6,7,8]         #subplot positions of the N plots
        cc=[[0], [1], [2], [0], [3], [1]]    #highest weights for the N plots
        dd = [0]*N               #titles of the N plots
        dd[0] = '0 du (anti)quarks'
        dd[1] = '1 du (anti)quarks'
        dd[2] = '2 du (anti)quarks'
        dd[3] = '2 du (anti)quarks'
        dd[4] = '3 du (anti)quarks'
        dd[5] = '3 du (anti)quarks'

        #put the N plots onto the figure
        for i in range(N):
            h[i] = plt.subplot(4,2,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            x = su2helper(suf(cc[i]))
            y = x * 0
            z = x * 0
            h[i].plot3D(x,y,z,'*')

        had(fig, h)    #animate the figure



  elif string in ['3b','3B']:
        N=7   #Number of plots

        fig = hadr('SU(3) Baryons')  #create and setup the figure

        #data needed later
        h = [0]*N                  #initialize the list of axis handles of the N plots
        bb=[1,4,7,8,10,11,12]         #subplot positions of the N plots
        cc=[[0,0], [1,0], [2,0], [0,1], [3,0], [1,1], [0,0]]      #highest weights for the N plots
        dd = [0]*N                 #titles of the N plots
        dd[0] = '1 (0 dus quarks)'
        dd[1] = '3 (1 dus quark)'
        dd[2] = '6 (2 dus quarks)'
        dd[3] = '3bar (2 dus quarks)'
        dd[4] = '10 (3, baryon decuplet)'
        dd[5] = '8 (3, baryon octet)'
        dd[6] = '1 (3, completely antisymmetric)'

        #put the N plots onto the figure
        for i in range(N):

            h[i] = plt.subplot(4,3,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            l = suf(cc[i])
            (x,y) = su3helper(l)
            z = x * 0
            h[i].plot3D(x,y,z,'*')

            # put an 'o' when there are two on the same coordinate
            a = np.argwhere(l[:,-1] > 1.5)[:,0]
            h[i].plot3D(x[a],y[a],z[a],'o', markersize = 10)
            if np.any( l[:,-1] > 2.5 ):
              print("huh")

        had(fig, h)    #animate the figure



  elif string in ['3m','3M']:
        N=5   #Number of plots

        fig = hadr('SU(3) Mesons')  #create and setup the figure

        #data needed later
        h = [0]*N             #initialize the list of axis handles of the N plots
        bb=[1,3,4,5,6]         #subplot positions of the N plots
        cc=[[0,0], [1,0], [0,1], [1,1], [0,0]]         #highest weights for the N plots
        dd = [0]*N            #titles of the N plots
        dd[0] = '1 (0 dus quarks and antiquarks)'
        dd[1] = '3 (1 dus quark)'
        dd[2] = '3bar (1 dus antiquark)'
        dd[3] = '8 (2, part of meson nonet)'
        dd[4] = '1 (2, part of meson nonet)'

        #put the N plots onto the figure
        for i in range(N):

            h[i] = plt.subplot(3,2,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            l = suf(cc[i])
            (x,y) = su3helper(l)
            z = x * 0
            h[i].plot3D(x,y,z,'*')

            # put an 'o' when there are two on the same coordinate
            a = np.argwhere(l[:,-1] > 1.5)[:,0]
            h[i].plot3D(x[a],y[a],z[a],'o', markersize = 10)
            if np.any( l[:,-1] > 2.5 ):
              print("huh")

        had(fig, h)    #animate the figure



  elif string in ['4b','4B']:
        N=7   #Number of plots

        fig = hadr('SU(4) Baryons')  #create and setup the figure

        #data needed later
        h = [0]*N             #initialize the list of axis handles of the N plots
        bb=[1,4,7,8,10,11,12]         #subplot positions of the N plots
        cc=[[0,0,0], [1,0,0], [2,0,0], [0,1,0], [3,0,0], [1,1,0], [0,0,1]]     #highest weights for the N plots
        dd = [0]*N             #titles of the N plots
        dd[0] = '1 (when B=-3)'
        dd[1] = '4 (when B=-2)'
        dd[2] = '10 (when B=-1)'
        dd[3] = '6 (when B=-1)'
        dd[4] = '20 (when B=0)'
        dd[5] = '20'' (when B=0)'
        dd[6] = '4bar (when B=0)'

        #put the N plots onto the figure
        for i in range(N):
            h[i] = plt.subplot(4,3,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            su4helper(suf(cc[i]), h[i])

        had(fig, h)    #animate the figure



  elif string in ['4m','4M']:
        N=4   #Number of plots

        fig = hadr('SU(4) Mesons')  #create and setup the figure

        #data needed later
        h = [0]*N            #initialize the list of axis handles of the N plots
        bb=[1,3,4,5]         #subplot positions of the N plots
        cc=[[0,0,1], [1,0,1], [0,0,0], [1,0,0]]         #highest weights for the N plots
        dd = [0]*N            #titles of the N plots
        dd[0] = '4bar (when B=-1)'
        dd[1] = '15 (when B=0)'
        dd[2] = '1 (when B=0)'
        dd[3] = '4 (when B=1)'

        #put the N plots onto the figure
        for i in range(N):
            h[i] = plt.subplot(3,2,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            su4helper(suf(cc[i]), h[i])

        had(fig, h)    #animate the figure



  elif string in ['5b','5B']:
        N=9   #Number of plots

        fig = hadr('SU(5) Baryons')  #create and setup the figure

        #data needed later
        h = [0]*N         #initialize the list of axis handles of the N plots
        bb=[1,4,7,10,2,5,8,3,6]         #subplot positions of the N plots
        dd = [0]*N         #titles of the N plots
        dd[0] = '35 (exited baryons)'
        dd[1] = ''
        dd[2] = ''
        dd[3] = ''
        dd[4] = '40 (unexcited baryons)'
        dd[5] = ''
        dd[6] = ''
        dd[7] = '10bar (completely antisymmetric flavor states)'
        dd[8] = ''

        #put the N plots onto the figure
        l = suf([3,0,0,0])
        j = 0
        for i in range(4):
            h[i] = plt.subplot(4,3,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            a = np.argwhere(l[:,-2] == j)[:,0]  # get the 3D slice of 4D space
            su5helper(np.array(l[a,:]), h[i])
            j += 1
        l = suf([1,1,0,0])
        j = 0
        for i in range(4,7):
            h[i] = plt.subplot(4,3,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            a = np.argwhere(l[:,-2] == j)[:,0]  # get the 3D slice of 4D space
            su5helper(np.array(l[a,:]), h[i])
            j += 1
        l = suf([0,0,1,0])
        j = 0
        for i in range(7,9):
            h[i] = plt.subplot(4,3,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            a = np.argwhere(l[:,-2] == j)[:,0]  # get the 3D slice of 4D space
            su5helper(np.array(l[a,:]), h[i])
            j += 1

        had(fig, h)    #animate the figure



  elif string in ['5m','5M']:
        N=4   #Number of plots

        fig = hadr('SU(5) Mesons')  #create and setup the figure

        #data needed later
        h = [0]*N         #initialize the list of axis handles of the N plots
        bb=[1,3,5,2]         #subplot positions of the N plots
        dd = [0]*N         #titles of the N plots
        dd[0] = '24'
        dd[1] = ''
        dd[2] = ''
        dd[3] = '1'

        #put the N plots onto the figure
        l = suf([1,0,0,1])
        j = 0
        for i in range(3):
            h[i] = plt.subplot(3,2,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            a = np.argwhere(l[:,-2] == j)[:,0]  # get the 3D slice of 4D space
            su5helper(np.array(l[a,:]), h[i])
            j += 1
        l = suf([0,0,0,0])
        j = 0
        for i in range(3,4):
            h[i] = plt.subplot(3,2,bb[i], projection="3d", title=dd[i])
            plt.axis('off')
            a = np.argwhere(l[:,-2] == j)[:,0]  # get the 3D slice of 4D space
            su5helper(np.array(l[a,:]), h[i])
            j += 1

        had(fig, h)    #animate the figure



  else:
        print('invalid input string')

