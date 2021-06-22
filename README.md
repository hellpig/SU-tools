# SU-tools

Many powerful tools for studying irreducible representations (irreps) of SU(n), including making animations of hadron flavor-state multiplets.



## Functions

See the top of su.py for how to import the following functions into Python 3!
The numpy and matplotlib modules are required.

The following powerful functions study irreps of SU(n>1)...
 * suf(L)  - given an irrep, finds all weights and their individual dimensions
 * suw(L)  - using Weyl dimension formula, finds total dimension of all weights corresponding to an irrep

The following very similar functions study specific SU(n) groups with the help of the above two functions...
 * su2(L)  - plots all weights of an SU(2) irrep
 * su3(L)  - plots all weights of an SU(3) irrep
 * su4(L)  - plots all weights of an SU(4) irrep
 * su5(L)  - plots all weights of an SU(5) irrep

The following function uses suw()...
 * suy(L1,L2,...)  - finds all irreps within tensor products of SU(n>1) irreps

The following function studies hadrons...
 * hadrons(string)  - creates animations of flavor-state multiplets for hadrons!

More documentation is found for each function in su.py.



## My goal

Representations of SU(n) interest me. SU(2) describes spin-x particles where x is any non-negative half-integer or integer. SU(3) describes color charge. There are many other physical uses for the SU(n) groups, and I have created general functions that help you study irreducible representations and their weights. I have also written some functions that draw and animate hadron multiplets. I am fascinated by the way that observed hadrons belong to the 4-dimensional diagrams (which is a series of 3-dimensional diagrams) created by the bizarre and abstract math of special unitary groups.

Basically, I intended to write some code that very quickly does all the very long calculations required to study these very interesting groups while giving hadron flavor-states special attention.

The big picture of my code is that you will use L to define an irrep. The length of L is r, which means that SU(r+1) is being studied. To find out what I mean by all these symbols and words, see my list of terminology below.

I have done my best to make the code readable and expandable! Please conform it to your own needs and/or wants!



## Terminology and facts

Important vocab that I use...
 * SU(n) — Lie group of all special unitary n×n matrices (I usually assume n>1)
 * irrep — irreducible matrix representation of SU(n)
 * weight — an "eigenvalue" of the irrep's Cartan subalgebra
 * alpha — simple roots of the Lie algebra for SU(n) that have been given the geometric interpretation as vectors so that weights can be plotted, where it is true that (1) all positive roots are sums of these, (2) the order of simple roots is chosen to be the standard choice that makes the Cartan matrix have zeros everywhere except the 3 central diagonals, and (3) the default basis used to write these vectors is the standard rectangular basis in n-1 dimensional space
 * alpha basis — I sometimes use alpha1=\[1,0,0,...\], alpha2=\[0,1,0,...\], etc. as a basis instead of the default rectangular basis
 * Dynkin coefficients of a weight — every weight has integer Dynkin coefficients that uniquely label the weight, where these coefficients are the weight vector in a basis obtained by multiplying the inverse Cartan matrix with the column vector of alphas
 * highest weight — this is the only weight of an irrep that can form all other weights by subtracting alphas (any list of non-negative Dynkin coefficients is a valid highest weight)
 * dimension — I sometimes refer to the number of spatial dimensions needed to draw something, and I even sometimes mean the dimension of the SU(n) manifold, but I usually am referring to the dimension of a weight space (i.e. the multiplicity of a weight) or the total dimension of the direct sum of all weight spaces (note that the total dimension of an n×n irrep is n-dimensional)

Some variables I will use consistently...
 * r = rank = number of alphas of a group = the number of Dynkin coefficients per weight = the number of spatial dimensions a plot of weights will require = rank of SU(r+1)
 * L = Lambda = Dynkin coefficients of highest weight as a row vector

Interesting L's...
 * L=\[1,0,0,...,0\] is the n×n defining irrep for SU(n), which is often labeled n
 * L=\[0,...,0,0,1\] is another n×n irrep for SU(n) if n>2, which is often labeled nbar
 * L=\[0,...,0\] is the unfaithful representation by the number 1
 * L=\[1,0,...,0,1\] corresponds to the adjoint irrep of SU(n) (for SU(2), L=\[2\] does the trick)

Interesting facts...
 * SU(1)=1 is a 0-dimensional manifold, so it is uninteresting.
 * U(1) is a 1-dimensional manifold, so its Lie algebra has no roots, so it is too simple to be interesting.

For doing tensor products, the following are useful background...  
  [https://youtu.be/hIUAkLVqHVQ](https://www.youtube.com/watch?v=hIUAkLVqHVQ)  
  [https://youtu.be/1rgm59g03M0](https://www.youtube.com/watch?v=1rgm59g03M0)  
The first video is short and to the point, but it is only for SU(2) since it does not give a complete set of rules, and he does not do the trick of removing extra columns of n=2 blocks.
To convert L to a Young diagram, each number in L tells you how many columns have the number of boxes given by its index (starting at 1, not 0).
For example, \[4,5,0\] has 0 columns with 3 boxes, 5 columns with 2 boxes, and 4 columns with 1 box.

Hadron facts...
 * Hadrons are either baryons or mesons. Baryons are 3 quarks, and mesons are a quark and antiquark.
 * SU(2≤n≤6) describes an approximate symmetry of the first n quarks: d u s c b t.
 * This symmetry gets worse as n increases until SU(6) is so bad that the t=top quark is never even found in hadrons.
 * d has D=-1. u has U=1. s has S=-1. c has C=1. b has B=-1. t has T=1.



## More info

 * If you find bugs or have questions, leave me a message at [www.BradleyKnockel.com](https://www.bradleyknockel.com)
 * This code is under the MIT license, so you can mostly do what you want with it (if you distribute the code or a modified version of the code, just make sure you give credit to me, Bradley Knockel)
 * Many thanks to Dr. Daniel Finley!
 * Keywords: Freudenthal's formula, group theory, irreducible representations, irreps, Lie algebra, Lie group, matrix representations, representation theory, special unitary group, SU(2), SU(3), SU(4), SU(5), SU(n), Weyl dimension formula, Young diagrams, Young tableau
 * I thought about trying to speed up suy() by removing repeated Young diagrams then adding an extra column in YY\[\] to store how many of that diagram there are, but this approach may not even speed things up, and I don't greatly care about speed (if I did, this code would be written in C)!

