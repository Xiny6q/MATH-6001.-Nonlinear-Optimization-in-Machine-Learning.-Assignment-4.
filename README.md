Download link :https://programming.engineering/product/math-6001-nonlinear-optimization-in-machine-learning-assignment-4/

# MATH-6001.-Nonlinear-Optimization-in-Machine-Learning.-Assignment-4.
MATH 6001. Nonlinear Optimization in Machine Learning. Assignment 4.
There are 3 problems, each problem is worth 5 points, total is 15 points.

1. Consider the minibatch SGD Algorithm 5.1 in the Lecture Notes applied to the standard objective function

1

n

Xi

f(x) =

n

fi(x)

=1

in machine learning, with mini{batch size m n. The iteration is given by

X

xk+1 = xk rfi(xk) ;

m i2Bt

in which Bt f1; 2; :::; ng is an i.i.d sequence of size{m minibatches uniformly sampled from the index set [n] = f1; 2; :::; ng and > 0 is the learning rate. That is to say, Bt is sampled uniformly from all size m{subsets of the index set [n] = f1; 2; :::; ng and for di erent t the choice of Bt are independent of each other. Show that the stochastic

1 n

P gradient term is an unbiased estimator of the standard gradient rf = n i=1

That is

E

1

rfi(x)

! = rf =

1 n

rfi(x)

(0.1)

m i2Bt

n i=1

X

X

(Hint: Each index i 2 f1; 2; :::; ng has been repeated Cnm 11 times in all Cnm possible size m minibatches sampled from f1; 2; :::; ng uniformly. Therefore

1

!

1 n

(n 1)!

n

1 n

r

r

(m 1)!(n m)!

r

r

m i2Bt

mCn i=1

mm!(n m)! i=1

n i=1

E

X

fi(x) =

X

Cnm 11 fi(x) =

X

fi(x) =

X

fi(x) :

m

n!

)

2. Following problem 1, calculate the covariance matrix of the stochastic gradient

term

Cov m

i2Bt rfi(x)! E

m

i2Bt rfi(x) r f(x)!

m

T

i2Bt rfi(x) r f(x)!

1

X

1

X

1

X

and show that

m

i

t rfi(x)! =

m

n

0(x) ;

(0.2)

Cov

1

X

1

1

2B

such that

n

1

Xi

0(x) =

n

1

(rf(x) r fi(x))(rf(x) r fi(x))T :

=1


In particular the equation (0.2) indicates quantitatively the e ect of batchsize on the covariance noise magnitude of SGD training. This has been validated to a ect the generalization of deep learning models trained by SGD. See Keskar, N.S., Mudigere, D., Nocedal, J., Smelyanskiy, M., Tang, P.T.P., On large{batch training for deep learning: generalization gap and sharp minima. ICLR, 2017.

(Hint: Rewrite the stochastic gradient term as

1

X

1

m

X

m i2Bt rfi(x) = m i=1 rf i (x) ;

where = ( 1; :::; m) is such that i is picked with equal probability from f1; 2; :::; ngnf j :

1 j i 1g for all 1 i m. Then

1

Pm

!

1

Pm

!T

T

E m i2Bt rfi(x)

m i2Bt rfi(x)

= E m i=1 rf i (x) m i=1 rf i (x)

1

P

1

TP

1

=

1

P

Erf i (x)rf j (x) :

m2

i;j m

If i = j the above expectation is

1

n

Xk

Erf i (x)rfTj (x) = n

rfk

(x)rfkT (x) :

=1

If i 6= j then the above expectation is

1

n

1

X

X

Erf i (x)rfTj (x) =

E(rf j (x)j i = k)rfkT (x) =

rfj(x)rfkT (x) :

n

n(n 1)

k=1

j6=k


We then make use of (0.1), to see that we have

Cov m i2Bt rfi(x)!

1

P

T

E m i2Bt rfi(x) r f(x)!

m i2Bt rfi(x) r f(x)!

1

P

1

P

!T

= E

1

P

rfi(x)

1

rfi(x)

r f(x)rfT (x)

m

m

T

1

P

T

i2Bt

i2Bt

1

1 1Pn

1

1

1

n

=

Erf i (x)rf j (x) r f(x)rf (x)

m2

i;j

m

=

m

P

rfk(x)rfkT (x) +

m(m 1)

rfj(x)rfkT (x)

rfj(x)rfkT (x)

m2

n

m2

n(n 1)

n2

1

m 1

n

m 1P

1

n

P

k=1

j6=k

j;k=1

=

k=1 rfk(x)rfkT (x) +

j;k=1 rfj(x)rfkT (x)

mn

mn(n

1)

mn(n

1)

n2

1

1

1

n

P

n(m

1)

P

=

k=1 rfk(x)rfkT (x) +

1 rf(x)rfT (x)

m

n

n 1

m(n 1)

1

1

1

P

T

n 1

n(m 1)

n

=

n

1

rf(x)rfT (x)

k=1 rfk(x)rfk (x)

m

n

n 1

n

m(n 1)

n 1

1

1

1

Pn

T

n

= m n

n 1 k=1 rfk(x)rfk (x)

n 1rf(x)rfT (x) :

P

1

n

rfi(x))T

=

n

1

1

n

n 1rf(x)rfT (x).

iP

This gives (0.2), provided that one can easily check 0(x) =

n 1

=1(rf(x)

fi(x))(rf(x)

n

P rfk(x)rfkT (x)

k=1

See Hu, W., Li, C.J., Li, L., Liu, J., On the di usion approximation of nonconvex stochastic gradient descent. Annals of Mathematical Science and Applications, Vol. 4, No. 1 (2019), pp. 3-32.)

Show that when f is strongly convex then the curvature condition holds, i.e., for any x; y 2 Dom(f) and x 6= y we have

(x y)T (rf(x) r f(y)) > 0 :

(Hint: Let us say f is strongly m{convex. Then by Taylor’s formula

1

rf(x) r f(y) = r2f(y + t(x y))(x y)dt ;

0

which implies that

1

(x y)T (rf(x) r f(y)) =

0

(x y)T r2f(y + t(x y))(x y)dt mkx yk2 > 0

as desired.)

3
