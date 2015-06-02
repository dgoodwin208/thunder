"""
Class for performing non-negative matrix factorization
"""

from thunder.rdds.series import Series


# TODO use RowMatrix throughout
class NMF(object):
    """
    Non-negative matrix factorization on a distributed matrix.

    Parameters
    ----------
    method : string, optional, default 'als'
        Specifies which iterative algorithm is to be used. Currently only 'als' supported

    k : int, optional, default = 5
        Size of low-dimensional basis

    maxIter : int, optional, default = 20
        Maximum number of iterations

    tol : float, optional, default = 0.001
        Tolerance for convergence of iterative algorithm

    h0 : non-negative k x ncols array, optional
        Value at which H is initialized

    wHist : Bool, optional, default = False
        If true, keep track of convergence of w at each iteration

    reconHist : str in {'none', 'final', 'all'}
        if 'none', reconstruction error is never computed. if 'all', it is computed at every iteration.
        if 'final', reconstruction error is computed on the final solution.

    verbose : boolean, optional, default = False
        Whether to print progress

    Attributes
    ----------
    `w` : RDD of nrows (tuple, array) pairs, each array of shape (k,)
        Left bases

    `h` : array, shape (k, ncols)
        Right bases

    'hConvergence` : list of floats
        List of Frobenius norms between successive estimates of h

    `wConvergence` : None or list of floats
        If w_hist==True, a list of Frobenius norms between successive estimates of w

    `reconErr` : None, int, or list
        Output of the reconstruction error at the iterations specified by parameter recon_hist
    """

    def __init__(self, k=5, method='als', maxIter=20, tol=0.001, h0=None, wHist=False,
                 reconHist='none', verbose=False):

        # initialize input variables
        self.k = int(k)
        self.method = method
        self.maxIter = maxIter
        self.tol = tol
        self.h0 = h0
        self.reconHist = reconHist
        self.verbose = verbose

        # initialize output variables
        self.h = None
        self.w = None
        self.hConvergence = list()

        if wHist is True:
            self.wConvergence = list()
        else:
            self.wConvergence = None

        if reconHist == 'all':
            self.reconErr = list()
        else:
            self.reconErr = None

    def fit(self, mat):
        """
        Calcuate the non-negative matrix decomposition.

        Parameters
        ----------
        mat : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        ----------
        self : returns an instance of self.
        """

        from numpy import add, any, diag, dot, inf, maximum, outer, sqrt, apply_along_axis
        from numpy.linalg import inv, norm, pinv
        from numpy.random import rand

        if not (isinstance(mat, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        mat = mat.rdd

        # a helper function to take the Frobenius norm of two zippable RDDs
        def rddFrobeniusNorm(A, B):
            return sqrt(A.zip(B).map(lambda ((keyA, x), (keyB, y)): sum((x - y) ** 2)).reduce(add))

        # input checking
        k = self.k
        if k < 1:
            raise ValueError("Supplied k must be greater than 1.")
        m = mat.values().first().size
        if self.h0 is not None:
            if any(self.h0 < 0):
                raise ValueError("Supplied h0 contains negative entries.")

        # alternating least-squares implementation
        if self.method == "als":

            # initialize NMF and begin als algorithm
            if self.verbose:
                print "Initializing NMF"
            alsIter = 0
            hConvCurr = 100

            if self.h0 is None:
                # noinspection PyUnresolvedReferences
                self.h0 = rand(k, m)

            h = self.h0
            w = None

            # goal is to solve R = WH subject to all entries of W,H >= 0
            # by iteratively updating W and H with least squares and clipping negative values
            while (alsIter < self.maxIter) and (hConvCurr > self.tol):
                # update values on iteration
                hOld = h
                wOld = w

                # precompute pinv(H) = inv(H' x H) * H' (easy here because h is an np array)
                # the rows of H should be a basis of dimension k, so in principle we could just compute directly
                pinvH = pinv(h)

                # update W using least squares row-wise with R * pinv(H); then clip negative values to 0
                w = mat.mapValues(lambda x: dot(x, pinvH))

                # clip negative values of W
                # noinspection PyUnresolvedReferences
                w = w.mapValues(lambda x: maximum(x, 0))

                # precompute inv(W' * W) to get inv_gramian_w, a np array
                # We have chosen k to be small, i.e., rank(W) = k, so W'*W is invertible
                gramianW = w.values().map(lambda x: outer(x, x)).reduce(add)
                invGramianW = inv(gramianW)

                # pseudoinverse of W is inv(W' * W) * W' = inv_gramian_w * w
                pinvW = w.mapValues(lambda x: dot(invGramianW, x))

                # update H using least squares row-wise with inv(W' * W) * W * R (same as pinv(W) * R)
                h = pinvW.values().zip(mat.values()).map(lambda (x, y): outer(x, y)).reduce(add)

                # clip negative values of H
                # noinspection PyUnresolvedReferences
                h = maximum(h, 0)

                # normalize the rows of H
                # noinspection PyUnresolvedReferences
                h = dot(diag(1 / maximum(apply_along_axis(norm, 1, h), 0.001)), h)

                # estimate convergence
                hConvCurr = norm(h-hOld)
                self.hConvergence.append(hConvCurr)
                if self.wConvergence is not None:
                    if wOld is not None:
                        self.wConvergence.append(rddFrobeniusNorm(w, wOld))
                    else:
                        self.wConvergence.append(inf)

                # calculate reconstruction error
                if self.reconHist == 'all':
                    recData = w.mapValues(lambda x: dot(x, h))
                    self.reconErr.append(rddFrobeniusNorm(mat, recData))

                # report progress
                if self.verbose:
                    print "finished als iteration %d with convergence = %.6f in H" % (alsIter, hConvCurr)

                # increment count
                alsIter += 1

            # report on convergence
            if self.verbose:
                if hConvCurr <= self.tol:
                    print "Converged to specified tolerance."
                else:
                    print "Warning: reached maxiter without converging to specified tolerance."

            # calculate reconstruction error
            if self.reconHist == 'final':
                    recData = w.mapValues(lambda x: dot(x, h))
                    self.reconErr = rddFrobeniusNorm(mat, recData)

            # report results
            self.h = h
            # TODO: need to propagate metadata through to this new Series object
            self.w = Series(w)

        else:
            raise Exception("Algorithm %s is not supported" % self.method)

        return self

class SparseNMF(NMF):
    """
        Extends NMF but with sparsification according to the
        "Sparse non-negative matrix factorizations
        via alternating non-negativity-constrained least squares for microarray data analysis"
        by Kim and Park, 2012.
        Note that their formulation is to sparsify H, but we changed it to sparsify W

        Beta is by default .1
        Eta should be by default the maximum value of the original matrix squared
    """

    def __init__(self, k=5, method='als', maxIter=20, tol=0.001, h0=None, wHist=False,
                 reconHist='none',beta=1.,eta=1, verbose=False):

        super(SparseNMF, self).__init__(k, method, maxIter, tol, h0, wHist,
                 reconHist, verbose)
        self.beta = beta
        self.eta = eta

    def fit(self, mat):
        """
        Calcuate the non-negative matrix decomposition.

        Parameters
        ----------
        mat : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        ----------
        self : returns an instance of self.
        """

        from numpy import add, any, diag, dot, inf, maximum, outer, sqrt, apply_along_axis
        from numpy import vstack,hstack,eye,zeros, ones
        from numpy.linalg import inv, norm, pinv, cond
        from numpy.random import rand

        if not (isinstance(mat, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        mat = mat.rdd

        # a helper function to take the Frobenius norm of two zippable RDDs
        def rddFrobeniusNorm(A, B):
            return sqrt(A.zip(B).map(lambda ((keyA, x), (keyB, y)): sum((x - y) ** 2)).reduce(add))

        # input checking
        k = self.k
        if k < 1:
            raise ValueError("Supplied k must be greater than 1.")
        m = mat.values().first().size
        if self.h0 is not None:
            if any(self.h0 < 0):
                raise ValueError("Supplied h0 contains negative entries.")

        # alternating least-squares implementation
        if self.method == "als":

            # initialize NMF and begin als algorithm
            if self.verbose:
                print "Initializing NMF"
            alsIter = 0
            hConvCurr = 100

            if self.h0 is None:
                # noinspection PyUnresolvedReferences
                print "Intializing h0 to be random, scaled by .1*sqrt(eta)"
                self.h0 = rand(k, m)*sqrt(self.eta)/10

            h = self.h0
            w = None

            # goal is to solve R = WH subject to all entries of W,H >= 0
            # by iteratively updating W and H with least squares and clipping negative values
            while (alsIter < self.maxIter) and (hConvCurr > self.tol):
                # update values on iteration
                print "starting"
                hOld = h
                wOld = w

                #Calculate the pseudoinverse with the beta weight appended
                pinvH = pinv(vstack((h.T,sqrt(self.beta)*ones((1, self.k)))))
                #Truncate the last colmn of the pinv before multiplying with mat
                #See paper in the class comments for details
                pinvH_trunc = pinvH[:, :-1]
                # update W using least squares row-wise with R * pinv(H); then clip negative values to 0
                w = mat.mapValues(lambda x: dot(x, pinvH_trunc.T))

                # clip negative values of W
                # noinspection PyUnresolvedReferences
                w = w.mapValues(lambda x: maximum(x, 0))

                # precompute inv(W' * W) to get inv_gramian_w, a np array
                # We have chosen k to be small, i.e., rank(W) = k, so W'*W is invertible
                gramianW = w.values().map(lambda x: outer(x, x)).reduce(add)
                mod_gramianW = gramianW + self.eta*eye(self.k)
                invGramianW = inv(mod_gramianW)

                # pseudoinverse of W is inv(W' * W) * W' = inv_gramian_w * w
                pinvW = w.mapValues(lambda x: dot(invGramianW, x))

                # update H using least squares row-wise with inv(W' * W) * W * R (same as pinv(W) * R)
                h = pinvW.values().zip(mat.values()).map(lambda (x, y): outer(x, y)).reduce(add)

                # clip negative values of H
                # noinspection PyUnresolvedReferences
                h = maximum(h, 0)

                # Because of the regularization there is no need for the normalization step
                # h = dot(diag(1 / maximum(apply_along_axis(norm, 1, h), 0.001)), h)

                # estimate convergence
                hConvCurr = norm(h-hOld)
                self.hConvergence.append(hConvCurr)
                if self.wConvergence is not None:
                    if wOld is not None:
                        self.wConvergence.append(rddFrobeniusNorm(w, wOld))
                    else:
                        self.wConvergence.append(inf)

                # calculate reconstruction error
                if self.reconHist == 'all':
                    recData = w.mapValues(lambda x: dot(x, h))
                    self.reconErr.append(rddFrobeniusNorm(mat, recData))

                # report progress
                if self.verbose:
                    print "finished als iteration %d with convergence = %.6f in H" % (alsIter, hConvCurr)

                # increment count
                alsIter += 1

            # report on convergence
            if self.verbose:
                if hConvCurr <= self.tol:
                    print "Converged to specified tolerance."
                else:
                    print "Warning: reached maxiter without converging to specified tolerance."

            # calculate reconstruction error
            if self.reconHist == 'final':
                recData = w.mapValues(lambda x: dot(x, h))
                self.reconErr = rddFrobeniusNorm(mat, recData)

            # report results
            self.h = h
            # TODO: need to propagate metadata through to this new Series object
            self.w = Series(w)

        else:
            raise Exception("Algorithm %s is not supported" % self.method)

        return self
