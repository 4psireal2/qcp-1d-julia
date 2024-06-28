using LinearAlgebra
using TensorKit


function expHam(omega::Float64, tau::Float64)
    sigmaX = 0.5 * [0 +1 ; +1 0];
    numberOp = [0 0; 0 1];
    ham = omega * (kron(sigmaX,numberOp) + kron(numberOp, sigmaX));


    propagator = exp(-1im * tau * ham);
    expHamOp = TensorMap(propagator, ℂ^2 ⊗ ℂ^2,  ℂ^2 ⊗ ℂ^2);

    return expHamOp
end


function expDiss(gamma::Float64, tau::Float64)
    annihilationOp = [0 1; 0 0];
    numberOp = [0 0; 0 1];
    Id = [+1 0 ; 0 +1];
    numberOpR = kron(numberOp, Id);
    numberOpL = kron(Id, numberOp);

    diss = gamma * kron(annihilationOp, annihilationOp) - (1/2) * numberOpR - (1/2) * numberOpL;
    diss = TensorMap(diss, ComplexSpace(2) ⊗ ComplexSpace(2)', ComplexSpace(2) ⊗ ComplexSpace(2)');
    diss = exp(tau * diss);

    # EVD    
    D, V = eig(diss, (1, 3), (2, 4));
    B = permute(sqrt(D) * V', (3, 1), (2, ));

    return B

end


function TEBD(X, uniOp, krausOp, bondDim, krausDim, truncErr=1e-6)
    """
    2nd order TEBD with dissipative layer
    """


    # sweep L ---> R [odd]
    for i = 1 : 2 : N-1
        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];

        # shift orthogonality center to right
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        X[i] = permute(U, (1, 2), (3, 4));
        X[i+1]  = permute(S * V, (1, 2), (3, 4));

        if i < N - 2
            Q, R = leftorth(X[i+1], (1, 2, 3), (4, ), alg = QRpos());
            X[i + 1] = permute(Q, (1, 2), (3, 4)) ;
            X[i + 2] = permute(R * permute(X[i + 2], (1, ), (2, 3, 4)), (1, 2), (3, 4));
        end
    end


    # sweep R ---> L [even]
    for i = reverse(2 : 2 : N-1)

        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];        

        # shift orthogonality center to left
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        X[i+1] = permute(V, (1, 2), (3, 4));
        X[i] = permute(U * S, (1, 2), (3, 4));

        L, Q = rightorth(X[i], (1, ), (2, 3, 4), alg = LQpos());
        X[i - 1] = permute(permute(X[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
        X[i] = permute(Q, (1, 2), (3, 4));
    end

    # dissipation
    for i = 1 : N
        @tensor Bx[-1 -2 -3; -4 -5] := krausOp[1, -2, -4] * X[i][-1, -3, 1, -5];
        U, S, V, ϵ = tsvd(Bx, (2, 3), (1, 4, 5), trunc = truncdim(krausDim) & truncerr(truncErr), alg = TensorKit.SVD());
        X[i] = permute(S * V, (2, 1), (3, 4));
        
        # shift orthogonality center to right
        if i < N
            Q, R = leftorth(X[i], (1, 2, 3), (4, ), alg = QRpos());
            X[i] = permute(Q, (1, 2), (3, 4)) ;
            X[i + 1] = permute(R * permute(X[i + 1], (1, ), (2, 3, 4)), (1, 2), (3, 4))
        end
    end
    
    # OC at the end (for chain of even length)
    if N%2 == 0
        L, Q = rightorth(X[N], (1, ), (2, 3, 4), alg = LQpos());
        X[N - 1] = permute(permute(X[N-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
        X[N] = permute(Q, (1, 2), (3, 4));
    end
    
    # sweep R ---> L [even]
    for i = reverse(2 : 2 : N-1)

        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];
        
        # shift orthogonality center to left
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        X[i+1] = permute(V, (1, 2), (3, 4));
        X[i] = permute(U * S, (1, 2), (3, 4));

        L, Q = rightorth(X[i], (1, ), (2, 3, 4), alg = LQpos());
        X[i - 1] = permute(permute(X[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
        X[i] = permute(Q, (1, 2), (3, 4));
    end

    # sweep L ---> R [odd]
    for i = 1 : 2 : N-1
        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];

        # shift orthogonality center to right
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        X[i] = permute(U, (1, 2), (3, 4));
        X[i+1]  = permute(S * V, (1, 2), (3, 4));

        if i < N - 2 
            Q, R = leftorth(X[i+1], (1, 2, 3), (4, ), alg = QRpos());
            X[i + 1] = permute(Q, (1, 2), (3, 4)) ;
            X[i + 2] = permute(R * permute(X[i + 2], (1, ), (2, 3, 4)), (1, 2), (3, 4));
        end
    end

    X = orthogonalizeX(X, 1);
    return X
end


