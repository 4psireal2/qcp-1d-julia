using LinearAlgebra
using TensorKit


function expHam(omega, tau)
    sigmaX = 0.5 * [0 +1 ; +1 0];
    numberOp = [0 0; 0 1];
    ham = omega * (kron(sigmaX,numberOp) + kron(numberOp, sigmaX));

    propagator = exp(-1im * tau * ham);
    expHamOp = TensorMap(propagator, ℂ^2 ⊗ ℂ^2,  ℂ^2 ⊗ ℂ^2);

    return expHamOp
end


# function expHam(omega, tau)
#     sigmaX = 0.5 * [0 +1 ; +1 0];
#     numberOp = [0 0; 0 1];
#     Id = [1 0; 0 1];
#     ham = sigmaX * numberOp + numberOp * sigmaX; 
#     ham = omega * (kron(ham, Id)  - kron(Id, ham));

#     propagator = exp(-1im * tau * ham);
#     expHamOp = TensorMap(propagator, ℂ^2 ⊗ ℂ^2,  ℂ^2 ⊗ ℂ^2);

#     return expHamOp
# end


function expDiss(gamma, tau)
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


function applyGate(Xa, Xb, oP, bondDim, truncErr)
    """
    Apply a 2-site gate on Xa and Xb
    """
    QR, R = leftorth(Xa, (1, 2,), (3, 4, ), alg = QRpos());
    L, QL = rightorth(Xb, (1, 3, ), (2, 4), alg = LQpos());
    @tensor bondTensor[-1; -2 -3 -4] := R[-1, 1, 2] * oP[1, 3, -2, -3] * L[2, 3, -4];
    bondTensor /= norm(bondTensor);

    U, S, V, ϵ = tsvd(bondTensor, (1, 2, ), (3, 4, ), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
    S /= norm(S); # to normalise truncated bondTensor

    @tensor U[-1, -2; -3, -4] := QR[-1, -2, 1] * U[1, -3, -4];
    @tensor V[-1, -2; -3, -4] := V[-1, -3, 1] * QL[1, -2, -4];

    return U, S, V, ϵ

end


function TEBD(X, uniOp, krausOp, bondDim, krausDim; truncErr=1e-6, canForm=true)
    """
    2nd order TEBD with dissipative layer for one time step

    Returns:
        X_t: left-canonical MPO
        ϵHTrunc: truncation errors in bond dimension
        ϵDTrunc: truncation errors in Kraus dimension
    """

    ϵHTrunc = Vector{Float64}();
    ϵDTrunc = Vector{Float64}();
    N = length(X);

    # sweep L ---> R [odd]
    for i = 1 : 2 : N-1

        U, S, V, ϵ = applyGate(X[i], X[i+1], uniOp, bondDim, truncErr);
        push!(ϵHTrunc, ϵ);

        if canForm
            # shift orthogonality center to right
            if i == N-1 && N%2==0 # OC on the left for the last bond tensor of chain of even length 
                X[i] = permute(permute(U, (1, 2, 3), (4, )) * S, (1, 2), (3, 4));
                X[i+1] = permute(V, (1, 2), (3, 4));
            else
                X[i] = permute(U, (1, 2), (3, 4));
                X[i+1] = permute(S * permute(V, (1, ), (2, 3, 4)), (1, 2), (3, 4))
            end

            if (i < N - 1 && N%2 == 1) || (i < N - 2 && N%2 == 0)
                Q, R = leftorth(X[i+1], (1, 2, 3), (4, ), alg = QRpos());
                X[i + 1] = permute(Q, (1, 2), (3, 4)) ;
                X[i + 2] = permute(R * permute(X[i + 2], (1, ), (2, 3, 4)), (1, 2), (3, 4));
            end
        else
            X[i] = permute(permute(U, (1, 2, 3), (4, )) * sqrt(S), (1, 2), (3, 4));
            X[i+1] = permute(sqrt(S) * permute(V, (1, ), (2, 3, 4)), (1, 2), (3, 4))
        end
    end

    # sweep R ---> L [even]
    for i = reverse(2 : 2 : N-1)

        U, S, V, ϵ = applyGate(X[i], X[i+1], uniOp, bondDim, truncErr);
        push!(ϵHTrunc, ϵ);

        # shift orthogonality center to left
        X[i+1] = permute(V, (1, 2), (3, 4));
        X[i] = permute(permute(U, (1, 2, 3), (4, )) * S, (1, 2), (3, 4));

        if canForm
            L, Q = rightorth(X[i], (1, ), (2, 3, 4), alg = LQpos());
            X[i - 1] = permute(permute(X[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
            X[i] = permute(Q, (1, 2), (3, 4));
        end
    end

    # dissipation
    for i = 1 : N
        @tensor Bx[-1 -2 -3; -4 -5] := krausOp[1, -2, -4] * X[i][-1, -3, 1, -5];
        Bx /= norm(Bx);

        U, S, V, ϵ = tsvd(Bx, (2, 3), (1, 4, 5), trunc = truncdim(krausDim) & truncerr(truncErr), alg = TensorKit.SVD());
        S /= norm(S); # normalise truncated bondTensor

        push!(ϵDTrunc, ϵ);
        X[i] = permute(S * V, (2, 1), (3, 4));
        
        # shift orthogonality center to right
        if canForm
            if i < N
                Q, R = leftorth(X[i], (1, 2, 3), (4, ), alg = QRpos());
                X[i] = permute(Q, (1, 2), (3, 4)) ;
                X[i + 1] = permute(R * permute(X[i + 1], (1, ), (2, 3, 4)), (1, 2), (3, 4))
            end
        end
    end
    
    # OC at the end for chain of even length => move OC to left
    if canForm
        if N%2 == 0
            L, Q = rightorth(X[N], (1, ), (2, 3, 4), alg = LQpos());
            X[N - 1] = permute(permute(X[N-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
            X[N] = permute(Q, (1, 2), (3, 4));
        end
    end
    
    # sweep R ---> L [even]
    for i = reverse(2 : 2 : N-1)

        U, S, V, ϵ = applyGate(X[i], X[i+1], uniOp, bondDim, truncErr);
        push!(ϵHTrunc, ϵ);

        # shift orthogonality center to left
        X[i+1] = permute(V, (1, 2), (3, 4));
        X[i] = permute(permute(U, (1, 2, 3), (4, )) * S, (1, 2), (3, 4));

        if canForm
            L, Q = rightorth(X[i], (1, ), (2, 3, 4), alg = LQpos());
            X[i - 1] = permute(permute(X[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
            X[i] = permute(Q, (1, 2), (3, 4));
        end
    end

    # sweep L ---> R [odd]
    for i = 1 : 2 : N-1
        U, S, V, ϵ = applyGate(X[i], X[i+1], uniOp, bondDim, truncErr);
        push!(ϵHTrunc, ϵ);

        X[i] = permute(U, (1, 2), (3, 4));
        X[i+1] = permute(S * permute(V, (1, ), (2, 3, 4)), (1, 2), (3, 4))

        if canForm
            if (i < N - 1 && N%2 == 1) || (i < N - 2 && N%2 == 0)
                Q, R = leftorth(X[i+1], (1, 2, 3), (4, ), alg = QRpos());
                X[i + 1] = permute(Q, (1, 2), (3, 4)) ;
                X[i + 2] = permute(R * permute(X[i + 2], (1, ), (2, 3, 4)), (1, 2), (3, 4));
            end
        end
    end

    X = orthonormalizeX(X, orthoCenter=1);
    return X, ϵHTrunc, ϵDTrunc
end


function TEBD_test(X, uniOp, krausOp, bondDim, krausDim, truncErr=1e-6, canForm=true)
    """
    without QR decomposition
    """

    ϵHTrunc = Vector{Float64}();
    ϵDTrunc = Vector{Float64}();
    N = length(X);

    # sweep L ---> R [odd]
    for i = 1 : 2 : N-1

        ## QR/LQ decomposition test
        ### without splitting the Kraus legs
        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];

        ### with splitting ...
        QR, R = leftorth(X[i], (1, 2,), (3, 4, ), alg = QRpos());
        L, QL = rightorth(X[i+1], (1, 3, ), (2, 4), alg = LQpos());
        @tensor bondTensor_test[-1; -2 -3 -4] := R[-1, 1, 2] * uniOp[1, 3, -2, -3] * L[2, 3, -4];
        @tensor bondTensor_test[-1 -2 -3; -4 -5 -6] := QR[-1, -2, 1] * bondTensor_test[1, -4, -5, 2] * QL[2, -3, -6];

        @show bondTensor == bondTensor_test # true 
        bondTensor /= norm(bondTensor);

        # shift orthogonality center to right
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        S /= norm(S); # normalise truncated bondTensor
        push!(ϵHTrunc, ϵ);

        if canForm
            if i == N-1 && N%2==0 # OC on the left for the last bond tensor of chain of even length 
                X[i] = permute(U * S, (1, 2), (3, 4));
                X[i+1] = permute(V, (1, 2), (3, 4));
            else
                X[i] = permute(U, (1, 2), (3, 4));
                X[i+1]  = permute(S * V, (1, 2), (3, 4));
            end

                if (i < N - 1 && N%2 == 1) || (i < N - 2 && N%2 == 0)
                Q, R = leftorth(X[i+1], (1, 2, 3), (4, ), alg = QRpos());
                X[i + 1] = permute(Q, (1, 2), (3, 4)) ;
                X[i + 2] = permute(R * permute(X[i + 2], (1, ), (2, 3, 4)), (1, 2), (3, 4));
                end
        else
            X[i] = permute(U * sqrt(S), (1, 2), (3, 4));
            X[i+1] = permute(sqrt(S) * V, (1, 2), (3, 4));
        end
    end


    # sweep R ---> L [even]
    for i = reverse(2 : 2 : N-1)

        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];        
        bondTensor /= norm(bondTensor);

        # shift orthogonality center to left
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        S /= norm(S); # normalise truncated bondTensor
        push!(ϵHTrunc, ϵ);

        # shift orthogonality center to left
        X[i+1] = permute(V, (1, 2), (3, 4));
        X[i] = permute(U * S, (1, 2), (3, 4));

        if canForm
            L, Q = rightorth(X[i], (1, ), (2, 3, 4), alg = LQpos());
            X[i - 1] = permute(permute(X[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
            X[i] = permute(Q, (1, 2), (3, 4));
        end
    end

    # dissipation
    for i = 1 : N
        @tensor Bx[-1 -2 -3; -4 -5] := krausOp[1, -2, -4] * X[i][-1, -3, 1, -5];
        Bx /= norm(Bx);

        U, S, V, ϵ = tsvd(Bx, (2, 3), (1, 4, 5), trunc = truncdim(krausDim) & truncerr(truncErr), alg = TensorKit.SVD());
        S /= norm(S); # normalise truncated bondTensor

        push!(ϵDTrunc, ϵ);
        X[i] = permute(S * V, (2, 1), (3, 4));
        
        # shift orthogonality center to right
        if canForm
            if i < N
                Q, R = leftorth(X[i], (1, 2, 3), (4, ), alg = QRpos());
                X[i] = permute(Q, (1, 2), (3, 4)) ;
                X[i + 1] = permute(R * permute(X[i + 1], (1, ), (2, 3, 4)), (1, 2), (3, 4))
            end
        end
    end
    
    # OC at the end for chain of even length => move OC to left
    if canForm
        if N%2 == 0
            L, Q = rightorth(X[N], (1, ), (2, 3, 4), alg = LQpos());
            X[N - 1] = permute(permute(X[N-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
            X[N] = permute(Q, (1, 2), (3, 4));
        end
    end
    
    # sweep R ---> L [even]
    for i = reverse(2 : 2 : N-1)

        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];
        bondTensor /= norm(bondTensor);
        
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        S /= norm(S); # normalise truncated bondTensor
        push!(ϵHTrunc, ϵ);


        # shift orthogonality center to left
        X[i+1] = permute(V, (1, 2), (3, 4));
        X[i] = permute(U * S, (1, 2), (3, 4));

        if canForm
            L, Q = rightorth(X[i], (1, ), (2, 3, 4), alg = LQpos());
            X[i - 1] = permute(permute(X[i-1], (1, 2, 3), (4, )) * L, (1, 2), (3, 4));
            X[i] = permute(Q, (1, 2), (3, 4));
        end
    end

    # sweep L ---> R [odd]
    for i = 1 : 2 : N-1
        @tensor bondTensor[-1, -2, -3; -4, -5, -6] := uniOp[1, 2, -4, -5] * X[i][-1, -2, 1, 3]  * X[i+1][3, -3, 2, -6];
        bondTensor /= norm(bondTensor);

        # shift orthogonality center to right
        U, S, V, ϵ = tsvd(bondTensor, (1, 2, 4), (3, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
        S /= norm(S); # normalise truncated bondTensor
        push!(ϵHTrunc, ϵ);

        X[i] = permute(U, (1, 2), (3, 4));
        X[i+1]  = permute(S * V, (1, 2), (3, 4));

        if canForm
            if (i < N - 1 && N%2 == 1) || (i < N - 2 && N%2 == 0)
                Q, R = leftorth(X[i+1], (1, 2, 3), (4, ), alg = QRpos());
                X[i + 1] = permute(Q, (1, 2), (3, 4)) ;
                X[i + 2] = permute(R * permute(X[i + 2], (1, ), (2, 3, 4)), (1, 2), (3, 4));
            end
        end
    end

    X = orthogonalizeX(X, orthoCenter=1);
    return X, ϵHTrunc, ϵDTrunc
end