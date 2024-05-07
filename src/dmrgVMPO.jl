# module dmrgVMPO
include("utility.jl")

using LinearAlgebra
using Printf
using KrylovKit
using MPSKit
using TensorKit


function updateMPOEnvL(mpoEnvL::TensorMap, mpsT::TensorMap, mpo::TensorMap, mpsB::TensorMap)::TensorMap
    @tensor newEnvL[-1; -2 -3] := mpoEnvL[1, 7, 4] * mpsT[7, 5, 6, -2] * mpo[4, 2, 3, 5, 6, -3] * conj(mpsB[1, 2, 3, -1]);
    
    return newEnvL;
end


function updateMPOEnvR(mpoEnvR::TensorMap, mpsT::TensorMap, mpo::TensorMap, mpsB::TensorMap)::TensorMap
    @tensor newEnvR[-1 -2; -3] := mpsT[-1, 2, 3, 1] * mpo[-2, 5, 6, 2, 3, 4] * conj(mpsB[-3, 5, 6, 7]) * mpoEnvR[1, 4, 7];
    
    return newEnvR;
end


function applyH1(X::TensorMap, EnvL::TensorMap, mpo::TensorMap, EnvR::TensorMap)::TensorMap
    # @tensor X[-1 -2 -3; -4 -5 -6] := EnvL[-1, 2, 1] * X[2, 3, 4, 6, 7, 8] * mpo1[1, -2, -3, 3, 4, 5] * mpo2[5, -4, -5, 6, 7, 9] * EnvR[8, 9, -6];
    @tensor X[-1 -2 -3; -4] := EnvL[-1, 2, 1] * X[2, 3, 4, 5] * mpo[1, -2, -3, 3, 4, 6] * EnvR[5, 6, -4];
    return X;
end


function applyH2(X::TensorMap, EnvL::TensorMap, mpo1::TensorMap, mpo2::TensorMap, EnvR::TensorMap)::TensorMap
    @tensor X[-1 -2 -3; -4 -5 -6] := EnvL[-1, 2, 1] * X[2, 3, 4, 6, 7, 8] * mpo1[1, -2, -3, 3, 4, 5] * mpo2[5, -4, -5, 6, 7, 9] * EnvR[8, 9, -6];
    
    return X;
end


function initializeMPOEnvs(mps::Vector{TensorMap}, mpo::Vector{TensorMap}; centerPos::Int64=1)
    """
    Create left and right environments for DMRG2
    """
    length(mps) == length(mpo) || throw(ArgumentError("dimension mismatch"))

    N = length(mps);

    mpoEnvL = Vector{TensorMap}(undef, N);
    mpoEnvR = Vector{TensorMap}(undef, N);

    mpoEnvL[1] = TensorMap(ones, space(mps[1], 1), space(mps[1], 1) ⊗ space(mpo[1], 1));
    mpoEnvR[N] = TensorMap(ones, space(mps[N], 4)' ⊗ space(mpo[N], 6)', space(mps[N], 4)');

    # compute mpoEnvL up to (centerPos - 1)
    for i = 1 : 1 : (centerPos - 1)
        mpoEnvL[i + 1] = updateMPOEnvL(mpoEnvL[i], mps[i], mpo[i], mps[i]);
    end

    # compute mpoEnvR up to (centerPos + 1)
    for i = N : -1 : (centerPos + 1)
        mpoEnvR[i - 1] = updateMPOEnvR(mpoEnvR[i], mps[i], mpo[i], mps[i]);
    end

    return mpoEnvL, mpoEnvR;
end


function DMRG1(mps::Vector{TensorMap}, mpo::Vector{TensorMap}; convTolE::Float64 = 1e-6,
    eigsTol::Float64 = 1e-16, maxIterations::Int64 = 1, verbosePrint::Bool = false)
    """
    Use computeExpVal for expectation values
    """
    N = length(mps);

    initialEnergy = (1 / N) * sum(computeExpVal(mps, mpo));
    mpsEnergy = Float64[initialEnergy];
    
    mpoEnvL, mpoEnvR = initializeMPOEnvs(mps, mpo);

    loopCounter = 1;
    runOptimizationDMRG = true;
    while runOptimizationDMRG

        # sweep L ---> R

        for i = 1 : +1 : N

            # optimize wave function to get newTensor
            _, eigenVec = 
                eigsolve(mps[i], 1, :SR, KrylovKit.Lanczos(tol = eigsTol, maxiter = maxIterations)) do x
                    applyH1(x, mpoEnvL[i], mpo[i], mpoEnvR[i])
                end

            newTensor = eigenVec[1];

            # shift orthogonality center to the right and update MPO environments
            if i < N

                # left-orthogonalize newTensor
                (Q, R) = leftorth(newTensor, (1, 2, 3), (4, ), alg = QRpos())
                R /= norm(R);
                mps[i] = permute(Q, (1, 2, 3), (4, ));

                # absorb R into next MPS site
                mps[i + 1] = permute(R * permute(mps[i + 1], (1, ), (2, 3, 4)), (1, 2, 3), (4, ));

                # update mpoEnvL
                mpoEnvL[i + 1] = updateMPOEnvL(mpoEnvL[i], mps[i], mpo[i], mps[i]);
            else
                mps[i] = newTensor;
            end
        end

        # sweep L <--- R
        for i = N : -1 : 1

            # optimize wave function to get newTensor
            _, eigenVec = 
                eigsolve(mps[i], 1, :SR, KrylovKit.Lanczos(tol = eigsTol, maxiter = maxIterations)) do x
                    applyH1(x, mpoEnvL[i], mpo[i], mpoEnvR[i])
                end
            newTensor = eigenVec[1];

            # shift orthogonality center to the left and update MPO environments
            if i > 1

                # right-orthogonalize newTensor
                (L, Q) = rightorth(newTensor, (1, ), (2, 3, 4), alg = LQpos());
                L /= norm(L);
                mps[i] = permute(Q, (1, 2, 3), (4, ));

                # absorb L into previous MPS site
                mps[i - 1] = permute(mps[i - 1] * L, (1, 2, 3), (4, ));

                # update mpoEnvR
                mpoEnvR[i - 1] = updateMPOEnvR(mpoEnvR[i], mps[i], mpo[i], mps[i]);

            else
                mps[i] = newTensor;
            end
        end

        # normalize 
        mpsNorm = tr(mps[1]' * mps[1]);
        mps[1] /= sqrt(mpsNorm)


        # compute expectation value
        expVal = (1 / N) * sum(computeExpVal(mps, mpo));
        if abs(imag(expVal)) < 1e-12
            expVal = real(expVal);
        else
            ErrorException("The Hamiltonian is not Hermitian, complex eigenvalue found.")
        end
        mpsEnergy = vcat(mpsEnergy, expVal)

        # check convergence of ground state energy
        energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
        verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, expVal, energyConvergence);
        if energyConvergence < convTolE
            runOptimizationDMRG = false;
        end

        loopCounter += 1;
    end

    # return optimized mps
    finalEnergy = mpsEnergy[end];
    return mps, finalEnergy;
end


function DMRG2(mps::Vector{TensorMap}, mpo::Vector{TensorMap};
               bondDim::Int64 = 20, truncErr::Float64 = 1e-6, convTolE::Float64 = 1e-6,
               eigsTol::Float64 = 1e-16, maxIterations::Int64 = 1, verbosePrint::Bool = false)
    
    N = length(mps);

    initialEnergy = computeExpVal(mps, mpo);
    mpsEnergy = Float64[initialEnergy];
    
    mpoEnvL, mpoEnvR = initializeMPOEnvs(mps, mpo);

    # main DMRG loop
    loopCounter = 1;
    runOptimizationDMRG = true;
    while runOptimizationDMRG
        
        # sweep L ---> R
        for i = 1 : 1 : (N - 1)

            # construct intial bond tensor
            bondTensor = mps[i] * permute(mps[i+1], (1, ), (2, 3, 4));

            # optimize wave function to get newTensor
            _, eigenVec = eigsolve(bondTensor, 1, :SR, Lanczos(tol = eigsTol, maxiter = maxIterations)) do x # 1 eigenVal
                applyH2(x, mpoEnvL[i], mpo[i], mpo[i + 1], mpoEnvR[i + 1])
            end

            newBondTensor = eigenVec[1];

            #  perform SVD and truncate to desired bond dimension
            U, S, V, ϵ = tsvd(newBondTensor, (1, 2, 3), (4, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
            U = permute(U, (1, 2, 3), (4, ));
            V = permute(S * V, (1, 2, 3), (4, ));

            mps[i], mps[i+1] = U, V;

            # update mpoEnvL
            mpoEnvL[i + 1] = updateMPOEnvL(mpoEnvL[i], mps[i], mpo[i], mps[i]);
        end

        # sweep R --> L
        for i = (N-1) : -1 : 1
            # construct intial bond tensor
            bondTensor = mps[i] * permute(mps[i+1], (1, ), (2, 3, 4));

            # optimize wave function to get newTensor
            _, eigenVec = eigsolve(bondTensor, 1, :SR, Lanczos(tol = eigsTol, maxiter = maxIterations)) do x # 1 eigenVal
                applyH2(x, mpoEnvL[i], mpo[i], mpo[i + 1], mpoEnvR[i + 1])
            end

            newBondTensor = eigenVec[1];

            #  perform SVD and truncate to desired bond dimension
            U, S, V, ϵ = tsvd(newBondTensor, (1, 2, 3), (4, 5, 6), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
            U = permute(U * S, (1, 2, 3), (4, ));
            V = permute(V, (1, 2, 3), (4, ));

            # update mpoEnvR
            mpoEnvR[i] = updateMPOEnvR(mpoEnvR[i+1], mps[i+1], mpo[i+1], mps[i+1]);
        end

        # normalize 
        mpsNorm = tr(mps[1]' * mps[1]);
        mps[1] /= sqrt(mpsNorm)

        # compute expectation value
        expVal = computeExpVal(mps, mpo);
        if abs(imag(expVal)) < 1e-12
            expVal = real(expVal);
        else
            ErrorException("The Hamiltonian is not Hermitian, complex eigenvalue found.")
        end
        mpsEnergy = vcat(mpsEnergy, expVal)

        # check convergence of ground state energy
        energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
        verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, expVal, energyConvergence);
        if energyConvergence < convTolE
            runOptimizationDMRG = false;
        end

        loopCounter += 1;
    end

    # return optimized mps
    finalEnergy = mpsEnergy[end];
    return mps, finalEnergy;
end