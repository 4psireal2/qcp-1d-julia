#!/usr/bin/env julia

# clear console
Base.run(`clear`)

# load packages
import LinearAlgebra
using LinearAlgebra: norm, dot, tr, diag, diagm

# using Plots
using KrylovKit
using Printf
using TensorKit


let

    function initializeRandomMPS(N; d::Int64 = 2, bondDim::Int64 = 1)
        """ Returns random MPS for N sites """
    
        # initialize randomMPS
        randomMPS = Vector{TensorMap}(undef, N);
        
        # left MPO boundary
        randomMPS[1] = TensorMap(randn, ComplexSpace(1) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
        for idxMPO = 2 : (N - 1)
            randomMPS[idxMPO] = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(bondDim));
        end
        randomMPS[N] = TensorMap(randn, ComplexSpace(bondDim) ⊗ ComplexSpace(d), ComplexSpace(1));
    
        # function return
        return randomMPS;
    
    end

    function orthogonalizeMPS!(finiteMPS::Vector{<:AbstractTensorMap}, orthCenter::Int)
        """ Function to bring MPS into mixed canonical form with orthogonality center at site 'orthCenter' """
    
        # get length of finiteMPS
        N = length(finiteMPS);
    
        # bring sites 1 to orthCenter - 1 into left-orthogonal form
        for siteIdx = 1 : +1 : (orthCenter - 1)
            (Q, R) = leftorth(finiteMPS[siteIdx], (1, 2), (3, ), alg = QRpos());
            finiteMPS[siteIdx + 0] = permute(Q, (1, 2), (3, ));
            finiteMPS[siteIdx + 1] = permute(R * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, ));
        end
    
        # bring sites orthCenter + 1 to N into right-canonical form
        for siteIdx = N : -1 : (orthCenter + 1)
            (L, Q) = rightorth(finiteMPS[siteIdx], (1, ), (2, 3), alg = LQpos());
            finiteMPS[siteIdx - 1] = permute(permute(finiteMPS[siteIdx - 1], (1, 2), (3, )) * L, (1, 2), (3, ));
            finiteMPS[siteIdx - 0] = permute(Q, (1, 2), (3, ));
        end
        return finiteMPS;
    
    end
    
    function orthogonalizeMPS(finiteMPS::Vector{<:AbstractTensorMap}, args...)
        newMPS = copy(finiteMPS);
        orthogonalizeMPS!(newMPS, args...);
        return newMPS
    end

    function normalizeMPS(finiteMPS::Vector{<:AbstractTensorMap})
        """ Brings MPS into canonical form and returns normalized MPS """
    
        # bring finiteMPS into canonical form
        orthogonalizeMPS!(finiteMPS, 1);
        normMPS = real(tr(finiteMPS[1]' * finiteMPS[1]));
        finiteMPS[1] /= sqrt(normMPS);
        return finiteMPS;
    
    end

    function constructHeisenbergMPO(J, h, N)
        """ Returns Heisenberg MPO for N sites """
    
        # set Pauli matrices
        Sx = 0.5 * [0 +1 ; +1 0];
        Sy = 0.5 * [0 -1im ; +1im 0];
        Sz = 0.5 * [+1 0 ; 0 -1];
        Id = [+1 0 ; 0 +1];
        d = 2;
    
        # initialize Heisenberg MPO
        heisenbergMPO = Vector{TensorMap}(undef, N);
        
        # left MPO boundary
        HL = zeros(ComplexF64, 1, d, d, 5);
        randomField = h * 2 * (rand() - 0.5);
        HL[1, :, :, 1] = Id;
        HL[1, :, :, 2] = Sx;
        HL[1, :, :, 3] = Sy;
        HL[1, :, :, 4] = Sz;
        HL[1, :, :, 5] = randomField * Sz;
        heisenbergMPO[1] = TensorMap(HL, ComplexSpace(1) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(5));
    
        # bulk MPO
        for idxMPO = 2 : (N - 1)
            HC = zeros(ComplexF64, 5, d, d, 5);
            randomField = h * 2 * (rand() - 0.5);
            HC[1, :, :, 1] = Id;
            HC[1, :, :, 2] = Sx;
            HC[1, :, :, 3] = Sy;
            HC[1, :, :, 4] = Sz;
            HC[1, :, :, 5] = randomField * Sz;
            HC[2, :, :, 5] = J * Sx;
            HC[3, :, :, 5] = J * Sy;
            HC[4, :, :, 5] = J * Sz;
            HC[5, :, :, 5] = Id;
            heisenbergMPO[idxMPO] = TensorMap(HC, ComplexSpace(5) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(5));
        end
    
        # right MPO boundary
        HR = zeros(ComplexF64, 5, d, d, 1);
        randomField = h * 2 * (rand() - 0.5);
        HR[1, :, :, 1] = randomField * Sz;
        HR[2, :, :, 1] = J * Sx;
        HR[3, :, :, 1] = J * Sy;
        HR[4, :, :, 1] = J * Sz;
        HR[5, :, :, 1] = Id;
        heisenbergMPO[N] = TensorMap(HR, ComplexSpace(5) ⊗ ComplexSpace(d), ComplexSpace(d) ⊗ ComplexSpace(1));
    
        # function return
        return heisenbergMPO;
    
    end

    # function applyH1(X, EL, mpo, ER)
    #     @tensor X[-1 -2; -3] := EL[-1, 2, 1] * X[1, 3, 4] * mpo[2, -2, 5, 3] * ER[4, 5, -3]
    #     return X
    # end
    
    function applyH2(X, EL, mpo1, mpo2, ER)
        @tensor X[-1 -2; -3 -4] := EL[-1, 2, 1] * X[2, 3, 5, 6] * mpo1[1, -2, 3, 4] * mpo2[4, -3, 5, 7] * ER[6, 7, -4];
        return X
    end

    function update_MPOEnvL(mpoEnvL, mpsTensorK, mpoTensor, mpsTensorB)
        @tensor newEL[-1; -2 -3] := mpoEnvL[1, 5, 3] * mpsTensorK[5, 4, -2] * mpoTensor[3, 2, 4, -3] * conj(mpsTensorB[1, 2, -1]);
        return newEL;
    end
    
    function update_MPOEnvR(mpoEnvR, mpsTensorK, mpoTensor, mpsTensorB)
        @tensor newER[-1 -2; -3] := mpsTensorK[-1, 2, 1] * mpoTensor[-2, 4, 2, 3] * conj(mpsTensorB[-3, 4, 5]) * mpoEnvR[1, 3, 5];
        return newER;
    end

    function initializeMPOEnvironments(finiteMPS::Vector{<:AbstractTensorMap}, finiteMPO::Vector{<:AbstractTensorMap}; centerPos::Int64 = 1)

        # get length of finiteMPS
        N = length(finiteMPS);
    
        # construct MPO environments
        mpoEnvL = Vector{TensorMap}(undef, N);
        mpoEnvR = Vector{TensorMap}(undef, N);
    
        # initialize end-points of mpoEnvL and mpoEnvR
        mpoEnvL[1] = TensorMap(ones, space(finiteMPS[1], 1), space(finiteMPS[1], 1) ⊗ space(finiteMPO[1], 1));
        mpoEnvR[N] = TensorMap(ones, space(finiteMPS[N], 3)' ⊗ space(finiteMPO[N], 4)', space(finiteMPS[N], 3)');
    
        # compute mpoEnvL up to (centerPos - 1)
        for siteIdx = 1 : +1 : (centerPos - 1)
            mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);
        end
    
        # compute mpoEnvR up to (centerPos + 1)
        for siteIdx = N : -1 : (centerPos + 1)
            mpoEnvR[siteIdx - 1] = update_MPOEnvR(mpoEnvR[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);
        end
        return mpoEnvL, mpoEnvR;
    
    end

    function DMRG2(finiteMPS::Vector{<:AbstractTensorMap}, finiteMPO::Vector{<:AbstractTensorMap}; bondDim::Int64 = 20, truncErr::Float64 = 1e-6, convTolE::Float64 = 1e-6, eigsTol::Float64 = 1e-16, maxIterations::Int64 = 1, subspaceExpansion::Bool = true, verbosePrint = false)

        # get length of finiteMPS
        N = length(finiteMPS);
    
        # initialize mpsEnergy
        initialEnergy = computeExpValMPO(finiteMPS, finiteMPO);
        mpsEnergy = Float64[initialEnergy];
    
        # initialize MPO environments
        mpoEnvL, mpoEnvR = initializeMPOEnvironments(finiteMPS, finiteMPO);

        # main DMRG loop
        loopCounter = 1;
        runOptimizationDMRG = true;
        while runOptimizationDMRG

            # sweep L ---> R
            for siteIdx = 1 : +1 : (N - 1)

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(theta, 1, :SR, KrylovKit.Lanczos(tol = eigsTol, maxiter = maxIterations)) do x
                        applyH2(x, mpoEnvL[siteIdx], finiteMPO[siteIdx], finiteMPO[siteIdx + 1], mpoEnvR[siteIdx + 1])
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
                U = permute(U, (1, 2), (3, ));
                V = permute(S * V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = update_MPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

            end

            # sweep L <--- R
            for siteIdx = (N - 1) : -1 : 1

                # construct initial theta
                theta = permute(finiteMPS[siteIdx] * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));

                # optimize wave function to get newAC
                eigenVal, eigenVec = 
                    eigsolve(theta, 1, :SR, KrylovKit.Lanczos(tol = eigsTol, maxiter = maxIterations)) do x
                        applyH2(x, mpoEnvL[siteIdx], finiteMPO[siteIdx], finiteMPO[siteIdx + 1], mpoEnvR[siteIdx + 1])
                    end
                # eigVal = eigenVal[1];
                newTheta = eigenVec[1];

                #  perform SVD and truncate to desired bond dimension
                U, S, V, ϵ = tsvd(newTheta, (1, 2), (3, 4), trunc = truncdim(bondDim) & truncerr(truncErr), alg = TensorKit.SVD());
                U = permute(U * S, (1, 2), (3, ));
                V = permute(V, (1, 2), (3, ));

                # assign updated tensors and update MPO environments
                finiteMPS[siteIdx + 0] = U;
                finiteMPS[siteIdx + 1] = V;

                # update mpoEnvR
                mpoEnvR[siteIdx + 0] = update_MPOEnvR(mpoEnvR[siteIdx + 1], finiteMPS[siteIdx + 1], finiteMPO[siteIdx + 1], finiteMPS[siteIdx + 1]);
                
            end

            # normalize MPS after DMRG step
            mpsNorm = tr(finiteMPS[1]' * finiteMPS[1]);
            finiteMPS[1] /= sqrt(mpsNorm);

            # compute MPO expectation value
            mpoExpVal = computeExpValMPO(finiteMPS, finiteMPO);
            if abs(imag(mpoExpVal)) < 1e-12
                mpoExpVal = real(mpoExpVal);
            else
                ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
            end
            mpsEnergy = vcat(mpsEnergy, mpoExpVal);
            
            # check convergence of ground state energy
            energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
            verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, mpoExpVal, energyConvergence);
            if energyConvergence < convTolE
                runOptimizationDMRG = false;
            end

            # increase loopCounter
            loopCounter += 1;

        end
    
        # return optimized finiteMPS
        finalEnergy = mpsEnergy[end];
        return finiteMPS, finalEnergy;
    
    end

    function computeExpValMPO(finiteMPS::Vector{<:AbstractTensorMap}, finiteMPO::Vector{<:AbstractTensorMap})
        """ Computes expectation value for MPS and MPO """
    
        # get length of finiteMPS
        N = length(finiteMPS);
    
        # normalize finiteMPS
        finiteMPS = normalizeMPS(finiteMPS);
        
        # contract from left to right
        boundaryL = TensorMap(ones, space(finiteMPS[1], 1), space(finiteMPS[1], 1) ⊗ space(finiteMPO[1], 1));
        for siteIdx = 1 : +1 : N
            @tensor boundaryL[-1; -2 -3] := boundaryL[1, 5, 3] * finiteMPS[siteIdx][5, 4, -2] * finiteMPO[siteIdx][3, 2, 4, -3] * conj(finiteMPS[siteIdx][1, 2, -1]);
        end
        boundaryR = TensorMap(ones, space(finiteMPS[N], 3)' ⊗ space(finiteMPO[N], 4)', space(finiteMPS[N], 3)');
    
        # contact to get expectation value
        expectationVal = tr(boundaryL * boundaryR);
        return expectationVal;
        
    end



    # set physical system
    N = 32;
    physicalSpin = 1/2;
    d = Int(2 * physicalSpin + 1);

    # construct MPO for Heisenberg model
    J = 1.0;
    h = 0.0;
    heisenbergMPO = constructHeisenbergMPO(J, h, N);

    # initialize MPS
    initialMPS = initializeRandomMPS(N);
    
    # run DMRG
    gsMPS, gsEnergy = DMRG2(initialMPS, heisenbergMPO, bondDim = 16, truncErr = 1e-6, convTolE = 1e-6, verbosePrint = true);
    @sprintf("ground state energy per site E = %0.6f", gsEnergy / N)

end