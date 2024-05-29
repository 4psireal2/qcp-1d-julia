include("utility.jl")
include("dmrg.jl")


using LinearAlgebra
using Printf
using KrylovKit
using MPSKit
using TensorKit

using Base: @kwdef


@kwdef struct DMRG1_params
    truncErr::Float64 = 1e-6
    convTolE::Float64 = 1e-6
    eigsTol::Float64 = 1e-12
    maxIterations::Int64 = 1
    subspaceExpansion::Bool = true
    verbosePrint = false
end


function update_MPSEnvL(mpsEnvL, mpsTensorK, mpsTensorB)
    @tensor newEL[-1; -2] := mpsEnvL[1, 3] * mpsTensorK[3, 2, 4, -2] * conj(mpsTensorB[1, 2, 4, -1]);
    return newEL;
end

function update_MPSEnvR(mpsEnvR, mpsTensorK, mpsTensorB)
    @tensor newER[-1; -2] := mpsTensorK[-1, 2, 4, 1] * conj(mpsTensorB[-2, 2, 4, 3]) * mpsEnvR[1, 3];
    return newER;
end

function initializePROEnvironments(ketMPS::Vector{TensorMap}, orthStates::Vector{Vector{TensorMap}}; centerPos::Int64 = 1)

    # get length of ketMPS
    N = length(ketMPS);

    # construct projector environments
    numOrthStates = length(orthStates);
    projEnvsL = Vector{Vector{TensorMap}}(undef, numOrthStates);
    projEnvsR = Vector{Vector{TensorMap}}(undef, numOrthStates);
    for orthIdx = eachindex(orthStates)

        # select braMPS
        braMPS = orthStates[orthIdx];

        # initialize projector environments
        projEnvL = Vector{TensorMap}(undef, N);
        projEnvR = Vector{TensorMap}(undef, N);

        # initialize end-points of projEnvL and projEnvR
        projEnvL[1] = TensorMap(ones, space(braMPS[1], 1), space(ketMPS[1], 1));
        projEnvR[N] = TensorMap(ones, space(ketMPS[N], 4)', space(braMPS[N], 4)');

        # compute projEnvL up to (centerPos - 1)
        for siteIdx = 1 : +1 : (centerPos - 1)
            projEnvL[siteIdx + 1] = update_MPSEnvL(projEnvL[siteIdx], ketMPS[siteIdx], braMPS[siteIdx]);
        end

        # compute projEnvR up to (centerPos + 1)
        for siteIdx = N : -1 : 2
            projEnvR[siteIdx - 1] = update_MPSEnvR(projEnvR[siteIdx], ketMPS[siteIdx], braMPS[siteIdx]);
        end

        # store projector environments
        projEnvsL[orthIdx] = projEnvL;
        projEnvsR[orthIdx] = projEnvR;

    end
    return projEnvsL, projEnvsR;

end

function applyH1_X(X, siteIdx, EL, MPO, ER, PLs, Y, PRs, penaltyWeights::Vector{Float64})

    # compute H|ψexcited⟩
    @tensor X1[-1 -2 -3; -4]  := EL[siteIdx][-1, 2, 1] * X[2, 3, 4, 5] * MPO[siteIdx][1, -2, -3, 3, 4, 6] * ER[siteIdx][5, 6, -4];

    # compute overlaps ⟨ψ0|ψexcited⟩, ⟨ψ1|ψexcited⟩, ...
    waveFunctionOverlaps = Vector{Union{Float64, ComplexF64}}(undef, length(Y));
    for orthIdx = eachindex(Y)
        waveFunctionOverlap = @tensor PLs[orthIdx][siteIdx][1, 2] * X[2, 3, 4, 5] * conj(Y[orthIdx][1, 3, 4, 6]) * PRs[orthIdx][siteIdx][5, 6]; #XXX:  PRs[orthIdx][siteIdx + 1]?
        waveFunctionOverlaps[orthIdx] = waveFunctionOverlap;
    end

    # compute ω0 P0|ψexcited⟩, ω1 P1|ψexcited⟩, ... 
    projOverlaps = Vector{TensorMap}(undef, length(Y));
    for orthIdx = eachindex(Y)
        @tensor XO[-1 -2 -3; -4] := conj(PLs[orthIdx][siteIdx][1, -1]) * Y[orthIdx][1, -2, -3, 4] * conj(PRs[orthIdx][siteIdx][-4, 4]); #XXX:  conj(PRs[orthIdx][siteIdx + 1][-4, 4])?
        projOverlaps[orthIdx] = penaltyWeights[orthIdx] * waveFunctionOverlaps[orthIdx] * XO;
    end
    
    # return sum of terms
    return X1 + sum(projOverlaps);

end

function find_excitedstate!(finiteMPS::Vector{TensorMap}, finiteMPO::Vector{TensorMap}, previousMPS::Vector{<:Vector{TensorMap}}, alg::DMRG1_params)

    N = length(finiteMPS);

    # set penaltyWeights
    penaltyWeights = 1e2 * ones(length(previousMPS));

    # initialize mpsEnergy
    mpsEnergy = Float64[1.0];

    # initialize MPO environments
    mpoEnvL, mpoEnvR = initializeMPOEnvs(finiteMPS, finiteMPO);

    # initialize projector environments
    projEnvsL, projEnvsR = initializePROEnvironments(finiteMPS, previousMPS);

    # main DMRG loop
    loopCounter = 1;
    runOptimizationDMRG = true;
    while runOptimizationDMRG

        # sweep L ---> R
        for siteIdx = 1 : +1 : N

            # construct initial theta
            thetaN = finiteMPS[siteIdx];

            # construct thetas for orthStates
            thetasO = Vector{TensorMap}(undef, length(previousMPS))
            for orthMPO = eachindex(previousMPS)
                # thetasO[orthMPO] = permute(previousMPS[orthMPO][siteIdx] * permute(previousMPS[orthMPO][siteIdx + 1], (1, ), (2, 3)), (1, 2), (3, 4));
                thetasO[orthMPO] = previousMPS[orthMPO][siteIdx];
            end

            # optimize wave function to get newAC
            eigenVal, eigenVec = 
                eigsolve(thetaN, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                    applyH1_X(x, siteIdx, mpoEnvL, finiteMPO, mpoEnvR, projEnvsL, thetasO, projEnvsR, penaltyWeights)
                end
            # eigVal = eigenVal[1];
            newTheta = eigenVec[1];

            # shift orthogonality center to the right and update MPO environments
            if siteIdx < N
                (Q, R) = leftorth(newTheta, (1, 2, 3), (4, ), alg = QRpos())
                R /= norm(R);
                finiteMPS[siteIdx] = permute(Q, (1, 2, 3), (4, ));
                finiteMPS[siteIdx + 1] = permute(R * permute(finiteMPS[siteIdx + 1], (1, ), (2, 3, 4)), (1, 2, 3), (4, ));

                # shift orthogonality center of previousMPS to the right
                for orthIdx = eachindex(previousMPS)
                    (Q, R) = leftorth(previousMPS[orthIdx][siteIdx], (1, 2, 3), (4, ), alg = QRpos());
                    previousMPS[orthIdx][siteIdx + 0] = permute(Q, (1, 2, 3), (4, ));
                    previousMPS[orthIdx][siteIdx + 1] = permute(R * permute(previousMPS[orthIdx][siteIdx + 1], (1, ), (2, 3, 4)), (1, 2, 3), (4, ));

                    projEnvsL[orthIdx][siteIdx + 1] = update_MPSEnvL(projEnvsL[orthIdx][siteIdx], finiteMPS[siteIdx], previousMPS[orthIdx][siteIdx]);
                end


                # update mpoEnvL
                mpoEnvL[siteIdx + 1] = updateMPOEnvL(mpoEnvL[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);
            ##################### XXX: Correct no?
            else
                finiteMPS[siteIdx] = newTheta;
            end
            #####################


        end

        # sweep L <--- R
        for siteIdx = (N - 1) : -1 : 1

            # construct initial theta
            thetaN = finiteMPS[siteIdx];

            # construct thetas for orthStates
            thetasO = Vector{TensorMap}(undef, length(previousMPS))
            for orthMPO = eachindex(previousMPS)
                thetasO[orthMPO] = previousMPS[orthMPO][siteIdx];
            end

            # optimize wave function to get newAC
            eigenVal, eigenVec = 
                eigsolve(thetaN, 1, :SR, KrylovKit.Lanczos(tol = alg.eigsTol, maxiter = alg.maxIterations)) do x
                    applyH1_X(x, siteIdx, mpoEnvL, finiteMPO, mpoEnvR, projEnvsL, thetasO, projEnvsR, penaltyWeights)
                end
            # eigVal = eigenVal[1];
            newTheta = eigenVec[1];


            if siteIdx > 1

                # right-orthogonalize newTensor
                (L, Q) = rightorth(newTheta, (1, ), (2, 3, 4), alg = LQpos());
                L /= norm(L);
                finiteMPS[siteIdx] = permute(Q, (1, 2, 3), (4, ));

                # absorb L into previous MPS site
                finiteMPS[siteIdx - 1] = permute(finiteMPS[siteIdx - 1] * L, (1, 2, 3), (4, ));

                # update mpoEnvR
                mpoEnvR[siteIdx - 1] = updateMPOEnvR(mpoEnvR[siteIdx], finiteMPS[siteIdx], finiteMPO[siteIdx], finiteMPS[siteIdx]);

                # shift orthogonality center of previousMPS to the right
                for orthIdx = eachindex(previousMPS)
                    (L, Q) = rightorth(previousMPS[orthIdx][siteIdx], (1, ), (2, 3, 4), alg = LQpos());
                    previousMPS[orthIdx][siteIdx + 0] = permute(permute(previousMPS[orthIdx][siteIdx + 0], (1, 2, 3), (4, )) * L, (1, 2, 3), (4, ));
                    previousMPS[orthIdx][siteIdx + 1] = permute(Q, (1, 2, 3), (4, ));
                end

                # update projEnvsR
                for orthIdx = eachindex(previousMPS)
                    projEnvsR[orthIdx][siteIdx + 0] = update_MPSEnvR(projEnvsR[orthIdx][siteIdx + 1], finiteMPS[siteIdx + 1], previousMPS[orthIdx][siteIdx + 1]);
                end
            else
                finiteMPS[siteIdx] = newTensor;
            end

        end

        # # normalize MPS after DMRG step
        # mpsNorm = tr(finiteMPS[1]' * finiteMPS[1]);
        # finiteMPS[1] /= sqrt(mpsNorm);

        # compute MPO expectation value
        mpoExpVal = computeExpVal(finiteMPS, finiteMPO);
        if abs(imag(mpoExpVal)) < 1e-12
            mpoExpVal = real(mpoExpVal);
        else
            ErrorException("the Hamiltonian is not Hermitian, complex eigenvalue found.")
        end
        mpsEnergy = vcat(mpsEnergy, mpoExpVal);

        # check convergence of ground state energy
        energyConvergence = abs(mpsEnergy[end - 1] - mpsEnergy[end]);
        alg.verbosePrint && @printf("DMRG step %d - energy, convergence = %0.8f, %0.8e\n", loopCounter, mpoExpVal, energyConvergence);
        if energyConvergence < alg.convTolE
            runOptimizationDMRG = false;
        end

        # increase loopCounter
        loopCounter += 1;

    end

    # return optimized finiteMPS
    finalEnergy = mpsEnergy[end];
    return finiteMPS, finalEnergy;

end

function find_excitedstate(finiteMPS::Vector{TensorMap}, finiteMPO::Vector{TensorMap}, previousMPS::Vector{Vector{TensorMap}}, alg::DMRG1_params)
    return find_excitedstate!(copy(finiteMPS), finiteMPO, previousMPS, alg)
end