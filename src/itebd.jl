using TensorKit
using KrylovKit # Lanczos - real EVal, Oddrnoldi - complex Eval


function contractEnvL(X, oddT, evenT, wOddEven, wEvenOdd)
    @tensor X[-2; -1] := X[7, 1] * wEvenOdd[1, 2] * oddT[2, 3, 4] * wOddEven[4, 5] * evenT[5, 6, -1] * conj(wEvenOdd[7, 8]) * conj(oddT[8, 3, 9]) * conj(wOddEven[9, 10]) * conj(evenT[10, 6, -2]);
    return X
end


function contractEnvR(X, oddT, evenT, wOddEven, wEvenOdd)
    @tensor X[-1; -2] := X[6, 10] * evenT[-1, 1, 2] * wEvenOdd[2, 3] * oddT[3, 4, 5] * wOddEven[5, 6] * conj(evenT[-2, 1, 7]) * conj(wEvenOdd[7, 8]) * conj(oddT[8, 4, 9]) * conj(wOddEven[9, 10]);
    return X
end


function leftContraction!(oddT, evenT, wOddEven, wEvenOdd, leftT_EO; niters=1000, tol=1e-12)
    """
    Contract an infinite 2-site unit cell from the left to create transfer operators
    leftT_OE and leftT_EO using Lanczos method as eigensolver
    """

    # create left transfer operator across even-odd link
    _, leftT_EO = eigsolve(leftT_EO, 1, :LM, KrylovKit.Arnoldi(tol = tol, maxiter = niters)) do x contractEnvL(x, oddT, evenT, wOddEven, wEvenOdd) end
    leftT_EO = leftT_EO[1]; # extract dominant eigenvector
    leftT_EO /= norm(leftT_EO);

    # create left transfer operator across odd-even link
    @tensor leftT_OE[-2; -1] :=  leftT_EO[4, 1] * wEvenOdd[1, 2] * conj(wEvenOdd[4, 5]) * oddT[2, 3, -1] * conj(oddT[5, 3, -2]);
    leftT_OE /= norm(leftT_OE);

    return leftT_OE, leftT_EO
end


function rightContraction!(oddT, evenT, wOddEven, wEvenOdd, rightT_OE; niters=1000, tol=1e-12)
    """
    Contract an infinite 2-site unit cell from the right to create transfer operators
    rightT_OE and rightT_EO
    """

    # create right transfer operator across odd-even link
    _, rightT_OE = eigsolve(rightT_OE, 1, :LM, KrylovKit.Arnoldi(tol = tol, maxiter = niters)) do x contractEnvR(x, oddT, evenT, wOddEven, wEvenOdd) end
    rightT_OE = rightT_OE[1]; # extract dominant eigenvector
    rightT_OE /= norm(rightT_OE);

    # create right transfer operator across odd-even link
    @tensor rightT_EO[-1; -2] :=  rightT_OE[3, 5] * oddT[-1, 1, 2] * conj(oddT[-2, 1, 4]) * wOddEven[2, 3] * conj(wOddEven[4, 5]);
    rightT_EO /= norm(rightT_EO);

    return rightT_OE, rightT_EO
end


function leftContraction_test!(oddT, evenT, wOddEven, wEvenOdd, leftT_EO; niters=1000, tol=1e-12)
    """
    Contract an infinite 2-site unit cell from the left to create transfer operators
    leftT_OE and leftT_EO using power method as eigensolver
    """

    # create left transfer operator across even-odd link
    for i = 1 : niters
        leftT_EO_new = contractEnvL(leftT_EO, oddT, evenT, wOddEven, wEvenOdd);
        leftT_EO_new /= norm(leftT_EO_new);

        if norm(leftT_EO - leftT_EO_new) <= tol
            leftT_EO = leftT_EO_new;
            break
        else
            leftT_EO = leftT_EO_new;
        end
    end

    # create left transfer operator across odd-even link
    @tensor leftT_OE[-2; -1] :=  leftT_EO[4, 1] * wEvenOdd[1, 2] * conj(wEvenOdd[4, 5]) * oddT[2, 3, -1] * conj(oddT[5, 3, -2]);
    leftT_OE /= norm(leftT_OE);

    return leftT_OE, leftT_EO
end


function rightContraction_test!(oddT, evenT, wOddEven, wEvenOdd, rightT_OE; niters=1000, tol=1e-12)
    """
    Contract an infinite 2-site unit cell from the right to create transfer operators
    rightT_OE and rightT_EO using power method as eigensolver
    """

    # create right transfer operator across odd-even link
    for _ = 1 : niters
        rightT_OE_new = contractEnvR(rightT_OE, oddT, evenT, wOddEven, wEvenOdd);
        rightT_OE_new /= norm(rightT_OE_new);

        if norm(rightT_OE - rightT_OE_new) <= tol
            rightT_OE = rightT_OE_new;
            break
        else
            rightT_OE = rightT_OE_new;
        end
    end

    # create right transfer operator across odd-even link
    @tensor rightT_EO[-1; -2] :=  rightT_OE[3, 5] * oddT[-1, 1, 2] * conj(oddT[-2, 1, 4]) * wOddEven[2, 3] * conj(wOddEven[4, 5]);
    rightT_EO /= norm(rightT_EO);

    return rightT_OE, rightT_EO
end


function orthogonalizeiMPS!(leftT, rightT, weight, envL, envR)
    
    # diagonalize left environment
    evalL, evecL = eig(envL, (1, ), (2, ));

    # diagonalize right environment
    evalR, evecR = eig(envR, (1, ), (2, ));

    # update weight
    @tensor weight[-1; -2] := sqrt(evalL)[-1, 1] * evecL'[1, 2] * weight[2, 3] * evecR[3, 4] * sqrt(evalR)[4, -2];
    weight /= norm(weight);

    @tensor leftT[-1 -2; -3] := leftT[-1, -2, 1] * evecL[1, 2] * sqrt(evalL)[2, -3];
    leftT /= norm(leftT);

    @tensor rightT[-1 -2; -3] := sqrt(evecR)[-1, 1] * evecR'[1, 2] * rightT[2, -2, -3];

    return leftT, rightT, weight
end