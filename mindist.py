import numpy as np
import galois
import z3


def xzdual(row):
    assert len(row)%2==0
    num_qubits = len(row)//2
    return row[num_qubits:] + row[:num_qubits]

def matrix_product(A, B, addfunc=lambda x,y: x+y):
    m1 = len(A)
    n1 = len(A[0])
    m2 = len(B)
    n2 = len(B[0])
    assert n1 == m2
    C = [
        [# n2 columns
         0 for j in range(n2)]
        for i in range(m1) # m1 rows
    ]
    for i in range(m1):
        for j in range(n2):
            for k in range(n1):
                C[i][j] = addfunc(C[i][j], A[i][k] * B[k][j])
    return C

def FastOr(a, b):
    a_is_const = a is True or a is False
    b_is_const = b is True or b is False
    if a_is_const or b_is_const:
        if a is True or b is True:
            return True
        if a is False:
            return b
        if b is False:
            return a
    return z3.Or(a, b)

def FastAnd(a,b):
    a_is_const = a is True or a is False
    b_is_const = b is True or b is False
    if a_is_const or b_is_const:
        if a is True:
            return b
        if b is True:
            return a
        return False
    else:
        return z3.And(a, b)

def FastNot(a):
    a_is_const = a is True or a is False
    if a_is_const:
        return not a
    else:
        return z3.Not(a)

def FastXor(a, b):
    a_is_const = a is True or a is False
    b_is_const = b is True or b is False
    if a_is_const or b_is_const:
        if a is False:
            return b
        if b is False:
            return a
        if a is True:
            if b is False:
                return True
            if b is True:
                return False
            return FastNot(b)
        if b is True:
            if a is False:
                return True
            if a is True:
                return False
            return FastNot(a)

    return z3.Xor(a, b)

def FastXorSum(li):
    const = False
    nonconsts = set(range(len(li)))
    for i in range(len(li)):
        val = li[i]
        val_is_const = val is True or val is False
        if val_is_const:
            nonconsts.remove(i)
            const = const != val
    if not nonconsts:
        return const
    else:
        nonconsts = list(nonconsts)
        nonconst_sum = li[nonconsts[0]]
        for i in nonconsts[1:]:
            nonconst_sum = FastXor(nonconst_sum, li[i])
        if const:
            return FastNot(nonconst_sum)
        else:
            return nonconst_sum

def minimum_distance(num_qubits, x_stabs, z_stabs):
    stabilizer_gens = []
    for stab in x_stabs:
        gen = [0 for _ in range(2*num_qubits)]
        for i in stab:
            gen[i] = 1
        stabilizer_gens.append(gen)
    for stab in z_stabs:
        gen = [0 for _ in range(2*num_qubits)]
        for i in stab:
            gen[num_qubits + i] = 1
        stabilizer_gens.append(gen)

    num_gens = len(stabilizer_gens)
    A = np.array(stabilizer_gens, dtype='int')
    GF = galois.GF(2)
    A = GF(A)
    NS = A.null_space()
    prod = np.array(matrix_product(
        np.array(A, dtype=int), np.array(NS.transpose(), dtype=int),
        addfunc=lambda x,y: (x+y)%2), dtype=int)
    assert (prod == 0).all
    dual_code_gens = np.array([
        xzdual(list(row)) for row in np.array(NS, dtype=int)], dtype='int')
    x = {}
    operator = [False for i in range(num_qubits*2)]
    assert len(operator) == 2*num_qubits
    for i, gen in enumerate(dual_code_gens):
        assert len(operator) == 2*num_qubits
        x[i] = z3.Bool(f'x_{i}')
        for j in range(num_qubits*2):
            operator[j] = FastXor(operator[j], FastAnd(x[i], bool(gen[j])))
    is_not_stabilizer = False
    for i, gen in enumerate(dual_code_gens):
        anticommutes = FastXorSum([
            FastAnd(bool(u), v)
            for u,v in zip(xzdual(gen), operator)])
        is_not_stabilizer = FastOr(is_not_stabilizer, anticommutes)
    total_weight = 0
    for j in range(num_qubits):
        total_weight += z3.If(FastOr(operator[j], operator[num_qubits+j]), 1, 0)
    constraints = []
    constraints.append(is_not_stabilizer)
    opt = z3.Optimize()
    handle = opt.minimize(total_weight)
    for constraint in constraints:
        opt.add(constraint)
    assert opt.check() == z3.sat
    assert opt.lower(handle) == opt.upper(handle)
    return opt.upper(handle)
