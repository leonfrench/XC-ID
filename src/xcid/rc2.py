import numpy as np
from pysat.examples.rc2 import RC2Stratified
from pysat.formula import WCNF
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .scoring import get_score


def rc2_phasing(counts_ref, counts_alt, rng=None, verbose=False):
    '''Exact alternative to simulated_annealing: assign haplotypes to SNPs by maximizing the same
    objective, the sum over cells of |hap0 - hap1|, with the RC2 MaxSAT solver from PySAT, which
    proves its answer optimal. Returns the assignment, its cost and the cell scores, like
    simulated_annealing.

    Each SNP's assignment and each cell's haplotype is a variable, and a cell's surplus UMIs at a
    SNP (|ref - alt|) are lost when they land on the other haplotype; RC2 minimizes the UMIs lost.
    Cells with fewer than two informative SNPs score the same under any assignment and are left
    out. The objective can't tell a group of linked SNPs (sharing cells) from its mirror image, or
    orient a SNP outside every group, so those orientations are random, as in annealing.'''
    if rng is None:
        rng = np.random.default_rng()

    diff = counts_ref.astype(np.int64) - counts_alt
    # identical columns (e.g. a SNP the bootstrap drew twice) are solved once, weighted by how
    # often they occur, which has the same optimum
    cols, inverse, mult = np.unique(diff, axis=1, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    diff_u = cols * mult
    assignment_u = rng.integers(0, 2, size=diff_u.shape[1]) # kept wherever the objective has no say

    informative = diff_u != 0
    rows = np.flatnonzero(informative.sum(axis=1) >= 2)
    links = csr_matrix(informative[rows].astype(np.int32))
    coverage = np.asarray(links.sum(axis=0)).ravel()
    used = np.flatnonzero(coverage > 0)
    _, group = connected_components(links.T @ links, directed=False)
    if verbose:
        print(f"Solving the phasing exactly with RC2: {int(np.isin(inverse, used).sum())} of {len(inverse)} SNPs "
              f"linked through {len(rows)} cells, in {len(np.unique(group[used]))} group(s).")

    if len(used):
        var = {j: v + 1 for v, j in enumerate(used)} # SNP variables: true = ALT on haplotype 0
        wcnf = WCNF()
        # the best-covered SNP of each group keeps its random orientation, which fixes the group's
        for g in np.unique(group[used]):
            members = used[group[used] == g]
            j = members[np.argmax(coverage[members])]
            wcnf.append([var[j] if assignment_u[j] else -var[j]])
        z = len(used)
        for i in rows:
            z += 1 # the cell's variable: true = the cell is on haplotype 0
            for j in np.flatnonzero(informative[i]):
                d, x = int(diff_u[i, j]), var[j]
                if d > 0: # surplus REF UMIs are lost when z == x
                    wcnf.append([-z, -x], weight=d)
                    wcnf.append([z, x], weight=d)
                else: # surplus ALT UMIs are lost when z != x
                    wcnf.append([-z, x], weight=-d)
                    wcnf.append([z, -x], weight=-d)
        with RC2Stratified(wcnf, solver="g3", adapt=True, exhaust=True, minz=True) as rc2:
            model = rc2.compute()
        for j in used:
            assignment_u[j] = int(model[var[j] - 1] > 0)

    assignment = assignment_u[inverse]
    cost = -float(np.abs(diff @ (1 - 2 * assignment)).sum()) # negative, as in simulated_annealing
    if verbose:
        print(f"Optimal cost: {cost}.")

    return assignment, cost, get_score(counts_ref, counts_alt, assignment)
