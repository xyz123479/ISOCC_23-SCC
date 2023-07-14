import numpy as np
import random

np.set_printoptions(linewidth=np.inf, ) 

# Error correction capability of this scheme is SCC : can correct 1-bit error and 1-cell error

# If you want to make another error case, change
#  1) range of 'error_pos' in for statement
#  2) method of making error using 'error_pos'
#  3) # of total cases in print statement

def make_syndrome(codeword, H_matrix) :
    # Calculate syndrome using H-matrix
    codeword_transpose = codeword.transpose()
    syndrome = H_matrix.dot(codeword_transpose) % 2
    return syndrome

def correction_SCC(codeword, syndrome, H_matrix, Sum_matrix) :
    corrected_codeword = codeword
    codeword_len = len(codeword)
    nonzero = False      # true if nonzero syndrome
    correctable = False  # true if the syndrome exists in the H_matrix

    # Try error correction if nonzero syndrome (There are 2 types of correction)
    if not np.all(syndrome == 0) :
        nonzero = True
        # Single error correction using H-matrix
        for error_idx in range(codeword_len) :
            if np.array_equal(H_matrix[:, error_idx], syndrome) :
                corrected_codeword[error_idx] ^= 1  # correct error
                correctable = True
                break
        # Single cell correction using Sum-matrix
        else :
            for error_idx in range(codeword_len // 2) :
                if np.array_equal(Sum_matrix[:, error_idx], syndrome) :
                    corrected_codeword[error_idx * 2] ^= 1  # correct error
                    corrected_codeword[error_idx * 2 + 1] ^= 1
                    correctable = True
                    break

    return corrected_codeword, nonzero, correctable

def verify(corrected_codeword, nonzero, correctable, CE_cnt, DUE_cnt, SDC_cnt) :
    # Verify if the error correction has successfully finished
    if not nonzero :
        SDC_cnt += 1          # error but zero syndrome
    else :
        if not correctable :
            DUE_cnt += 1      # detected but not correctable
        else :
            if not np.all(corrected_codeword == 0) :
                SDC_cnt += 1  # detected but miscorrected
            else :
                CE_cnt += 1   # detected and well-corrected

    return CE_cnt, DUE_cnt, SDC_cnt

def main() :
    H_matrix_T = np.loadtxt("SCC_H_matrix_T.txt", dtype = "int")
    Sum_matrix_T = np.loadtxt("SCC_Sum_matrix_T.txt", dtype = "int")
    H_matrix = H_matrix_T.transpose()
    Sum_matrix =  Sum_matrix_T.transpose()
    redundancy_len = len(H_matrix)
    codeword_len = len(H_matrix[0])
    data_len = codeword_len - redundancy_len

    print(H_matrix)

    print("-------------------------------------")
    print(" SCC Code configuration ({0}, {1})".format(codeword_len, data_len))
    print("-------------------------------------")

    #
    # 1) Error case : 1-bit error
    #
    Ncase = 0
    CE_cnt = 0
    DUE_cnt = 0
    SDC_cnt = 0
    
    for Nrun in range (0, 1000000) :
        error_pos = random.randrange(codeword_len)

        # Generate error
        codeword = np.zeros(codeword_len, dtype = int)  # original codeword (no error)
        codeword[error_pos] ^= 1  # make 1-bit error

        # Decoding
        syndrome = make_syndrome(codeword, H_matrix)
        corrected_codeword, nonzero, correctable = correction_SCC(codeword, syndrome, H_matrix, Sum_matrix)
        CE_cnt, DUE_cnt, SDC_cnt = verify(corrected_codeword, nonzero, correctable, CE_cnt, DUE_cnt, SDC_cnt)

        Ncase = Ncase + 1

    print("-------------------------------------")
    print(" Case 1 : Single bit error")
    print("   - CE  : {0} / {1} ({2}%)".format(CE_cnt, Ncase, CE_cnt * 100 / Ncase))
    print("   - DUE : {0} / {1} ({2}%)".format(DUE_cnt, Ncase, DUE_cnt * 100 / Ncase))
    print("   - SDC : {0} / {1} ({2}%)".format(SDC_cnt, Ncase, SDC_cnt * 100 / Ncase))
    print("-------------------------------------")

    #
    # 2) Error case : 1-cell error
    #
    Ncase = 0
    CE_cnt = 0
    DUE_cnt = 0
    SDC_cnt = 0

    for Nrun in range (0, 1000000) :
        error_pos = random.randrange(0, codeword_len - 1, 2)

        # Generate error
        codeword = np.zeros(codeword_len, dtype = int)  # original codeword (no error)
        codeword[error_pos] ^= 1  # make 1-cell error
        codeword[error_pos + 1] ^= 1

        # Decoding
        syndrome = make_syndrome(codeword, H_matrix)
        corrected_codeword, nonzero, correctable = correction_SCC(codeword, syndrome, H_matrix, Sum_matrix)
        CE_cnt, DUE_cnt, SDC_cnt = verify(corrected_codeword, nonzero, correctable, CE_cnt, DUE_cnt, SDC_cnt)

        Ncase = Ncase + 1

    print("-------------------------------------")
    print(" Case 2 : Single cell error")
    print("   - CE  : {0} / {1} ({2}%)".format(CE_cnt, Ncase, CE_cnt * 100 / Ncase))
    print("   - DUE : {0} / {1} ({2}%)".format(DUE_cnt, Ncase, DUE_cnt * 100 / Ncase))
    print("   - SDC : {0} / {1} ({2}%)".format(SDC_cnt, Ncase, SDC_cnt * 100 / Ncase))
    print("-------------------------------------")

    #
    # 3) Error case : Double bit error (exclude 1-cell error)
    #
    Ncase = 0
    CE_cnt = 0
    DUE_cnt = 0
    SDC_cnt = 0

    for Nrun in range (0, 1000000) :
        error_pos1 = random.randrange(codeword_len - 2)
        if error_pos1 % 2 == 1 :
            error_pos2 = random.randrange(error_pos1 + 1, codeword_len)
        else :
            error_pos2 = random.randrange(error_pos1 + 2, codeword_len)

        # Generate error
        codeword = np.zeros(codeword_len, dtype = int)  # original codeword (no error)
        codeword[error_pos1] ^= 1  # make double bit error
        codeword[error_pos2] ^= 1

        # Decoding
        syndrome = make_syndrome(codeword, H_matrix)
        corrected_codeword, nonzero, correctable = correction_SCC(codeword, syndrome, H_matrix, Sum_matrix)
        CE_cnt, DUE_cnt, SDC_cnt = verify(corrected_codeword, nonzero, correctable, CE_cnt, DUE_cnt, SDC_cnt)

        Ncase = Ncase + 1

    print("-------------------------------------")
    print(" Case 3 : Double bit error")
    print("   - CE  : {0} / {1} ({2}%)".format(CE_cnt, Ncase, CE_cnt * 100 / Ncase))
    print("   - DUE : {0} / {1} ({2}%)".format(DUE_cnt, Ncase, DUE_cnt * 100 / Ncase))
    print("   - SDC : {0} / {1} ({2}%)".format(SDC_cnt, Ncase, SDC_cnt * 100 / Ncase))
    print("-------------------------------------")

    '''
    #
    # 3) Error case : Double adjacent error
    #
    Ncase = 0
    CE_cnt = 0
    DUE_cnt = 0
    SDC_cnt = 0

    for Nrun in range (0, 1000000) :
        error_pos = random.randrange(codeword_len - 1)

        # Generate error
        codeword = np.zeros(codeword_len, dtype = int)  # original codeword (no error)
        codeword[error_pos] ^= 1  # make double adjacent error
        codeword[error_pos + 1] ^= 1

        # Decoding
        syndrome = make_syndrome(codeword, H_matrix)
        corrected_codeword, nonzero, correctable = correction_SCC(codeword, syndrome, H_matrix, Sum_matrix)
        CE_cnt, DUE_cnt, SDC_cnt = verify(corrected_codeword, nonzero, correctable, CE_cnt, DUE_cnt, SDC_cnt)

        Ncase = Ncase + 1

    print("-------------------------------------")
    print(" Case 3 : Double adjacent error")
    print("   - CE  : {0} / {1} ({2}%)".format(CE_cnt, Ncase, CE_cnt * 100 / Ncase))
    print("   - DUE : {0} / {1} ({2}%)".format(DUE_cnt, Ncase, DUE_cnt * 100 / Ncase))
    print("   - SDC : {0} / {1} ({2}%)".format(SDC_cnt, Ncase, SDC_cnt * 100 / Ncase))
    print("-------------------------------------")

    #
    # 4) Error case : Double non-adjacent error
    #
    Ncase = 0
    CE_cnt = 0
    DUE_cnt = 0
    SDC_cnt = 0

    for Nrun in range (0, 1000000) :
        error_pos1 = random.randrange(codeword_len - 2)
        error_pos2 = random.randrange(error_pos1 + 2, codeword_len)

        # Generate error
        codeword = np.zeros(codeword_len, dtype = int)  # original codeword (no error)
        codeword[error_pos1] ^= 1  # make double bit error
        codeword[error_pos2] ^= 1

        # Decoding
        syndrome = make_syndrome(codeword, H_matrix)
        corrected_codeword, nonzero, correctable = correction_SCC(codeword, syndrome, H_matrix, Sum_matrix)
        CE_cnt, DUE_cnt, SDC_cnt = verify(corrected_codeword, nonzero, correctable, CE_cnt, DUE_cnt, SDC_cnt)

        Ncase = Ncase + 1

    print("-------------------------------------")
    print(" Case 4 : Double non-adjacent error")
    print("   - CE  : {0} / {1} ({2}%)".format(CE_cnt, Ncase, CE_cnt * 100 / Ncase))
    print("   - DUE : {0} / {1} ({2}%)".format(DUE_cnt, Ncase, DUE_cnt * 100 / Ncase))
    print("   - SDC : {0} / {1} ({2}%)".format(SDC_cnt, Ncase, SDC_cnt * 100 / Ncase))
    print("-------------------------------------")
    '''

if __name__ == "__main__":
    main()

