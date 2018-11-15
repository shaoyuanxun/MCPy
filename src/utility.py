import numpy as np

def eq_mul(IA1, IA2, MC1, MC2, SG1, SG2):

    min_value = min(IA1[0]*IA2[0], IA1[0]*IA2[1], IA1[1]*IA2[0], IA1[1]*IA2[1])
    max_value = max(IA1[0]*IA2[0], IA1[0]*IA2[1], IA1[1]*IA2[0], IA1[1]*IA2[1])

    alpha1 = min(IA2[0]*MC1[0], IA2[0]*MC1[1])
    alpha2 = min(IA1[0]*MC2[0], IA1[0]*MC2[1])
    beta1  = min(IA2[1]*MC1[0], IA2[1]*MC1[1])
    beta2  = min(IA1[1]*MC2[0], IA1[1]*MC2[1])
    gamma1 = max(IA2[0]*MC1[0], IA2[0]*MC1[1])
    gamma2 = max(IA1[1]*MC2[0], IA1[1]*MC2[1])
    delta1 = max(IA2[1]*MC1[0], IA2[1]*MC1[1])
    delta2 = max(IA1[0]*MC2[0], IA1[0]*MC2[1])

    cv = max(min_value, max(alpha1+alpha2-IA1[0]*IA2[0], beta1+beta2-IA1[1]*IA2[1]))
    cc = min(max_value, min(gamma1+gamma2-IA1[1]*IA2[0], delta1+delta2-IA1[0]*IA2[1]))

    if IA2[0] >= 0:
        sg_alpha1 = IA2[0]*SG1[:,0]
    else:
        sg_alpha1 = IA2[0]*SG1[:,1]

    if IA1[0] >= 0:
        sg_alpha2 = IA1[0]*SG2[:,0]
    else:
        sg_alpha2 = IA1[0]*SG2[:,1]

    if IA2[1] >= 0:
        sg_beta1 = IA2[1]*SG1[:,0]
    else:
        sg_beta1 = IA2[1]*SG1[:,1]

    if IA1[1] >= 0:
        sg_beta2 = IA1[1]*SG2[:,0]
    else:
        sg_beta2 = IA1[1]*SG2[:,1]

    if IA2[0] >= 0:
        sg_gamma1 = IA2[0]*SG1[:,1]
    else:
        sg_gamma1 = IA2[0]*SG1[:,0]

    if IA1[1] >= 0:
        sg_gamma2 = IA1[1]*SG2[:,1]
    else:
        sg_gamma2 = IA1[1]*SG2[:,0]

    if IA2[1] >= 0:
        sg_delta1 = IA2[1]*SG1[:,1]
    else:
        sg_delta1 = IA2[1]*SG1[:,0]

    if IA1[0] >= 0:
        sg_delta2 = IA1[0]*SG2[:,1]
    else:
        sg_delta2 = IA1[0]*SG2[:,0]

    n = len(SG1[:,0])
    if min_value > max(alpha1+alpha2-IA1[0]*IA2[0], beta1+beta2-IA1[1]*IA2[1]):
        SG_cv = np.zeros((n,))            
    elif alpha1+alpha2-IA1[0]*IA2[0] >= beta1+beta2-IA1[1]*IA2[1]:
        SG_cv = sg_alpha1 + sg_alpha2
    else:
        SG_cv = sg_beta1 + sg_beta2

    if max_value < min(gamma1+gamma2-IA1[1]*IA2[0], delta1+delta2-IA1[0]*IA2[1]):
        SG_cc = np.zeros((n,))
    elif gamma1+gamma2-IA1[1]*IA2[0] <= delta1+delta2-IA1[0]*IA2[1]:
        SG_cc = sg_gamma1 + sg_gamma2
    else:
        SG_cc = sg_delta1 + sg_delta2
        
    return min_value, max_value, cv, cc, SG_cv, SG_cc

