import math
import numpy as np 
    
class MCPy:
    '''
    MCPy is a python library for automatically calculate convex\concave relaxations and subgradients 
    of factorable nonconvex functions according to McCormick relaxation rules and interval arithmetic. 
    '''
 
    def __init__(self, IA, MC, SG):
        '''
        Initialization:
        MCPy.IA
        1-D numpy array of two elements [LB, UB]. 
        LB/UB are the lower/upper bound the function calculated by the intervarl arithmetic. 
        MCPy.MC 
        1-D numpy array of two elements [cv, cc]. 
        cv/cc are the convex underestimator/concave overestimator of the function calculated by the McCormick rules. 
        MCPy.SG 
        2-D numpy n-by-2 matrix [SG_cv,SG_cc]. 
        SG_cv/SG_cc are n-by-1 column verctors of subgradients for convex/concave relaxations.
        '''
        
        self.IA = IA
        self.MC = MC
        self.SG = SG
        
    def __add__(self, MCPy2):
        if type(MCPy2) == MCPy:
            IA = self.IA + MCPy2.IA
            MC = self.MC + MCPy2.MC
            SG = self.SG + MCPy2.SG
            return MCPy( IA, MC, SG )
        else:
            return MCPy(self.IA+[MCPy2,MCPy2], self.MC+[MCPy2,MCPy2], self.SG)
    
    def __radd__(self, MCPy2):
        if type(MCPy2) != MCPy:
            return self+MCPy2
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return self*(-1)
    
    def __sub__(self, MCPy2):
        if type(MCPy2) == MCPy:
            return self+MCPy2*(-1)
        else:
            return MCPy(self.IA-[MCPy2,MCPy2], self.MC-[MCPy2,MCPy2], self.SG)
        
    def __rsub__(self, MCPy2):
        if type(MCPy2) != MCPy:
            return self*(-1)+MCPy2
    
    def __pow__(self, power):
        if power % 2 == 0 and power > 0:
            LB = mid(self.IA[0], self.IA[1], 0)**power
            UB = max(self.IA[0]**power, self.IA[1]**power)
            IA = np.array([LB, UB])
            
            xmin = mid(self.IA[0], self.IA[1], 0)
            xmax = max(abs(self.IA[0]), abs(self.IA[1]))
            cv_arg = mid(self.MC[0], self.MC[1], xmin)
            cc_arg = mid(self.MC[0], self.MC[1], xmax)
            cv = max(IA[0], cv_arg**power)
            cc = min(IA[1], self.IA[0]**power + (self.IA[1]**power - self.IA[0]**power)/(self.IA[1]-self.IA[0])*(cc_arg-self.IA[0]))
            MC = np.array([cv, cc])
            
            sigma_uu = power*self.MC[0]**(power-1)
            sigma_ou = (self.IA[1]**power - self.IA[0]**power)/(self.IA[1]-self.IA[0])
            sigma_uo = power*self.MC[1]**(power-1)
            sigma_oo = (self.IA[1]**power - self.IA[0]**power)/(self.IA[1]-self.IA[0])
            
            n = len(self.SG[:,0])
            if IA[0] > cv_arg**power:
                SG_cv = np.zeros((n,1))
            elif xmin > self.MC[1]:
                SG_cv = sigma_uo*self.SG[:,1]
            elif xmin < self.MC[0]:
                SG_cv = sigma_uu*self.SG[:,0]
            else:
                SG_cv = np.asmatrix(np.zeros((n,1)))
                
            if IA[1] < self.MC[0]**power + (self.IA[1]**power - self.IA[0]**power)/(self.IA[1]-self.IA[0])*(cc_arg-self.MC[0]):
                SG = np.hstack((SG_cv, np.zeros((n,1))))
            elif xmax > self.MC[1]:
                SG = np.hstack((SG_cv, sigma_oo*self.SG[:,1]))
            elif xmax < self.MC[0]:
                SG = np.hstack((SG_cv, sigma_ou*self.SG[:,0]))
            else:
                SG = np.hstack((SG_cv, np.zeros((n,1))))
                
            return MCPy(IA, MC, SG) 
        
        elif power == -1:
            if self.IA[0] <= 0:
                raise ValueError('This power rule does not support nonpositive domains.')
            values = [1/self.IA[0], 1/self.IA[1]]
            IA = np.array([min(values), max(values)])
                      
            xmin = self.IA[1]
            xmax = self.IA[0]
            cv_arg = mid(self.MC[0],self.MC[1],xmin)
            cc_arg = mid(self.MC[0],self.MC[1],xmax)
            cv = max(IA[0],1/cv_arg)
            cc = min(IA[1],(1/self.IA[1]-1/self.IA[0])/(self.IA[1]-self.IA[0])*(cc_arg-self.IA[0])+1/self.IA[0])
            MC = np.array([cv, cc])

            sigma_uu = -1/self.MC[0]**2
            sigma_ou = (1/self.IA[1]-1/self.IA[0])/(self.IA[1]-self.IA[0])
            sigma_uo = -1/self.MC[1]**2
            sigma_oo = (1/self.IA[1]-1/self.IA[0])/(self.IA[1]-self.IA[0])

            n = len(self.SG[:,0])
            if IA[0] > 1/cv_arg:
                SG_cv = np.zeros((n,1))
            elif xmin > self.MC[1]:
                SG_cv = sigma_uo*self.SG[:,1]
            elif xmin < self.MC[0]:
                SG_cv = sigma_uu*self.SG[:,0]
            else:
                SG_cv = np.zeros((n,1))
            
            if IA[1] < (1/self.IA[1]-1/self.IA[0])/(self.IA[1]-self.IA[0])*(cc_arg-self.IA[0])+1/self.IA[0]:
                SG_cc = np.zeros((n,1))
            elif xmax > self.MC[1]:
                SG_cc = sigma_oo*self.SG[:,1]
            elif xmax < self.MC[0]:
                SG_cc = sigma_ou*self.SG[:,0]
            else:
                SG_cc = np.zeros((n,1))
            SG = np.asmatrix(np.hstack((SG_cv, SG_cc)))
            
            return MCPy(IA, MC, SG)                         
        
        elif power == 1:
            return self
        
        elif power % 2 == 1 and power > 2:        
            return self*self**(power-1)
        
        elif power % 2 == 1 and power < -1:
            temp = self**(-1)
            return temp**(-power)
        
        elif power % 2 == 0 and power < -1:
            temp = self**(-1)
            return temp**(-power)
        
        else:
            raise ValueError('This power rule is not supported yet.')
            
    def __mul__(self, MCPy2):
        if type(MCPy2) == MCPy:
            values = [self.IA[0]*MCPy2.IA[0], self.IA[0]*MCPy2.IA[1], self.IA[1]*MCPy2.IA[0], self.IA[1]*MCPy2.IA[1]]
            IA = np.array([min(values), max(values)])

            alpha1 = min(MCPy2.IA[0]*self.MC[0], MCPy2.IA[0]*self.MC[1])
            alpha2 = min(self.IA[0]*MCPy2.MC[0], self.IA[0]*MCPy2.MC[1])
            beta1  = min(MCPy2.IA[1]*self.MC[0], MCPy2.IA[1]*self.MC[1])
            beta2  = min(self.IA[1]*MCPy2.MC[0], self.IA[1]*MCPy2.MC[1])
            gamma1 = max(MCPy2.IA[0]*self.MC[0], MCPy2.IA[0]*self.MC[1])
            gamma2 = max(self.IA[1]*MCPy2.MC[0], self.IA[1]*MCPy2.MC[1])
            delta1 = max(MCPy2.IA[1]*self.MC[0], MCPy2.IA[1]*self.MC[1])
            delta2 = max(self.IA[0]*MCPy2.MC[0], self.IA[0]*MCPy2.MC[1])

            cv = max(IA[0], max(alpha1+alpha2-self.IA[0]*MCPy2.IA[0], beta1+beta2-self.IA[1]*MCPy2.IA[1]))
            cc = min(IA[1], min(gamma1+gamma2-self.IA[1]*MCPy2.IA[0], delta1+delta2-self.IA[0]*MCPy2.IA[1]))
            MC = np.array([cv, cc])

            if MCPy2.IA[0] >= 0:
                sg_alpha1 = MCPy2.IA[0]*self.SG[:,0]
            else:
                sg_alpha1 = MCPy2.IA[0]*self.SG[:,1]

            if self.IA[0] >= 0:
                sg_alpha2 = self.IA[0]*MCPy2.SG[:,0]
            else:
                sg_alpha2 = self.IA[0]*MCPy2.SG[:,1]

            if MCPy2.IA[1] >= 0:
                sg_beta1 = MCPy2.IA[1]*self.SG[:,0]
            else:
                sg_beta1 = MCPy2.IA[1]*self.SG[:,1]

            if self.IA[1] >= 0:
                sg_beta2 = self.IA[1]*MCPy2.SG[:,0]
            else:
                sg_beta2 = self.IA[1]*MCPy2.SG[:,1]

            if MCPy2.IA[0] >= 0:
                sg_gamma1 = MCPy2.IA[0]*self.SG[:,1]
            else:
                sg_gamma1 = MCPy2.IA[0]*self.SG[:,0]

            if self.IA[1] >= 0:
                sg_gamma2 = self.IA[1]*MCPy2.SG[:,1]
            else:
                sg_gamma2 = self.IA[1]*MCPy2.SG[:,0]

            if MCPy2.IA[1] >= 0:
                sg_delta1 = MCPy2.IA[1]*self.SG[:,1]
            else:
                sg_delta1 = MCPy2.IA[1]*self.SG[:,0]

            if self.IA[0] >= 0:
                sg_delta2 = self.IA[0]*MCPy2.SG[:,1]
            else:
                sg_delta2 = self.IA[0]*MCPy2.SG[:,0]

            if alpha1+alpha2-self.IA[0]*MCPy2.IA[0] >= beta1+beta2-self.IA[1]*MCPy2.IA[1]:
                SG_cv = sg_alpha1 + sg_alpha2
            else:
                SG_cv = sg_beta1 + sg_beta2

            if gamma1+gamma2-self.IA[1]*MCPy2.IA[0] <= delta1+delta2-self.IA[0]*MCPy2.IA[1]:
                SG_cc = sg_gamma1 + sg_gamma2
            else:
                SG_cc = sg_delta1 + sg_delta2

            n = len(self.SG[:,0])
            if IA[0] > cv:
                SG_cv = np.zeros((n,1))
            elif IA[1] < cc:
                SG_cc = np.zeros((n,1))
            SG = np.asmatrix(np.hstack((SG_cv, SG_cc)))

            return MCPy(IA, MC, SG)  
        
        elif MCPy2>=0:
            return MCPy(self.IA*MCPy2, self.MC*MCPy2, self.SG*MCPy2)
        
        elif MCPy2<0:
            IA = MCPy2*swap_elements(np.copy(self.IA), 0, 1)
            MC = MCPy2*swap_elements(np.copy(self.MC), 0, 1)
            SG = MCPy2*swap_columns(np.matrix.copy(self.SG), 0, 1)
            return MCPy(IA, MC, SG)
        
        else:
            raise ValueError('This rule is not defined yet.')

    def __rmul__(self, MCPy2):
        return self*MCPy2
            
    def __truediv__(self, MCPy2):
        return self*MCPy2**(-1)
        
    def __rtruediv__(self, MCPy2):
        return self**(-1)*MCPy2

        
def swap_columns(my_matrix, index1, index2):
    temp = np.matrix.copy(my_matrix[:,index1])
    my_matrix[:,index1] = my_matrix[:,index2]
    my_matrix[:,index2] = temp
    return my_matrix

def swap_elements(my_array, index1, index2):
    temp = np.copy(my_array[index1])
    my_array[index1] = my_array[index2]
    my_array[index2] = temp
    return my_array   

def mid(a, b ,c):
    return max(a, min(b,c))

def log(self): 
    if type(self) == MCPy:
        IA = np.array([log(self.IA[0]), log(self.IA[1])])
        
        xmin = self.IA[0]
        xmax = self.IA[1]
        cv_arg = mid(self.MC[0],self.MC[1],xmin)
        cc_arg = mid(self.MC[0],self.MC[1],xmax)      
        cv = max(IA[0],(math.log(self.IA[1])-math.log(self.IA[0]))/(self.IA[1]-self.IA[0])*(cv_arg-self.IA[0])+math.log(self.IA[0]))
        cc = min(IA[1],math.log(cc_arg))
        MC = np.array([cv, cc])
        
        sigma_uu = (math.log(self.IA[1])-math.log(self.IA[0]))/(self.IA[1]-self.IA[0])
        sigma_ou = 1/self.MC[0]
        sigma_uo = (math.log(self.IA[1])-math.log(self.IA[0]))/(self.IA[1]-self.IA[0])
        sigma_oo = 1/self.MC[1]

        n = len(self.SG[:,0])
        if IA[0] > (math.log(self.IA[1])-math.log(self.IA[0]))/(self.IA[1]-self.IA[0])*(cv_arg-self.IA[0])+math.log(self.IA[0]):
            SG_cv = np.zeros((n,1))
        elif xmin > self.MC[1]:
            SG_cv = sigma_uo*self.SG[:,1]
        elif xmin < self.MC[0]:
            SG_cv = sigma_uu*self.SG[:,0]
        else:
            SG_cv = np.zeros((n,1))

        if IA[1] < math.log(cc_arg):
            SG_cc = np.zeros((n,1))
        elif xmax > self.MC[1]:
            SG_cc = sigma_oo*self.SG[:,1]
        elif xmax < self.MC[0]:
            SG_cc = sigma_ou*self.SG[:,0]
        else:
            SG_cc = np.zeros((n,1))         
        SG = np.asmatrix(np.hstack((SG_cv, SG_cc)))
            
        return MCPy(IA, MC, SG)    
    
    else:
        return math.log(self)


def sqrt(self): 
    if type(self) == MCPy:
        IA = np.array([math.sqrt(self.IA[0]), math.sqrt(self.IA[1])])

        xmin = self.IA[0]
        xmax = self.IA[1]
        cv_arg = mid(self.MC[0],self.MC[1],xmin)
        cc_arg = mid(self.MC[0],self.MC[1],xmax)
        cv = max(IA[0],(math.sqrt(self.IA[1])-math.sqrt(self.IA[0]))/(self.IA[1]-self.IA[0])*(cv_arg-self.IA[0])+math.sqrt(self.IA[0]))
        cc = min(IA[1],math.sqrt(cc_arg))
        MC = np.array([cv, cc])

        sigma_uu = (math.sqrt(self.IA[1])-math.sqrt(self.IA[0]))/(self.IA[1]-self.IA[0])
        sigma_ou = self.MC[0]**(-1/2)/2
        sigma_uo = (math.sqrt(self.IA[1])-math.sqrt(self.IA[0]))/(self.IA[1]-self.IA[0])
        sigma_oo = self.MC[1]**(-1/2)/2

        n = len(self.SG[:,0])
        if IA[0] > (math.sqrt(self.IA[1])-math.sqrt(self.IA[0]))/(self.IA[1]-self.IA[0])*(cv_arg-self.IA[0])+math.sqrt(self.IA[0]):
            SG_cv = np.zeros((n,1))
        elif xmin > self.MC[1]:
            SG_cv = sigma_uo*self.SG[:,1]
        elif xmin < self.MC[0]:
            SG_cv = sigma_uu*self.SG[:,0]
        else:
            SG_cv = np.zeros((n,1))        

        if IA[1] < math.sqrt(cc_arg):
            SG_cc = np.zeros((n,1))
        elif xmax > self.MC[1]:
            SG_cc = sigma_oo*self.SG[:,1]
        elif xmax < self.MC[0]:
            SG_cc = sigma_ou*self.SG[:,0]
        else:
            SG_cc = np.zeros((n,1))
        SG = np.asmatrix(np.hstack((SG_cv, SG_cc)))

        return MCPy(IA, MC, SG) 
    
    else:
        return math.sqrt(self)
    
def exp(self):
    if type(self) == MCPy:
        IA = np.array([math.exp(self.IA[0]), math.exp(self.IA[1])])

        xmin = self.IA[0]
        xmax = self.IA[1]
        cv_arg = self.MC[0]
        cc_arg = self.MC[1]       
        cv = max(IA[0], math.exp(cv_arg))
        cc = min(IA[1], (math.exp(self.IA[1])-math.exp(self.IA[0]))/(self.IA[1]-self.IA[0])*(cc_arg-self.IA[0])+math.exp(self.IA[0]))
        MC = np.array([cv, cc])

        sigma_uu = math.exp(self.MC[0])
        sigma_ou = (math.exp(self.IA[1])-math.exp(self.IA[0]))/(self.IA[1]-self.IA[0])
        sigma_uo = math.exp(self.MC[1])
        sigma_oo = (math.exp(self.IA[1])-math.exp(self.IA[0]))/(self.IA[1]-self.IA[0])

        n = len(self.SG[:,0])
        if IA[0] == cv:
            SG_cv = np.zeros((n,1))
        elif xmin > self.MC[1]:
            SG_cv = sigma_uo*self.SG[:,1]
        elif xmin < self.MC[0]:
            SG_cv = sigma_uu*self.SG[:,0]
        else:
            SG_cv = np.zeros((n,1))        

        if IA[1] == cc:
            SG_cc = np.zeros((n,1))
        elif xmax > self.MC[1]:
            SG_cc = sigma_oo*self.SG[:,1]
        elif xmax < self.MC[0]:
            SG_cc = sigma_ou*self.SG[:,0]
        else:
            SG_cc = np.zeros((n,1))
        SG = np.asmatrix(np.hstack((SG_cv, SG_cc)))

        return MCPy(IA, MC, SG) 
    
    else:
        return math.exp(self)
    
