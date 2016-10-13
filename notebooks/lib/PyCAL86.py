
# coding: utf-8

# In[1]:

from __future__ import division, print_function

# In[2]:

import numpy as np
from IPython.core.magic import register_cell_magic, register_line_cell_magic, register_line_magic
from IPython.utils import warn
from IPython.core import error
import re
np.set_printoptions(precision=5,linewidth=200,suppress=True)


# In[3]:

import io

class CAL86Error(StandardError): pass
class CAL86Warning(Warning): pass

class CAL86(object):
    
    COMMANDS = {}     # register_cal_cmd decorator adds stuff here
    
    def __init__(self,stream,echo=True):
        if type(stream) in [type(''),type(u'')]:
            stream = io.StringIO(unicode(stream),newline=None)
        else:
            if not callable(getattr(stream,'readline',None)):
                raise ValueError, 'stream does not appear to be an i/o stream'
        self.stream = stream
        self.echo = echo    # echo input command lines
        self.eol = False    # most recently echoed line ended in newline
        
        shell = get_ipython()
        self.ns = shell.user_global_ns
        
        register_cal_cmd('HELP',[],['TOPIC'])(self.help)
        
    def run(self):
        """Read and interpret commands from input stream."""
        try:
            self._dorun()
        except CAL86Error as e:
            raise error.UsageError, str(e.__class__.__name__)+': '+str(e)
        except:
            raise
            
    def split(self,line):
        mo = re.match(r'\s*([A-Za-z][A-Za-z0-9_]*)(.*)$',line)
        if mo:
            return mo.groups()
        return None,line
            
    def _dorun(self):
        for line in self._get_lines():
            if line == '':
                continue
            word,remainder = self.split(line)
            if self.iscmd(word):
                fn,req,opt = self.getcmd(word)
                args = self.parseargs(remainder,req,opt)
                fn(self,**args)
                continue
            raise CAL86Error, "Invalid command: "+line
    
    def lines(self,n):
        """Return a list of at most n lines from the input stream.
        Stop, perhaps early, when a line beginning with a command
        word is found."""
        ans = []
        for l in self._get_lines():
            ans.append(l)
            if len(ans) >= n:
                break
        return ans
        
    def nextline(self):
        """Return the next logical line from the input stream, handling
        line continuations."""
        
        
    def iscmd(self,word):
        if type(word) not in [type(''),type(u'')]:
            return False
        word = word.upper()
        return word in self.__class__.COMMANDS
    
    def getcmd(self,word):
        word = word.upper()
        return self.__class__.COMMANDS.get(word,None)
    
    def help(self,cal,TOPIC=None):
        """Display help messages"""
        if TOPIC is None or not self.iscmd(TOPIC):
            cmds = sorted(self.__class__.COMMANDS.keys())
            self.display("Available CAL commands: {}".format(', '.join(cmds)))
            return
        fn,ra,oa = self.getcmd(TOPIC)
        self.display("{}:".format(TOPIC))
        self.display(getattr(fn,'__doc__',''))
        
    def _get_lines(self):
        stream = self.stream
        while True:
            l = stream.readline()
            if l == '':
                break
            if self.echo:
                self.display(l,end="")
            line = l.rstrip()
            while line and line[-1] == '\\':
                l = stream.readline()
                if self.echo:
                    self.display(l,end="")
                line = line[:-1] + l.rstrip()
                if l == '':
                    break
            yield line
        
    def parseargs(self,line,params,optional=[]):
        if params is None and optional is None:
            return {}
        posparams = [x for x in params if type(x) in [str,unicode]]
        posoptional = [x for x in optional if type(x) in [str,unicode]]
        keyparams = {x[0]:x[1:] for x in params+optional if type(x) is tuple}
        args = re.split(r' +',line.strip())
        posargs = [x for x in args if '=' not in x]
        keyargs = [x for x in args if '=' in x]
        if len(posargs) < len(posparams):
            raise CAL86Error, "Too few positional arguments given.  Expected {}, found only: {}"                                .format(len(posparams), ' '.join(posargs))
        if len(posargs) > len(posparams)+len(posoptional):
            raise CAL86Error, "Too many positional arguments given.  Expected {}, but found: {}"                                .format(len(posparams)+len(posoptional), ' '.join(posargs))
        ans = dict(zip(posparams+posoptional,posargs))
        for ka in keyargs:
            var,val = ka.split('=')
            if var not in keyparams:
                raise CAL86Error, "Invalid keyword argument: {}".format(var)
            typ = keyparams[var]
            if len(typ) == 1:
                val = typ[0](eval(val))
                ans[var] = val
            else:
                val = val.split(',')
                if len(val) != len(typ):
                    raise CAL86Error, "Keyword argument '{}' must have {} values; values found: {}"                                        .format(var,len(typ),ka)
                val = [f(eval(s)) for f,s in zip(typ,val)]
                ans[var] = val
        for var in [x[0] for x in params if type(x) is tuple]:
            if var not in ans:
                raise CAL86Error, "Keyword argument '{}' not given.".format(var)
        return ans
    
    def display(self,s,end='\n'):
        if not self.eol:
            print('')
        print(s,end=end)
        self.eol = (end and end[-1] == '\n') or (s and s[-1] == '\n')
        
    def check(self,name,optional=False,missing=False,msg=None):
        if missing:
            if name not in self.ns:
                return True
            if optional:
                return False
            if msg is None:
                msg = "Matrix named '{}' must not exist."
            raise CAL86Error, msg.format(name)
        else:
            if name in self.ns:
                if type(self.ns[name]) is np.matrixlib.matrix:
                    return True
            if optional:
                return False
            if msg is None:
                msg = "Matrix named '{}' does not exist."
            raise CAL86Error, msg.format(name)


# In[4]:

def register_cal_cmd(cmd,required=[],optional=[]):
    def _doreg(fn,cmd=cmd,required=required,optional=optional):
        cmd = cmd.upper()
        CAL86.COMMANDS[cmd] = (fn,required,optional)
        return fn
    return _doreg


# In[5]:

@register_cell_magic('cal86')
@register_cell_magic('CAL86')
def magic_cal86(linetext,celltext):
    """Usage:
             %%CAL86 [-q|-v]
    Read and execute the follwing CAL86 commands.
    -q : do not echo commands in ouput area.
    -v : echo commands in output area.
    """
    echo = False
    for arg in re.split(r'\s+',linetext):
        if arg == '-q':
            echo = False
        elif arg == '-v':
            echo = True
        elif arg == '':
            pass
        else:
            raise CAL86Error, 'Invalid argument: {}'.format(arg)
    cal = CAL86(celltext,echo=echo)
    cal.run()


# In[6]:

@register_cal_cmd('P',['M'])
@register_cal_cmd('PRINT',['M'])
def cal_print(cal,M):
    """Print a matrix.
    
    Usage:
    
    PRINT M
    
    P M
    
    The print command will display the matrix that is the value of variable M."""
    cal.check(M)
    #print()
    cal.display("{}:".format(M))
    cal.display(cal.ns[M])


# In[8]:

@register_cal_cmd('LOAD',['M',('R',int),('C',int)])
def cal_load(cal,M,R,C):
    """Load a matrix with data.
    
Usage, in cell mode:  

  LOAD  M  R=nr  C=nc
  data ...  (nr lines)
    
The **LOAD** command will create a matrix in variable *M*, with *nr* rows and 
*nc* columns.  The data must immediately follow the LOAD command and it must be 
supplied one row of the matrix per line of data.  The column values must be 
separated by one comma, and/or one or more spaces.  A line of  data may be 
continued by use of a '\\' character at the end of each continued line."""

    lines = cal.lines(R)
    data = []
    for line in lines:
        svals = re.split(r'\s+',line.strip())
        vals = [eval(x,cal.ns) for x in svals]
        data.append(vals)
    m = np.matrix(data,dtype=np.float64)
    nr,nc = m.shape
    if nr != R:
        raise ValueError, 'Expected {} rows, found {}'.format(R,nr)
    if nc != C:
        raise ValueError, 'Expected {} columns, found {}'.format(C,nc)
    cal.ns[M] = m
    if cal.echo:
        cal.display('ARRAY NAME = {:<5} NUMBER OF ROWS = {:4d} NUMBER OF COLUMNS = {:4d}'.format(M,R,C))


# In[11]:

@register_cal_cmd('LOADI',['M',('R',int),('C',int)])
def cal_loadi(cal,M,R,C):
    """Load a matrix with integer data.
    
Usage:  

  LOADI  M  R=nr  C=nc
  data ... (nr lines)
    
The LOADI command will create a matrix in variable M, with nr rows and 
nc columns.  The data must immediately follow the LOAD command and it must be 
supplied one row of the matrix per line of data.  The column values must be 
separated by one comma, and/or one or more spaces.  A line of  data may be 
continued by use of a '\\' character at the end of each continued line.
All data values are converted to integer values."""
    
    lines = cal.lines(R)
    s = str('; '.join(lines))
    m = np.matrix(s,dtype=np.int)
    nr,nc = m.shape
    if nr != R:
        raise CAL86Error, 'Expected {} rows, found {}'.format(R,nr)
    if nc != C:
        raise CAL86Error, 'Expected {} columns, found {}'.format(C,nc)
    cal.ns[M] = m
    if cal.echo:
        cal.display('ARRAY NAME = {:<5} NUMBER OF ROWS = {:4d} NUMBER OF COLUMNS = {:4d}'.format(M,R,C))


# In[14]:

@register_cal_cmd('TRAN',['M1','M2'])
def cal_tran(cal,M1,M2):
    """Transpose a matrix.
    
    Usage:
    
    TRAN  M1  M2
    
    The matrix in variable M1 is transposed and stored in variable M2."""
    
    cal.check(M1)
    cal.ns[M2] = cal.ns[M1].T


# In[16]:

@register_cal_cmd('ADD',['M1','M2'])
def cal_add(cal,M1,M2):
    """Add two matrices.
    
    Usage:
    
    %ADD  M1  M2
    
    The matrix sum M1+M2 is formed and stored back into matrix M1."""

    cal.check(M1)
    cal.check(M2)
    m1 = cal.ns[M1]
    m2 = cal.ns[M2]
    if m1.shape != m2.shape:
        raise CAL86Error, "Shapes of '{} {}' and '{} {}' are not conformable for addition.".format(M1,m1.shape,M2,m2.shape)
    cal.ns[M1] = m1 + m2


# In[18]:

@register_cal_cmd('MULT',['M1','M2','M3'])
def cal_mult(cal,M1,M2,M3):
    """Multiply two matrices.
    
    Usage:
    
    MULT  M1  M2  M3
    
    The matrices in variables M1 and M2 are multiplied and stored in variable M3."""
    
    cal.check(M1)
    cal.check(M2)
    m1 = cal.ns[M1]
    m2 = cal.ns[M2]
    if m1.shape[1] != m2.shape[0]:
        raise CAL86Error, "Shapes of {} {} and {} {} are not conformable for multiplication".format(M1,m1.shape,M2,m2.shape)
    cal.ns[M3] = m1*m2


# In[20]:

@register_cal_cmd('TMULT',['M1','M2','M3'])
def cal_tmult(cal,M1,M2,M3):
    """Multiply two matrices.
    
    Usage:
    
    TMULT  M1  M2  M3
    
    The transpose of the matrix in M1 and the untransposed M2 are multiplied 
    and stored in variable M3."""
    
    cal.check(M1)
    cal.check(M2)
    m1 = cal.ns[M1]
    m2 = cal.ns[M2]
    if m1.shape[0] != m2.shape[0]:
        raise CAL86Error, "Shape of {} {} and {} {} are not conformable for transposed multiplication".format(M1,m1.shape,M2,m2.shape)
    cal.ns[M3] = m1.T*m2


@register_cal_cmd('TTMULT',['M1','M2','M3'])
def cal_ttmult(cal,M1,M2,M3):
    """Multiply two matrices.
    
    Usage:
    
    TTMULT  M1  M2  M3
    
    The matrix product M1' * M2 * M1 is formed
    and stored in variable M3."""
    
    cal.check(M1)
    cal.check(M2)
    m1 = cal.ns[M1]
    m2 = cal.ns[M2]
    if m1.shape[0] != m2.shape[0] or m1.shape[0] != m2.shape[1]:
        raise CAL86Error, "Shape of {} {} and {} {} are not conformable for transposed multiplication".format(M1,m1.shape,M2,m2.shape)
    cal.ns[M3] = m1.T*m2*m1


# In[22]:

@register_cal_cmd('ZERO',['M',('R',int),('C',int)],[('T',float),('D',float)])
def cal_zero(cal,M,R,C,T=0.0,D=None):
    """Set a matrix to given values (default 0).
    
    Usage:  

    ZERO  M  R=nr  C=nc  [T=t]  [D=d]
    
    The **ZERO** command creates a matrix of size nr x nc.  If t is specified, all elements
    will be set to this value (the default value of t is 0 (zero)).  If d is specified and the
    matrix is square (nr=nc), the diagonal values will be set to this (the default value of d
    is t)"""
    
    if D is None:
        D = T
    m = np.matrix(np.zeros((R,C)))
    if T != 0.:
        m[:,:] = T
    if D != T and R == C:
        np.fill_diagonal(m,D)
    cal.ns[M] = m


# In[24]:

@register_cal_cmd('FRAME',['K','T',('I',float),('A',float),('E',float),
                           ('X',float,float),('Y',float,float)])
def cal_frame(cal,K,T,I,A,E,X,Y):
    """Form a 6x6 element stiffness matrix in global coordinates.
    
    Usage:
    
    FRAME  K  T  I=Ix  A=A  E=E  X=xj,xk  Y=yj,yk
    
    The FRAME command forms the 6x6 element stiffness matrix, K, 
    and a 4x6 force-displacement matrix T for a general two-dimensional 
    bending member with axial deformations included in the formulation.  
    The properties of the member are given as: Ix = the moment of inertia 
    of the member, and A = the cross-sectional area of the member, and
    E = the Modulus of Elasticity of the member.
  
    The coordinates of the "j" and "k" ends of the member are defined by 
    xj,xk and yj,yk respectively.  Note that the user is responsible for 
    the establishment of the j and k ends of the member."""
    
    xj,xk = X
    yj,yk = Y
    dx = xk-xj
    dy = yk-yj
    L = (dx*dx + dy*dy)**0.5
    cx = dx/L
    cy = dy/L
    k0 = E*A/L
    k12 = 12.*E*I/L**3
    k6 = 6.*E*I/L**2
    k4 = 4.*E*I/L
    k2 = 2.*E*I/L
    KL = np.mat([[ k0,  0,    0,   -k0,   0,    0],
                 [ 0,   k12,  k6,   0,   -k12,  k6],
                 [ 0,   k6,   k4,   0,   -k6,   k2],
                 [-k0,  0,    0,    k0,   0,    0],
                 [ 0,  -k12, -k6,   0,    k12, -k6],
                 [ 0,   k6,   k2,   0,   -k6,   k4]])
    Tm = np.mat([[ cx,  cy,  0,   0,   0,   0],
                 [-cy,  cx,  0,   0,   0,   0],
                 [ 0,   0,   1,   0,   0,   0],
                 [ 0,   0,   0,   cx,  cy,  0],
                 [ 0,   0,   0,  -cy,  cx,  0],
                 [ 0,   0,   0,   0,   0,   1]])
    KG = Tm.T * KL * Tm
    cal.ns[K] = KG
    f2a = np.mat([[0, 0, 1, 0, 0, 0.],  # translate 6 member directions to 4
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0, 0]])
    cal.ns[T] = f2a * KL * Tm


@register_cal_cmd('TRUSS',['K','T',('A',float),('E',float),
                           ('X',float,float),('Y',float,float)])
def cal_truss(cal,K,T,A,E,X,Y):
    """Form a 4x4 2D truss element stiffness matrix in global coordinates.
    
    Usage:
    
    TRUSS  K  T  A=A  E=E  X=xj,xk  Y=yj,yk
    
    The TRUSS2D command forms the 4x4 element stiffness matrix, K, 
    and a 4x4 force displacement matrix T for a general two-dimensional 
    truss member with only axial deformations included in the formulation.  
    The properties of the member are given as: A = the cross-sectional area 
    of the member, and E = the Modulus of Elasticity of the member.
  
    The coordinates of the "j" and "k" ends of the member are defined by 
    xj,xk and yj,yk respectively.  Note that the user is responsible for 
    the establishment of the j and k ends of the member."""
    
    xj,xk = X
    yj,yk = Y
    dx = xk-xj
    dy = yk-yj
    L = (dx*dx + dy*dy)**0.5
    cx = dx/L
    cy = dy/L
    k = 1.0*E*A/L
    KL = np.mat([[ k,  0,   -k,   0 ],
                 [ 0,  0,    0,   0 ],
                 [-k,  0,    k,   0 ],
                 [ 0,  0,    0,   0 ],
                 ])
    Tm = np.mat([[ cx,  cy,  0,   0 ],
                 [-cy,  cx,  0,   0 ],
                 [ 0,   0,  cx,  cy ],
                 [ 0,   0, -cy,  cx ],
                 ])
    KG = Tm.T * KL * Tm
    cal.ns[K] = KG
    cal.ns[T] = KL * Tm


# In[26]:

@register_cal_cmd('ADDK',['K','EK','ID',('N',int)])
def cal_addk(cal,K,EK,ID,N):
    """Add Element Stiffness to Global Stiffness.
    
    Usage:
    
    ADDK  K  EK  ID  N=n
    
    The element stiffness matrix EK is added in to the total stiffness matrix K . 
    The row and column numbers where the terms are to be added are obtained from 
    column n of the  L×m integer matrix ID (where  m  is the total number of members
    and L is the size of EK and the number of rows in ID).
    """
    cal.check(K)
    cal.check(EK)
    cal.check(ID)
    k = cal.ns[K]
    ek = cal.ns[EK]
    id = cal.ns[ID]
    if ek.shape[0] != ek.shape[1]:
        raise CAL86Error, "Matrix '{}' must be square.  It is {}x{}".format(EK,*ek.shape)
    if id.shape[0] != ek.shape[0] or id.dtype != np.int:
        raise CAL86Error, "Matrix '{}' must be an integer matrix with {} rows".format(ID,ek.shape[0])
    g = list(id[:,N].flat)
    k[np.ix_(g,g)] += ek


# In[41]:

import numpy.linalg as la

@register_cal_cmd('PSOLVE',['A','D','P'],[('PS',int)])
def cal_psolve(cal,A,D,P,PS=None):
    """Solve a (possibly partitioned) set of linear equations.
    
    Usage:
    
    PSOLVE  A  D  P  PS=p
    
    The command PSOLVE solves the partitioned linear set of equations A D = P 
    and places the result in matrix D. If PS=p is given, then p is 
    the unconstrained partion size (the number of unconstrained degrees
    of freedom), and the matrices are assummed to be partionable as:

    [A] = [[Auu][Auc]     D = [[Du]       P = [[Pu]
           [Acu][Acc]]         [Dc]]           [Pc]]

    In this case,  partition Auu Du + Auc Dc = Pu is solved for Du, 
    with given Dc (constrained displacements - normally 0). For a partioned 
    solution, Pc (the support reactions) are determined from: 
    Pc = Acu Du + Acc Dc.
"""
    
    cal.check(A)
    cal.check(P)
    mA = cal.ns[A]
    mP = cal.ns[P]
    p = PS
    if mA.shape[0] != mA.shape[1]:
        raise CAL86Error, "Stiffness matrix '{}' must be square.".format(A)
    if mA.shape[0] != mP.shape[0]:
        raise CAL86Error, "Load matrix '{}' must have the same number of rows ({}) as '{}'".format(P,mA.shape[0],A)
    if p and p > mA.shape[0]:
        raise CAL86Error, "Partion size ({}) is larger than the maximum ({})".format(p,mA.shape[0])
    if p is None or p == mA.shape[0]:
        cal.ns[D] = la.solve(mA,mP)
        return

    cal.check(D,msg="Matrix '{}' must pre-exist for a partioned solution")
    mD = cal.ns[D]
    if mA.shape[0] != mD.shape[0]:
        raise CAL86Error, "Displacement matrix '{}' must have the same number of rows ({}) as '{}'".format(D,mA.shape[0],A)
    Auu = mA[:p,:p]
    Auc = mA[:p,p:]
    Acu = mA[p:,:p]
    Acc = mA[p:,p:]
    Pu = mP[:p,:]
    mD = cal.ns[D]
    Dc = mD[p:,:]
    Du = la.solve(Auu,Pu-Auc*Dc)
    Pc = Acu*Du + Acc*Dc
    mD[:p,:] = Du
    mP[p:,:] = Pc

@register_cal_cmd('SOLVE',['A','B'],[('S',int)])
def cal_solve(A,B,S=0):
    """Solve a set of linear equations.
    
    Usage:
    
    SOLVE  A  B  S=0
    
    The command SOLVE solves the set of equations A x = B for x and places the
    result x back into the matrix B.  A is not modified.   S is provided only for
    compatibility with CAL86, and S=0 is the only allowed value.  S is optional.
"""

    cal.check(A)
    cal.check(B)
    mA = cal.ns[A]
    mB = cal.ns[B]
    if S != 0:
        raise CAL86Error, 'Only S=0 is supported.'
    if mA.shape[0] != mA.shape[1]:
        raise CAL86Error, "Stiffness matrix '{}' must be square.".format(A)
    if mA.shape[0] != mB.shape[0]:
        raise CAL86Error, "Load matrix '{}' must have {} rows.".format(B,mA.shape[0])
    x = la.solve(mA,mB)
    cal.ns[B] = x

# In[45]:

@register_cal_cmd('MEMFRC',['T','U','ID','P',('N',int)])
def cal_memfrc(cal,T,U,ID,P,N):
    """Evaluate member end forces.
    
    Usage:
    
    MEMFRC  T  U  ID  P  N=n
    
    The member end forces are evaluated by multiplying the matrix T 
    by the joint displacements U and storing the results in matrix P.  
    The joint displacements that are used are obtained from column n 
    of integer array ID.  If T is the 6x6 element stiffness matrix 
    returned in the first matrix of the FRAME command, the forces 
    are given according to the global coordinate system. If T is the 
    4x6 force-displacement transformation matrix returned in the second 
    matrix of the FRAME command, the forces will be given in a simplified, 
    4-element local coordinate system.
    """
    cal.check(T)
    cal.check(U)
    cal.check(ID)
    mT,mU,mID = [cal.ns[x] for x in [T,U,ID]]
    if mT.shape[1] != mID.shape[0]:
        raise CAL86Error, "# of cols in '{}' and rows in '{}' must be equal".format(T,ID)
    n = N
    g = list(mID[:,n].flat)
    mP = mT*mU[g,:]
    cal.ns[P] = mP


# In[ ]:

@register_cal_cmd('C',None,None)
def cal_c(cal):
    """A comment.
    
    Usage:
    
    C any text can appear here.
    
    The line is ignored."""
    pass


# In[ ]:



