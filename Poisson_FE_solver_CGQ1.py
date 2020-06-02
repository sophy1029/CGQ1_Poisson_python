import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

'''
This program implements a Poisson equation finite element solver
by continuous Galerkin (CG) Q^1 finite element method (CGQ1).
Since by using CGQ1, then the total degrees of freedom is
4*(Number of elements).
The right hand side of the Poisson equation is give
$f = 4*(-y^2+y)*sin(pi*x)$.
The unit square domain only has homogeneous Dirichlet boundary conditions.

This program first defines some functions needed for the mesh construction,
Gaussian Quadrature points, and a function for local matrices constuction.

After mesh info is ready, we assemble local matrices and assemble them
into the global system.

In the end, we show a graph of the numerical solution.
'''


# This function will be used to constuct mesh and matrices used for the system.
def sparse(i, j, v, m, n):
    sparse_matrix = coo_matrix((v, (i,j)), shape = (m, n)).toarray()

    return sparse_matrix

# mesh generator
def RectDom_RectMesh_GenUnfm(xa,xb,nx,yc,yd,ny):

    NumNds = (nx+1)*(ny+1)
    NumEms = nx*ny

    x = np.linspace(xa,xb,nx+1)
    y = np.linspace(yc,yd,ny+1)
    X,Y = np.meshgrid(x,y,sparse=True)
    mesh = np.array(np.meshgrid(x,y))
    hx = float((xb-xa))/nx
    hy = float((yd-yc))/ny

    # Node coordinates of the mesh.
    nodes = mesh.T.reshape(-1, 2)

    a = np.arange((nx+1)*(ny+1))
    b = np.arange((nx+1)*ny)
    c = np.arange(nx*(ny+1))
    LblNd = np.reshape(a,(ny+1,nx+1))
    LblEgVert = np.reshape(b,(ny+1,nx))
    LblEgHori = np.reshape(c,(ny,nx+1))

    # Vertices' indices of each element.
    elem = np.zeros((NumEms,4),dtype=int)
    lbl = LblNd[0:ny,0:nx]
    elem[:,0] = np.ravel(lbl)
    elem[:,1] = elem[:,0] + (ny+1)
    elem[:,2] = elem[:,0] + (ny+2)
    elem[:,3] = elem[:,0] + 1

    # Separate edges as vertical edges and horizonal edges.
    NumEgsVert = (nx+1)*ny
    NumEgsHori = nx*(ny+1)
    NumEgs = NumEgsVert + NumEgsHori

    edge = np.zeros((NumEgs,2),dtype=int)
    lbl = LblNd[0:(nx+1),0:ny]
    edge[0:NumEgsVert,0] = np.ravel(lbl)
    edge[0:NumEgsVert,1] = edge[0:NumEgsVert,0]+1

    lbl = LblNd[0:nx,0:ny+1]
    edge[NumEgsVert:NumEgs,0] = np.ravel(lbl)
    edge[NumEgsVert:NumEgs,1] = edge[NumEgsVert:NumEgs,0] + (ny+1)

    # Each element has 4 edges. Label their edges.
    elem2edge = np.zeros((NumEms,4),dtype=int)
    lbl = LblEgVert[0:ny,0:nx]
    elem2edge[:,3] = np.ravel(lbl)
    elem2edge[:,1] = elem2edge[:,3] + ny
    lbl = LblEgHori[0:ny,0:nx]
    elem2edge[:,0] = NumEgsVert + np.ravel(lbl)
    elem2edge[:,2] = elem2edge[:,0] + 1;

    edge2elem = np.full((NumEgs,2),-1,dtype=int)
    CntEmsEg = np.full((NumEgs,1),-1,dtype=int)
    for ie in range(NumEms):
        LblEg = elem2edge[ie,0:4]
        CntEmsEg[LblEg] = CntEmsEg[LblEg]+1
        for k in range(4):
            edge2elem[LblEg[k],CntEmsEg[LblEg[k]]-1] = ie
    ig = np.where(edge2elem[:,0]>edge2elem[:,1])
    tmp = edge2elem[ig,0];
    edge2elem[ig,0] = edge2elem[ig,1]
    edge2elem[ig,1] = tmp

    ig = np.where(edge2elem[:,0]==-1);
    edge2elem[ig,0] = edge2elem[ig,1];
    edge2elem[ig,1] = -1;


    # Will be used for indices of boundaries.
    BndryEdge = np.full((NumEgs,1),-1,dtype=int)
    for i in range(NumEgs):
        if edge2elem[i,1] > -1:
            continue
        x1 = nodes[edge[i,0],0]
        x2 = nodes[edge[i,1],0]
        y1 = nodes[edge[i,0],1]
        y2 = nodes[edge[i,1],1]
        for j in range(len(BndryEdge)):
            X1 = BndryDescMat[j,0]
            Y1 = BndryDescMat[j,1]
            X2 = BndryDescMat[j,2]
            Y2 = BndryDescMat[j,3]
            if (x1-X1)*(Y2-Y1)==(X2-X1)*(y1-Y1) and (x2-X1)*(Y2-Y1)==(X2-X1)*(y2-Y1):
                BndryEdge[i] = j
                break

    x1 = nodes[elem[:,0],0]
    y1 = nodes[elem[:,0],1]
    x2 = nodes[elem[:,2],0]
    y2 = nodes[elem[:,2],1]
    xc = 0.5*(x1+x2)
    yc = 0.5*(y1+y2)
    EmCntr = np.stack((xc, yc), axis=1)
    AllEg = nodes[edge[:,1],:] - nodes[edge[:,0],:]
    LenEg = np.sqrt(AllEg[:,0]*AllEg[:,0]+AllEg[:,1]*AllEg[:,1])
    TanEg = np.zeros((NumEgs,2))
    for i in range(NumEgs):
        TanEg[i,0] = float(AllEg[i,0])/LenEg[i]
        TanEg[i,1] = float(AllEg[i,1])/LenEg[i]
    NmlEg = np.zeros((NumEgs,2))
    NmlEg[:,0] = TanEg[:,1]
    NmlEg[:,1] = TanEg[:,0]*(-1)
    for ig in range(NumEgs):
        if BndryEdge[ig]>-1:
            NmlEg[ig,0] = BndryDescMat[BndryEdge[ig],4]
            NmlEg[ig,1] = BndryDescMat[BndryEdge[ig],5]

    return elem, edge,nodes, NumEgs,NumNds,NumEms,BndryEdge

# Values of Gaussian quadrature points
def GaussQuad(a,b):
    GaussQuad_Line = np.asarray((3,3))
    GaussQuad_Line = [[0.88729833462074, 0.11270166537926 ,0.27777777777778],
                      [0.50000000000000,  0.50000000000000,  0.44444444444444],
                      [0.11270166537926,  0.88729833462074,  0.27777777777778]]
    GaussQuad_Line = np.array(GaussQuad_Line)

    GaussQuad_Rectangle = np.asarray((9,3))
    GaussQuad_Rectangle= [[0.887298334620740, 0.887298334620740, 0.077160493827160],
                          [0.887298334620740, 0.500000000000000, 0.123456790123460],
                          [0.887298334620740, 0.112701665379260, 0.077160493827160],
                          [0.500000000000000, 0.887298334620740, 0.123456790123460],
                          [0.500000000000000, 0.500000000000000, 0.197530864197530],
                          [0.500000000000000, 0.112701665379260, 0.123456790123460],
                          [0.112701665379260, 0.887298334620740, 0.077160493827160],
                          [0.112701665379260, 0.500000000000000, 0.123456790123460],
                          [0.112701665379260, 0.112701665379260, 0.077160493827160]]
    GaussQuad_Rectangle = np.array(GaussQuad_Rectangle)

    return GaussQuad_Line, GaussQuad_Rectangle


# This part calculates gradient times gradient local matrix.
def CG_RectQ1_GradGradMat(nodes,NumEms,BndryEdge):

    EGGM = np.zeros((NumEms,4,4))

    k1 = elem[:,0]
    k2 = elem[:,2]
    x1 = nodes[k1,0]
    y1 = nodes[k1,1]
    x2 = nodes[k2,0]
    y2 = nodes[k2,1]
    Deltax = x2 - x1
    Deltay = y2 - y1
    hxhy = np.zeros(NumEms)
    hyhx = np.zeros(NumEms)
    for  i in range(NumEms):
        hxhy[i] = (1.0/6) * float(Deltax[i]) / Deltay[i]
        hyhx[i] = (1.0/6) * float(Deltay[i]) / Deltax[i]

    EGGM = np.zeros((NumEms,4,4))

    EGGM[:,0,0] =  2*hxhy + 2*hyhx
    EGGM[:,0,3] = -2*hxhy +   hyhx
    EGGM[:,0,1] =    hxhy - 2*hyhx
    EGGM[:,0,2] =   -hxhy -   hyhx

    EGGM[:,3,0] = -2*hxhy +   hyhx
    EGGM[:,3,3] =  2*hxhy + 2*hyhx
    EGGM[:,3,1] =   -hxhy -   hyhx
    EGGM[:,3,2] =    hxhy - 2*hyhx

    EGGM[:,1,0] =    hxhy - 2*hyhx
    EGGM[:,1,3] =   -hxhy -   hyhx
    EGGM[:,1,1] =  2*hxhy + 2*hyhx
    EGGM[:,1,2] = -2*hxhy +   hyhx

    EGGM[:,2,0] =   -hxhy -   hyhx
    EGGM[:,2,3] =    hxhy - 2*hyhx
    EGGM[:,2,1] = -2*hxhy +   hyhx
    EGGM[:,2,2] =  2*hxhy + 2*hyhx

    return EGGM

# This is the right hand side of the equation.
def rhs(x,y):
    rhs = 4.0*(-y*y+y)*np.sin(np.pi*x)
    return rhs

print('Constructing mesh and preparing basic info ...')
# This is the unit square domain.
xa = 0
xb = 1
yc = 0
yd = 1

# Mesh size. Now, assume x and y directions have the same discretization.
n = 16
nx = n
ny = n

# Vertices of each edge and normal directions.
BndryDescMat = np.asarray([[xa,yc,xb,yc,0,-1],
                          [xb,yc,xb,yd,1,0],
                          [xb,yd,xa,yd,0,1],
                          [xa,yd,xa,yc,-1,0]])

# "1" here means all boundaries are Dirichlet boundaries.
BndryCondType = np.asarray([[0],[1],[1],[1],[1]])

elem, edge,nodes, NumEgs,NumNds,NumEms,BndryEdge, = RectDom_RectMesh_GenUnfm(xa,xb,nx,yc,yd,ny)

GaussQuad_Line, GaussQuad_Rectangle = GaussQuad(3,9)

# Find indices of nodes on boundaries.
DirichletEdge = np.where(BndryCondType[BndryEdge+1]==1)
DirichletEdge = np.asarray(DirichletEdge)
DirichletEdge = DirichletEdge[0,:]

# This is the main component.
print('assembling and solving ...')

NumDirichletEgs = DirichletEdge.shape[0]
DirichletNodeFlag = np.zeros((NumNds,1))
for ig in  np.arange(NumDirichletEgs):
    k1 = edge[DirichletEdge[ig],0]
    k2 = edge[DirichletEdge[ig],1]
    DirichletNodeFlag[k1,0] = 1
    DirichletNodeFlag[k2,0] = 1
DirichletNode = np.nonzero(DirichletNodeFlag)
DirichletNode = np.asarray(DirichletNode)
DirichletNode = DirichletNode[0,:]
NumDirichletNds = DirichletNodeFlag.sum()
NumDirichletNds = NumDirichletNds.astype(int)

DOFs = NumNds

EGGM = CG_RectQ1_GradGradMat(nodes,NumEms,BndryEdge)

# Global matrix is a sparse matrix.
GlbMat = coo_matrix((DOFs,DOFs), dtype = np.float32).toarray()

for i in np.arange(4):
    II = elem[:,i]
    for j in np.arange(4):
        JJ = elem[:,j]
        Aij = EGGM[:,i,j]

        GlbMat = GlbMat + sparse(II,JJ,Aij,NumNds,NumNds)

GlbRHS = np.zeros((DOFs,1))
NumQuadPts = GaussQuad_Rectangle.shape[0]

qp = np.zeros((1,2))

# Global right hand side.
for ie in np.arange(NumEms):
  k1 = elem[ie,0]
  k2 = elem[ie,2]
  x1 = nodes[k1,0]
  y1 = nodes[k1,1]
  x2 = nodes[k2,0]
  y2 = nodes[k2,1]
  ar = (x2-x1) * (y2-y1)

  for k in np.arange(NumQuadPts):
    qp[0,0] = GaussQuad_Rectangle[k,0]*x1 + (1-GaussQuad_Rectangle[k,0])*x2
    qp[0,1] = GaussQuad_Rectangle[k,1]*y1 + (1-GaussQuad_Rectangle[k,1])*y2

    X = (qp[0,0]-x1)/(x2-x1)
    Y = (qp[0,1]-y1)/(y2-y1)
    fval = rhs(qp[0,0],qp[0,1])
    GlbRHS[elem[ie,0], 0] = GlbRHS[elem[ie,0], 0]  + GaussQuad_Rectangle[k,2] * ar * fval * (1.0-X)*(1.0-Y)
    GlbRHS[elem[ie,1], 0] = GlbRHS[elem[ie,1], 0]  + GaussQuad_Rectangle[k,2] * ar * fval *     X*(1.0-Y)
    GlbRHS[elem[ie,2], 0] = GlbRHS[elem[ie,2], 0]  + GaussQuad_Rectangle[k,2] * ar * fval *     X*Y
    GlbRHS[elem[ie,3], 0] = GlbRHS[elem[ie,3], 0]  + GaussQuad_Rectangle[k,2] * ar * fval * (1.0-X)*Y

flag = np.zeros((DOFs,1))
flag[DirichletNode] = np.ones((NumDirichletNds,1))
FreeNd = np.where(flag != 1)
FreeNd = np.array(FreeNd)
FreeNd = FreeNd[0,:]
FreeNd_mat = FreeNd.reshape((nx-1,ny-1))
FreeNd_int = FreeNd.astype(int)

# Solve for solution
sln = np.zeros((DOFs,1))

# Assign boundary values to solution
DiriNd = nodes[DirichletNode,:]
sln[DirichletNode,0] = 0

# update right hand side
GlbRHS = GlbRHS - np.matmul(GlbMat,sln)


# Extract submatrix of the system matrix and system right hand side.
# Solve for unknowns at inteior nodes.
GlbMat_FreeNd = coo_matrix((len(FreeNd_int),len(FreeNd_int)), dtype = np.float32).toarray()
for i in range(len(FreeNd_int)):
    for j in range(len(FreeNd_int)):
        b = FreeNd_int[j]
        GlbMat_FreeNd[i,j] = GlbMat[FreeNd_int[i],b]

GlbRHS_FreeNd = np.zeros((len(FreeNd_int),1))
for i in range(len(FreeNd_int)):
    GlbRHS_FreeNd[i] = GlbRHS[FreeNd_int[i]]

GlbMat_FreeNd_inv = np.linalg.inv(GlbMat_FreeNd)
# This is the solution of interior nodes
sln_sub = np.dot(GlbMat_FreeNd_inv,GlbRHS_FreeNd)

# Construct whole solution array.
for i in range(len(FreeNd_int)):
    sln[FreeNd_int[i]] = sln_sub[i]



# Plot solutions in 3D points.
print('Graphing...')
ax = plt.axes(projection='3d')
x = nodes[:,0]
y = nodes[:,1]
ax.scatter(x, y, sln)

plt.show()
