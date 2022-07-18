import matplotlib.pyplot as plt
import trimesh # for reading mesh only
import pdb
import vtk
from scipy.spatial.transform import Rotation as R

import numpy as np
NORM = np.linalg.norm

def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper


def get_neibour(id,f):
    neibour_list = []
    for i in id:
        neibour = set(f[np.where(f==i)[0]].flatten())
        if i in neibour:
            neibour.remove(i)
        neibour = list(neibour)
        neibour_list.extend(neibour)
    return neibour_list

def get_adj(v,f,level):
    adj = []
    for i,vt in enumerate(v):
        neibour_list = [i]
        for j in range(level):
            neibour_list.extend( get_neibour(neibour_list,f) )
        neibour_list.remove(i)
        adj.append(neibour_list)
    return adj


def get_edges(v,f):
    edges = []
    adj = []
    for i,vt in enumerate(v):
        neibour = get_neibour([i],f)
        adj.append(neibour)
        edges.append( np.sort(np.vstack(( np.repeat(i,len(neibour)),neibour)).T,axis=1) )
    return edges,adj

def find_clockwise_nearest(vector_a,vector_b_arr,id_list):

    """
    This function find the smallest clockwise angle between vector_a and vector in vector_b_arr.
    Clockwise->Positive
    from B to A.
    Args:
    1.  vector_a
    2.  vector_b_arr , array of vectors
    3.  id_list: id of verts in vert_b_arr
    Return:
    1 . find the vector b in vector b array that has the smallest angle between vector a
    and return the id of the point that consist vector b

    """
    ang = np.arctan2(vector_a[0]*vector_b_arr[:,1]-vector_a[1]*vector_b_arr[:,0],vector_a[0]*vector_b_arr[:,0]+vector_a[1]*vector_b_arr[:,1]) #正切的加减.
    # id_list = vector_b_arr[:,2]
    positive_id = np.where(ang > 1e-3)[0] # when ang == 0, means vector_a find it self.  using 1e-12 to aviod float precision error,
    if positive_id.shape[0] > 0 :
        # e.g angle [-20,20,30] we wanna get 20 degree, rather than -20 degree,
        # because -20 degree means the vector has neg direction compare to vector_a
        clockwise_nearest = positive_id[np.argmin(ang[positive_id])]
    else:
        negative_id = np.where(ang < 0)[0]
        clockwise_nearest = negative_id[np.argmin(ang[negative_id])]
    next_pt_id = int(id_list[clockwise_nearest])
    return next_pt_id

def find_inters(pv,rv,qv,sv):
    # find intersections https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    # p is one vector
    # q is a set of vectors
    cross = rv[0]*sv[:,1]-rv[1]*sv[:,0]
    cross [cross == 0] = 1 # once cross product equals zero,no intersection happened.
    if np.any(cross==0):
        pdb.set_trace()
    qv_minus_pv = qv - pv
    t = (qv_minus_pv[:,0]*sv[:,1]-qv_minus_pv[:,1]*sv[:,0]) / cross
    u = (qv_minus_pv[:,0]*rv[1]-qv_minus_pv[:,1]*rv[0]) / cross
    line_has_inters = np.where( ( (t < 1-1e-6) & (t>1e-6))  & ( (u<1-1e-6) & (u>1e-6)) & (cross !=0) )[0]
    #这里我们要求 t 以及 u均在0与1之间.
    if line_has_inters.shape[0] !=0:
        intersections = pv + t[line_has_inters].reshape(-1,1) * rv
        return intersections,line_has_inters
    else:
        return None,None
# @timer
def tracing_outline_robust(verts,faces):
    """
    this is the version not require tree building, it calculates all intersections from all edges
    args:
    1.N X 2 verts
    2.M X 3 faces
    Return:
    outline points coordinates
    """
    start_id = np.argmin(verts[:,0])
    center_pt = verts[start_id]
    pre_pt = center_pt.copy()
    pre_pt[0] = pre_pt[0] - 1 #start from left
    next_id = start_id
    break_c = verts.shape[0]
    i = 0
    edges_list,adj = get_edges(verts,faces) #这里的edge list指的是
    edges_list = [[[i] + [tiny_each] for tiny_each in sorted(each)] for i, each in enumerate(adj)]
    edge_arr = np.vstack((np.asarray(edges_list)))
    edge_arr = edge_arr.astype('int')

    out_points = []
    connect_id = []
    out_id = []
    out_id.append(next_id)
    out_points.append(verts[next_id])

    while True and i < break_c:
        i += 1
        if len(connect_id) == 0:
            connect_id = adj[next_id]
        vector_a = pre_pt - center_pt
        vector_b_arr = verts[connect_id] - center_pt
        next_id = find_clockwise_nearest(vector_a,vector_b_arr,connect_id)

        if next_id == start_id:
            break
        pre_pt = center_pt
        center_pt = verts[next_id]
        arr_q = verts[edge_arr[:,0]]
        arr_r = verts[edge_arr[:,1]] - arr_q
        inters,inter_edge_id = find_inters(center_pt,pre_pt-center_pt,arr_q,arr_r)
        if inters is not None:
            nearest = np.argmin(NORM(inters - pre_pt,axis=1))
            center_pt = inters[nearest]
            connect_id = np.ndarray.tolist(edge_arr[inter_edge_id[nearest]])
            connect_id.append(next_id)
            inters = None
        else:
            connect_id = []
            inters = None
            out_id.append(next_id)
        out_points.append(center_pt)

    return np.asarray(out_points),out_id

def getCellIds(polydata):
    cells = polydata.GetPolys()
    ids = []
    idList = vtk.vtkIdList()
    cells.InitTraversal()
    while cells.GetNextCell(idList):
        for i in range(0, idList.GetNumberOfIds()):
            pId = idList.GetId(i)
            ids.append(pId)
    ids = np.array(ids)
    return ids
#
# import time
#
# import vtkmodules.all as vtk
# import vtkmodules.util.numpy_support
# mesh_path = '/data/test_test.vtk'
# mesh = vtk.vtkPolyDataReader()
# mesh.SetFileName(mesh_path)
# mesh.Update()
# data = mesh.GetOutput()
#
#
# v,f = vtkmodules.util.numpy_support.vtk_to_numpy(data.GetPoints().GetData()),getCellIds(data).reshape(-1,3)
#
# random_euler_angel = np.random.uniform(0,0,(3,1))
# rot = R.from_euler('xyz',random_euler_angel.flatten())
# rot_matrix = rot.as_matrix()
# v = v @ rot_matrix.T
#
# v_2d = v[:,:2]
# # can add transformation here,not that hard just a matrix of projection.
# points,ids = tracing_outline_robust(v_2d,f)
#
# st = time.time()
# fig= plt.figure(figsize=(9,9))
# plt.triplot(v_2d[:, 0], v_2d[:, 1], f)
# plt.plot(v[ids][:,0],v[ids][:,1])
# plt.plot(v[ids][:,0],v[ids][:,1],'r.')
# print(time.time()-st)
# plt.show()