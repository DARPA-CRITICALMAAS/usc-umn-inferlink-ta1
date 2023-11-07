import pdb
import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


patch_size = [256,256,1]
# patch_size = [512,512,1]
pad = [5,5,0]

def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min = -1, a_max=1))

def convert_graph(graph):
    node_list = []
    edge_list = []
    for n, v in graph.items():
        node_list.append(n)
    node_array = np.array(node_list)

    for ind, (n, v) in enumerate(graph.items()):
        for nei in v:
            idx = node_list.index(nei)
            edge_list.append(np.array((ind,idx)))
    edge_array = np.array(edge_list)
    return node_array, edge_array

vector_norm = 25.0 

def neighbor_transpos(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (k[1], k[0])
		nv = []

		for _v in v :
			nv.append((_v[1],_v[0]))

		n_out[nk] = nv 

	return n_out 

def neighbor_to_integer(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (int(k[0]), int(k[1]))
		
		if nk in n_out:
			nv = n_out[nk]
		else:
			nv = []

		for _v in v :
			new_n_k = (int(_v[0]),int(_v[1]))

			if new_n_k in nv:
				pass
			else:
				nv.append(new_n_k)

		n_out[nk] = nv 

	return n_out


def save_input(path, patch, patch_seg, patch_coord, patch_edge, idx):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    region_id, img_x, img_y = idx
    imageio.imwrite(path+'raw/sample_'+str(region_id)+'_'+str(img_x)+'_'+str(img_y)+'_data.png', patch.astype('uint8'))
    imageio.imwrite(path+'seg/sample_'+str(region_id)+'_'+str(img_x)+'_'+str(img_y)+'_seg.png', patch_seg.astype('uint8'))
    
    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)
    
    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'mesh/sample_'+str(idx).zfill(4)+'_segmentation.stl')
    
    if patch_coord != []:
        patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
        mesh = pyvista.PolyData(patch_coord)
        mesh.lines = patch_edge.flatten()
    else:
        mesh = pyvista.PolyData(np.transpose(np.array([[],[],[]])))
        mesh.lines = np.array([]) #None
    # print(patch_edge.shape)
    mesh.save(path+'vtp/sample_'+str(region_id)+'_'+str(img_x)+'_'+str(img_y)+'_graph.vtp')


def patch_extract(save_path, image, seg, mesh, indice, device=None):
    """[summary]

    Args:
        image ([type]): [description]
        coordinates ([type]): [description]
        lines ([type]): [description]
        patch_size (tuple, optional): [description]. Defaults to (64,64,64).
        num_patch (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    
    h, w, d= image.shape
    print('img height, width: ', h, w)
    x_ = np.int32(np.linspace(5, h-5-p_h, 100)) 
    y_ = np.int32(np.linspace(5, w-5-p_w, 100)) 
    ind = np.meshgrid(x_, y_, indexing='ij')
    # Center Crop based on foreground

    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
#         start = np.array((start[0],start[1],0))
        start = np.array((start[1],start[0],0))
        end = start + np.array(patch_size)-1 -2*np.array(pad)
        
        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :],\
                       ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')
#         patch = np.pad(image[start[1]:start[1]+p_h, start[0]:start[0]+p_w, :], ((pad_h,pad_h),(pad_w,pad_w),(0,0)))
        patch_list = [patch]

        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w,],\
                           ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
#         patch_seg = np.pad(seg[start[1]:start[1]+p_h, start[0]:start[0]+p_w,], ((pad_h,pad_h),(pad_w,pad_w)))
#         print('patch_seg: ', patch_seg.shape, np.unique(patch_seg))
        seg_list = [patch_seg]

        # collect all the nodes
        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]

        clipped_mesh = mesh.clip_box(bounds, invert=False)
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
#         print('patch_coordinates: ', patch_coordinates)
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)
#         print('patch_edge shape: ', patch_edge.shape)
#         print('patch_edge: ', patch_edge)
        patch_coord_ind = np.where((np.prod(patch_coordinates>=start, 1)*np.prod(patch_coordinates<=end, 1))>0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]  # all coordinates inside the patch
        patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]
        
        temp = np.array(patch_edge).flatten()  # flatten all the indices of the edges which completely lie inside patch
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]  # remap the edge indices according to the new order
        patch_edge = np.array(temp).reshape(-1,2)  # reshape the edge list into previous format
            
        if patch_coordinates.shape[0] < 2 or patch_edge.shape[0] < 1:
            if patch_seg.shape[0] != patch_size[0] or patch_seg.shape[1] != patch_size[0]:
                continue
            for patch, patch_seg in zip(patch_list, seg_list):
                save_input(save_path, patch, patch_seg, [], [], [indice, start[0], start[1]])
        
        else:
            # concatenate final variables
            patch_coordinates = (patch_coordinates-start+np.array(pad))/np.array(patch_size)
            patch_coord_list = [patch_coordinates]#.to(device))
            patch_edge_list = [patch_edge]#.to(device))

            mod_patch_coord_list, mod_patch_edge_list = prune_patch(patch_coord_list, patch_edge_list)
    #         print('prune: ', mod_patch_coord_list)
            # save data
            for patch, patch_seg, patch_coord, patch_edge in zip(patch_list, seg_list, mod_patch_coord_list, mod_patch_edge_list):
                if patch_seg.shape[0] != patch_size[0] or patch_seg.shape[1] != patch_size[0]:
                    continue
                
                save_input(save_path, patch, patch_seg, patch_coord, patch_edge, [indice, start[0], start[1]])

def prune_patch(patch_coord_list, patch_edge_list):
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):

        # find largest graph segment in graph and in skeleton and see if they match
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:,0], edge[:,1]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        dist_adj[edge[:,1], edge[:,0]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)

        # straighten the graph by removing redundant nodes
        start = True
        node_mask = np.ones(coord.shape[0], dtype=np.bool)
        while start:
            degree = (dist_adj > 0).sum(1)
            deg_2 = list(np.where(degree==2)[0])
            if len(deg_2) == 0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx,:]>0)[0]

                p1 = coord[idx,:]
                p2 = coord[deg_2_neighbor[0],:]
                p3 = coord[deg_2_neighbor[1],:]
                l1 = p2-p1
                l2 = p3-p1
                node_angle = angle(l1,l2)*180 / math.pi
                if node_angle > 160:
                    node_mask[idx]=False
                    dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum((p2-p3)**2)
                    dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = np.sum((p2-p3)**2)

                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    break
                elif n==len(deg_2)-1:
                    start = False

        new_coord = coord[node_mask,:]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj)>0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list


# indrange_test = [5, 7, 11] # louisville wt test patches
# indrange_test = [10, 13, 32, 34, 8, 16] # bray wt test patches
# indrange_test = [26,37,8,15,22,28] # bray rr test patches
# indrange_test = [27,6,16] # louisville rr test patches
indrange_test = [43]

if __name__ == "__main__":
    root_dir = "/data/weiweidu/criticalmaas_data/validation_fault_line"
    
    image_id = 1
    test_path = './data/darpa/fault_lines/OR_Carlton_g256_s100/'
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
        os.makedirs(test_path+'/seg')
        os.makedirs(test_path+'/vtp')
        os.makedirs(test_path+'/raw')
#     else:
#         raise Exception("Test folder is non-empty")

    print('Preparing Test Data')

    raw_files = []
    seg_files = []
    vtk_files = []
    
    '''
    AK_Dillingham, AK_Kechumstuk, CA_Elsinore, 
    AK_Hughes, AR_StJoe, AZ_GrandCanyon, 
    AZ_PipeSpring, CA_NV_DeathValley, CO_Alamosa
    MN, OR_Camas, OR_Carlton
    '''

    map_names = ['OR_Carlton'] 
    
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in map_names:
            for f in files:
                fmt = f.split('.')[1]
# #                 print(f)
# #                 if 'scarp_line' in f and 'png' == fmt and name in f:
                if 'fault_line' in f and 'png' == fmt and name in f:
                    seg_files.append(os.path.join(root, f))
                elif 'png' == fmt and name in f:
                    raw_files.append(os.path.join(root, name))
                elif 'p' == fmt and name in f:
                    vtk_files.append(os.path.join(root, f))    
    print(len(seg_files), len(raw_files), len(vtk_files))      
    
    for i, ind in enumerate((raw_files)):
        print(ind, raw_files[i], seg_files[i])
        try:
            sat_img = imageio.imread(raw_files[i]+".png")
        except:
            sat_img = imageio.imread(raw_files[i]+".jpg")

#         with open(vtk_files[i], 'rb') as f:
#             graph = pickle.load(f)
#         node_array, edge_array = convert_graph(graph)
        node_array, edge_array = np.array([]), np.array([])
        if node_array.size == 0:
            node_array = np.array([[2050, 2050], [2070, 2070]]).astype('float32')
            edge_array = np.array([[0, 1]])
        if seg_files == []:    
            gt_seg = np.zeros(sat_img.shape[:2])
        else:    
            gt_seg = imageio.imread(seg_files[i])
        patch_coord = np.concatenate((node_array, np.int32(np.zeros((node_array.shape[0],1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()

#         patch_extract(test_path, sat_img, gt_seg, mesh, ind)
        patch_extract(test_path, sat_img, gt_seg, mesh, map_names[i])

