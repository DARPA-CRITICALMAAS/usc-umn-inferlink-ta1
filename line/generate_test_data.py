import pdb
import cv2
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
    imageio.imwrite(path+'/raw/sample_'+str(region_id)+'_'+str(img_x)+'_'+str(img_y)+'_data.png', patch.astype('uint8'))
    imageio.imwrite(path+'/seg/sample_'+str(region_id)+'_'+str(img_x)+'_'+str(img_y)+'_seg.png', patch_seg.astype('uint8'))
    
    if patch_coord != []:
        patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
        mesh = pyvista.PolyData(patch_coord)
        mesh.lines = patch_edge.flatten()
    else:
        mesh = pyvista.PolyData(np.transpose(np.array([[],[],[]])))
        mesh.lines = np.array([]) #None
    # print(patch_edge.shape)
    mesh.save(path+'/vtp/sample_'+str(region_id)+'_'+str(img_x)+'_'+str(img_y)+'_graph.vtp')


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
        start = np.array((start[1],start[0],0))
        end = start + np.array(patch_size)-1 -2*np.array(pad)
        
        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :],\
                       ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')
        patch_list = [patch]

        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w,],\
                           ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
        seg_list = [patch_seg]

        # collect all the nodes
        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]

        clipped_mesh = mesh.clip_box(bounds, invert=False)
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)

        patch_coord_ind = np.where((np.prod(patch_coordinates>=start, 1)*np.prod(patch_coordinates<=end, 1))>0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]  # all coordinates inside the patch
        patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]
        
        temp = np.array(patch_edge).flatten()  # flatten all the indices of the edges which completely lie inside patch
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]  # remap the edge indices according to the new order
        patch_edge = np.array(temp).reshape(-1,2)  # reshape the edge list into previous format
            
        
        if patch_seg.shape[0] != patch_size[0] or patch_seg.shape[1] != patch_size[0]:
            continue
        for patch, patch_seg in zip(patch_list, seg_list):
            save_input(save_path, patch, patch_seg, [], [], [indice, start[0], start[1]])
        
def check_path(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")
    except FileNotFoundError as e:
        print(e)
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--map_dir', type=str, default='/data/weiweidu/criticalmaas_data/validation_fault_line',
                       help='the path to the folder storing the test maps')
    
    parser.add_argument('--output_dir', type=str, default='/data/weiweidu/criticalmaas_data')
    
    args = parser.parse_args()
    
    root_dir = args.map_dir
    
    check_path(root_dir)
    
    image_id = 1
    test_path = args.output_dir
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
    
    raw_files = []
    seg_files = []
    vtk_files = []
    output_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        for f_name in files:
            map_name, fmt = f_name.split('.')
            if (fmt != 'tif' and fmt != 'png') or '_poly' in map_name or '_line' in map_name or '_pt' in map_name:
                continue
            output_dir = os.path.join(test_path, map_name+'_g256_s100')
            output_dirs.append(output_dir)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
                os.makedirs(output_dir+'/seg')
                os.makedirs(output_dir+'/vtp')
                os.makedirs(output_dir+'/raw')
            else:
                print(f'--- Test images are generated in {output_dir} ---')

            if os.path.exists(os.path.join(root, f_name)):
                print(os.path.join(root, f_name))
                raw_files.append(os.path.join(root, f_name))
            else:
                print(f'--- {map_name} does not exist in {root}')
                    
    print(f'the number of test maps = {len(raw_files)}')      
    
    for i, ind in enumerate((raw_files)):
        map_path = ind.split('/')[-1]
        map_name = map_path.split('.')[0]
        print(f'Preparing Test Data for {map_name}')
        sat_img = cv2.imread(raw_files[i])
        node_array, edge_array = np.array([]), np.array([])
        if node_array.size == 0:
            node_array = np.array([[2050, 2050], [2070, 2070]]).astype('float32')
            edge_array = np.array([[0, 1]])
  
        gt_seg = np.zeros(sat_img.shape[:2])
        
        patch_coord = np.concatenate((node_array, np.int32(np.zeros((node_array.shape[0],1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()

        patch_extract(output_dirs[i], sat_img, gt_seg, mesh, map_name)

