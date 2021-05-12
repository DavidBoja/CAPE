import sys
import smplx
sys.path.append('../lib')
from utils import pose2rot

# OVO JE IMPORT OD CAPE JUPYTERA
import numpy as np
import os
import copy
from demos import demo_full
from lib import models, mesh_sampling
from lib.load_data import BodyData, load_graph_mtx
from config_parser import parse_config
from psbody.mesh import Mesh
import yaml
import torch


# OVO SU IMPORTI OD 3D CODED
# import pymesh
# mesh_ref JE OD 3DCODED
# ref_mesh JE OD CAPE
# from smpl_webuser.serialization import load_model
mesh_ref = pymesh.load_mesh("./template/template_color.ply") # 3d CODED
import pickle

def generate_surreal(pose, beta, outmesh_path):
    """
    This function generation 1 human using a random pose and shape estimation from surreal
    """
    ## Assign gaussian pose
    m.pose[:] = pose
    m.betas[:] = beta
    m.pose[0:3]=0
    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid

    # ovdje ga bojaju kao smpl tempalte iz ./data/template/tempalte_color.ply
    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return


def generate_benthuman(pose, beta, outmesh_path):
    """
    This function generation 1 human using a random gaussian pose and shape, with random gaussian parameters for specific pose parameters
    """
    ## Assign random pose parameters except for certain ones to have random bent humans
    m.pose[:] = pose
    m.betas[:] = beta

    a = np.random.randn(12)
    m.pose[1] = 0
    m.pose[2] = 0
    m.pose[3] = -1.0 + 0.1*a[0] # right upper leg part 
    m.pose[4] = 0 + 0.1*a[1] # right upper leg part 
    m.pose[5] = 0 + 0.1*a[2] # right upper leg part 
    m.pose[6] = -1.0 + 0.1*a[0] # left upper leg part
    m.pose[7] = 0 + 0.1*a[3] # left upper leg part
    m.pose[8] = 0 + 0.1*a[4] # left upper leg part
    m.pose[9] = 0.9 + 0.1*a[6] # wrist 1??
    m.pose[0] = - (-0.8 + 0.1*a[0] ) 
    m.pose[18] = 0.2 + 0.1*a[7] # wrist 2??
    m.pose[43] = 1.5 + 0.1*a[8] # left hand
    m.pose[40] = -1.5 + 0.1*a[9] # right hand
    m.pose[44] = -0.15  # left hand
    m.pose[41] = 0.15 # right hand
    m.pose[48:54] = 0 # right upper arm  # left lower arm

    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid

    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return

def find_joint_influence(pose, beta, outmesh_path,i):
    m.pose[:] = 0
    m.betas[:] = beta
    m.pose[i] = 1
    point_set = m.r.astype(np.float32)
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) 
    point_set[:,0:3] = point_set[:,0:3] - centroid

    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return

def generate_potential_templates(pose, beta, outmesh_path):
    # template 0
    m.pose[:] = 0
    m.betas[:] = beta
    m.pose[5] = 0.5
    m.pose[8] = -0.5
    m.pose[53] = -0.5
    m.pose[50] = 0.5

    point_set = m.r.astype(np.float32)
    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh('search/template0.ply', mesh, "red", "green", "blue", ascii=True)

    # template 1
    m.pose[:] = 0
    point_set = m.r.astype(np.float32)

    mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))


    pymesh.meshio.save_mesh('search/template1.ply', mesh, "red", "green", "blue", ascii=True)
    return

def get_random(poses, betas):
    beta_id = np.random.randint(np.shape(betas)[0]-1)
    beta = betas[beta_id]
    pose_id = np.random.randint(len(poses)-1)
    pose_ = database[poses[pose_id]]
    pose_id = np.random.randint(np.shape(pose_)[0])
    pose = pose_[pose_id]
    return pose, beta

def get_random_cape(poses, betas, clothtypes):
    beta_id = np.random.randint(np.shape(betas)[0]-1)
    beta = betas[beta_id]
    pose_id = np.random.randint(len(poses)-1)
    pose_ = database[poses[pose_id]]
    pose_id = np.random.randint(np.shape(pose_)[0])
    pose = pose_[pose_id]
    cloth_id = np.random.randint(np.shape(clothtypes)[0]-1)
    clothtype = clothtypes[cloth_id]
    return pose, beta, clothtype


def generate_database_surreal(male):
    #TRAIN DATA
    nb_generated_humans = 100000
    nb_generated_humans_val = 100
    if male:
        betas = database['maleshapes']
        offset = 0
        offset_val = 0
    else:
        betas = database['femaleshapes']
        offset = nb_generated_humans
        offset_val = nb_generated_humans_val

    poses = [i for i in database.keys() if "pose" in i]
    print(len(poses))
    num_poses= 0
    for i in poses:
        num_poses = num_poses + np.shape(database[i])[0]
    print('Number of poses ' + str(num_poses))
    print('Number of betas ' + str(np.shape(betas)[0]))
    params = []
    # spremaju ih kao i.ply
    for i in range(nb_generated_humans):
        pose, beta = get_random(poses, betas)
        generate_surreal(pose, beta, 'dataset-surreal/' + str(offset + i) + '.ply')
    

    #VAL DATA
    for i in range(nb_generated_humans_val):
        pose, beta = get_random(poses, betas)
        generate_surreal(pose, beta, 'dataset-surreal-val/' + str(offset_val + i) + '.ply')

    return 0


def generate_database_benthumans(male):
    #TRAIN DATA
    nb_generated_humans = 15000
    nb_generated_humans_val = 100
    if male:
        betas = database['maleshapes']
        offset = 0
        offset_val = 0
    else:
        betas = database['femaleshapes']
        offset = nb_generated_humans
        offset_val = nb_generated_humans_val

    poses = [i for i in database.keys() if "pose" in i]
    print(len(poses))
    num_poses= 0
    for i in poses:
        num_poses = num_poses + np.shape(database[i])[0]
    print('Number of poses ' + str(num_poses))
    print('Number of betas ' + str(np.shape(betas)[0]))
    params = []
    for i in range(nb_generated_humans):
        pose, beta = get_random(poses, betas)
        generate_benthuman(pose, beta, 'dataset-bent/' + str(offset + i) + '.ply')
    
    #VAL DATA
    for i in range(nb_generated_humans_val):
        pose, beta = get_random(poses, betas)
        generate_benthuman(pose, beta, 'dataset-bent-val/' + str(offset_val + i) + '.ply')

    return 0

############################################################################################################################################
def load_cape_network():
    reference_mesh_file = 'data/template_mesh.obj'
    reference_mesh = Mesh(filename=reference_mesh_file)

    ds_factors = [1, 2, 1, 2, 1, 2, 1, 1]

    print("Pre-computing mesh pooling matrices ..")
    M,A,D,U, _ = mesh_sampling.generate_transform_matrices(reference_mesh, ds_factors)
    p = list(map(lambda x: x.shape[0], A))
    A = list(map(lambda x: x.astype('float32'), A))
    D = list(map(lambda x: x.astype('float32'), D))
    U = list(map(lambda x: x.astype('float32'), U))
    L = [mesh_sampling.laplacian(a, normalized=True) for a in A]

    # load pre-computed graph laplacian and pooling matrices for discriminator
    L_ds2, D_ds2, U_ds2 = load_graph_mtx(project_dir)

    with open('configs/CAPE-affineconv_nz64_pose32_clotype32_male.yaml') as fl:
        params = yaml.load(fl, Loader=yaml.FullLoader)

    params['lr_scaler'] = 1e-1
    params['lambda_gan'] = 0.1
    params['regularization'] = 2e-3
    nf = params["nf"]
    if params["num_conv_layers"]==4:
        params['F'] = [nf, 2*nf, 2*nf, nf]
    elif params["num_conv_layers"]==6:
        params['F'] = [nf, nf, 2*nf, 2*nf, 4*nf, 4*nf]
    elif params["num_conv_layers"] == 8:
        params['F'] = [nf, nf, 2*nf, 2*nf, 4*nf, 4*nf, 8*nf, 8*nf]
    else:
        raise NotImplementedError
    
    params['p'] = p
    params['K'] = [2] * params["num_conv_layers"]
    params['restart'] = 1
    params['nn_input_channel'] = 3
    params['Kd'] = 3
    params['cond_dim'] = 14*9
    params['cond2_dim'] = 4
    params['n_layer_cond'] = 1
    params['optimizer'] = 'sgd'
    params['optim_condnet'] = 1

    non_model_params = ['demo_n_sample', 'mode', 'dataset', 'num_conv_layers', 'ds_factor',
                    'nf', 'config', 'pose_type', 'decay_every', 'gender',
                    'save_obj', 'vis_demo', 'smpl_model_folder']

    for key in non_model_params:
        params.pop(key,None)

    print("Building model graph...")
    model = models.CAPE(L=L, D=D, U=U, L_d=L_ds2, D_d=D_ds2, **params)
    model.build_graph(model.input_num_verts, model.nn_input_channel, phase='demo')
    print('Model loaded')

    return model

def generate_database_surreal_cape(male, model, train_stats, clothing_verts_idx, nr_cloth_types):
    # 100000 train examples and 100 validation examples
    nb_generated_humans = 100000
    nb_generated_humans_val = 100

    # SHAPE PARAM
    if male:
        betas = database['maleshapes']
        offset = 0
        offset_val = 0
    else:
        betas = database['femaleshapes']
        offset = nb_generated_humans
        offset_val = nb_generated_humans_val

    # POSE PARAM
    poses = [i for i in database.keys() if "pose" in i]
    print(len(poses))
    num_poses= 0
    for i in poses:
        num_poses = num_poses + np.shape(database[i])[0]

    # PRINTOUT
    print('Number of poses ' + str(num_poses))
    print('Number of betas ' + str(np.shape(betas)[0]))

    # CREATE TRAIN DATA
    for i in range(nb_generated_humans):
        # returns pose (72,) pose params for 24 joints x 3
        #         beta (10,) shape param for 10 PCA comp
        #         clothtype (4,) clothes param onehot vector
        pose, beta, clothtype = get_random_cape(poses, betas, np.eye(nr_cloth_types))
        z = np.random.normal(loc=0.0, scale=1.0, size=(1, model.nz))
        generate_surreal_cape(pose, beta, clothtype, z, 'dataset-surreal-cape/' + str(offset + i) + '.ply',
                              model, train_stats, clothing_verts_idx)
    

    #VAL DATA
    for i in range(nb_generated_humans_val):
        pose, beta, clothtype = get_random_cape(poses, betas, np.eye(nr_cloth_types))
        generate_surreal_cape(pose, beta, clothtype, z, 'dataset-surreal-val-cape/' + str(offset_val + i) + '.ply',
                              model, train_stats, clothing_verts_idx)

    return 0       

def generate_surreal_cape(pose, beta, clothtype, latent_code, outmesh_path, model, train_stats, clothing_verts_idx):
    """
    This function generation 1 human using a random pose and shape estimation from surreal
    input:  pose (72,)
            beta (10,)
            clothtype (4,)
            latent_code (1,model.nz)
    """

    # GENERATE CLOTHING
    rot = pose2rot(pose.reshape(1,-1))
    # takes pose param as 24x9 = 216 as 1x216 vector rot and clotype as onehot vector 1x4 of clothing type
    pose_emb, clothtype_emb = model.encode_only_condition(rot, clothtype.reshape(1,-1)) # 1x32, 1x32
    z_sample_c = np.concatenate([latent_code, pose_emb, clotype_emb],axis=1).reshape(1,-1) # 1x 128

    predictions = model.decode(z_sample_c, cond=pose_emb, cond2=clotype_emb)
    predictions = predictions[0] * train_stats['std'] + train_stats['mean']

    disp_masked = np.zeros_like(predictions)
    disp_masked[clothing_verts_idx, :] = predictions[clothing_verts_idx, :]

    # CREATE SMPL MODEL
    smpl_model.v_template[:] = smpl_model.v_template[:] + torch.from_numpy(disp_masked)
    smpl_model.betas[:] = torch.from_numpy(beta)
    smpl_model.body_pose[:] = torch.from_numpy(pose[3:].reshape(1,-1))
    smpl_model.global_orient[:] = torch.zeros(1,3)

    point_set = smpl_model().vertices.detach().numpy()
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid

    # ovdje ga bojaju kao smpl tempalte iz ./data/template/tempalte_color.ply
    mesh = pymesh.form_mesh(vertices=point_set, faces=smpl_model.faces)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return

def generate_database_benthumans_cape(male, model, train_stats, clothing_verts_idx, nr_cloth_types):
    # 15000 train examples and 100 validation examples
    nb_generated_humans = 15000
    nb_generated_humans_val = 100

    # SHAPE PARAM
    if male:
        betas = database['maleshapes']
        offset = 0
        offset_val = 0
    else:
        betas = database['femaleshapes']
        offset = nb_generated_humans
        offset_val = nb_generated_humans_val

    # POSE PARAM
    poses = [i for i in database.keys() if "pose" in i]
    print(len(poses))
    num_poses= 0
    for i in poses:
        num_poses = num_poses + np.shape(database[i])[0]
    print('Number of poses ' + str(num_poses))
    print('Number of betas ' + str(np.shape(betas)[0]))

    for i in range(nb_generated_humans):
        # pose, beta = get_random(poses, betas)
        pose, beta, clothtype = get_random_cape(poses, betas, np.eye(nr_cloth_types))
        z = np.random.normal(loc=0.0, scale=1.0, size=(1, model.nz))
        generate_benthuman_cape(pose, beta, clothtype, z, 'dataset-bent-cape/' + str(offset + i) + '.ply',
                                model, train_stats, clothing_verts_idx)
    
    #VAL DATA
    for i in range(nb_generated_humans_val):
        # pose, beta = get_random(poses, betas)
        pose, beta, clothtype = get_random_cape(poses, betas, np.eye(nr_cloth_types))
        z = np.random.normal(loc=0.0, scale=1.0, size=(1, model.nz))
        generate_benthuman_cape(pose, beta, clothtype, z, 'dataset-bent-val-cape/' + str(offset_val + i) + '.ply',
                                model, train_stats, clothing_verts_idx)

    return 0

def generate_benthuman_cape(pose, beta, clothtype, latent_code, outmesh_path, model, train_stats, clothing_verts_idx):
    """
    This function generation 1 human using a random gaussian pose and shape, with random gaussian parameters for specific pose parameters
    """
    # BEND POSE
    a = np.random.randn(12)
    pose[1] = 0
    pose[2] = 0
    pose[3] = -1.0 + 0.1*a[0] # right upper leg part 
    pose[4] = 0 + 0.1*a[1] # right upper leg part 
    pose[5] = 0 + 0.1*a[2] # right upper leg part 
    pose[6] = -1.0 + 0.1*a[0] # left upper leg part
    pose[7] = 0 + 0.1*a[3] # left upper leg part
    pose[8] = 0 + 0.1*a[4] # left upper leg part
    pose[9] = 0.9 + 0.1*a[6] # wrist 1??
    pose[0] = - (-0.8 + 0.1*a[0] ) 
    pose[18] = 0.2 + 0.1*a[7] # wrist 2??
    pose[43] = 1.5 + 0.1*a[8] # left hand
    pose[40] = -1.5 + 0.1*a[9] # right hand
    pose[44] = -0.15  # left hand
    pose[41] = 0.15 # right hand
    pose[48:54] = 0 # right upper arm  # left lower arm


    # GENERATE CLOTHING
    rot = pose2rot(pose.reshape(1,-1))
    # takes pose param as 24x9 = 216 as 1x216 vector rot and clotype as onehot vector 1x4 of clothing type
    pose_emb, clothtype_emb = model.encode_only_condition(rot, clothtype.reshape(1,-1)) # 1x32, 1x32
    z_sample_c = np.concatenate([latent_code, pose_emb, clotype_emb],axis=1).reshape(1,-1) # 1x 128

    predictions = model.decode(z_sample_c, cond=pose_emb, cond2=clotype_emb)
    predictions = predictions[0] * train_stats['std'] + train_stats['mean']

    disp_masked = np.zeros_like(predictions)
    disp_masked[clothing_verts_idx, :] = predictions[clothing_verts_idx, :]

    # CREATE SMPL MODEL
    smpl_model.v_template[:] = smpl_model.v_template[:] + torch.from_numpy(disp_masked)
    smpl_model.betas[:] = torch.from_numpy(beta)
    smpl_model.body_pose[:] = torch.from_numpy(pose[3:].reshape(1,-1))
    smpl_model.global_orient[:] = torch.zeros(1,3)

    point_set = smpl_model().vertices.detach().numpy()
    
    #normalize
    centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
    point_set[:,0:3] = point_set[:,0:3] - centroid

    mesh = pymesh.form_mesh(vertices=point_set, faces=smpl_model.faces)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)
    return


if __name__ == '__main__':
    os.mkdir("dataset-surreal-cape")
    os.mkdir("dataset-surreal-val-cape")
    os.mkdir("dataset-bent-cape")
    os.mkdir("dataset-bent-val-cape")

    ### LOAD CAPE NETWORK
    model = load_cape_network()
    train_stats = np.load('data/demo_data/trainset_stats.npz')

    ### CLOTHES 
    clo_type_readable = np.array(['shortlong', 'shortshort', 'longshort', 'longlong'])
    nr_cloth_types = len(clo_type_readable)
    clothing_verts_idx = np.load('data/clothing_verts_idx.npy')

    ### GENERATE MALE EXAMPLES
    # m = load_model("./smpl_data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    m = smplx.body_models.create(model_type='smpl',
                                 model_path='body_models',
                                 gender='male')
    # database = np.load("/home/thibault/tmp/SURREAL/smpl_data/smpl_data.npz")
    database = np.load("../../datasets/SURREAL/DATASET/SURREAL/smpl_data/smpl_data.npz")
    generate_database_surreal_cape(male=True, model, train_stats, clothing_verts_idx, nr_cloth_types)
    generate_database_benthumans_cape(male=True, model, train_stats, clothing_verts_idx, nr_cloth_types)
   
    ### GENERATE FEMALE EXAMPLES
    # m = load_model('./smpl_data/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    m = smplx.body_models.create(model_type='smpl',
                                 model_path='body_models',
                                 gender='female')
    database = np.load("../../datasets/SURREAL/DATASET/SURREAL/smpl_data/smpl_data.npz")
    generate_database_surreal_cape(male=False, model, train_stats, clothing_verts_idxm nr_cloth_types)
    generate_database_benthumans_cape(male=False, model, train_stats, clothing_verts_idx, nr_cloth_types)
