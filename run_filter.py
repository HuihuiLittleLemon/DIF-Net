
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from torch.utils.tensorboard import SummaryWriter
import utils
import io
import yaml
import cv2
import numpy as np
from itertools import product
import os
import configargparse
from NeuralLightField import TransformNetwork
import torch
from torch import nn
import tqdm
from torch.optim.lbfgs import LBFGS
import math
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import convolve
import triangle
import matplotlib.pyplot as plt

record_energy = True

if record_energy:
    total_energy = open("total_energy.txt", 'w')
    img_energy = open("img_energy.txt", 'w')
    regular_energy = open("regular_energy.txt", 'w')

p = configargparse.ArgumentParser()
opt = p.parse_args()

fp = open("./configs/name_and_mode.txt", 'r')
name_and_mode = []
for line in fp:
    line = line.strip('\n')  # 将\n去掉
    name_and_mode.append(line)
fp.close()
img_name = name_and_mode[0]
run_model = name_and_mode[1]  # train optimize

if run_model == 'train' or 'optimize':
    opt.config = 'configs/transform.yml'


config_name = os.path.basename(opt.config)
config_name = img_name + '_' + config_name
experiment_name = os.path.splitext(config_name)[0]
if opt.config == '':
    meta_params = vars(opt)
else:
    with open(opt.config, 'r') as stream:
        meta_params = yaml.safe_load(stream)

meta_params['experiment_name'] = img_name
meta_params['transform_path'] = './' + img_name + '_transform/transform_network_final.pth'

## create save path
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
exp_path = os.path.join(root_path, experiment_name)
intermediate_path = os.path.join(root_path, 'some_intermediate_results')
utils.cond_mkdir(root_path)
utils.cond_mkdir(exp_path)
utils.cond_mkdir(intermediate_path)

with io.open(os.path.join(exp_path, config_name), 'w', encoding='utf8') as outfile:
    yaml.dump(meta_params, outfile, default_flow_style=False, allow_unicode=True)

if run_model == 'show_train':
    checkpoints_dir = os.path.join(exp_path, 'train' + '/checkpoints')
else:
    checkpoints_dir = os.path.join(exp_path, run_model + '/checkpoints')
utils.cond_mkdir(checkpoints_dir)

def scale_img(img, coef):
    new_img = img.repeat([coef], axis=0).repeat([coef], axis=1)
    return new_img

def intensity_matrix(img):
    intensity = img.copy()
    intensity = intensity.sum(axis=-1)
    min_bound = 1
    max_bound = intensity.sum()
    while 1:
        mid_bound = (max_bound + min_bound) / 2
        new_intensity = np.around(intensity / mid_bound).astype(int)
        error = new_intensity.sum() - img.shape[0] * img.shape[1]
        if error > 0:
            min_bound = mid_bound
        else:
            max_bound = mid_bound

        if max_bound - min_bound < 1e-5:
            break

    # max_bound = 85 * 3
    new_intensity = np.around(intensity / max_bound).astype(int)





    # new_intensity_1 = np.around(intensity / (img.shape[0] * img.shape[1])).astype(int)
    # new_intensity_2 = np.around(intensity / (img.shape[0] * img.shape[1] + 0.001)).astype(int)
    # error_1 = new_intensity_1.sum() - img.shape[0] * img.shape[1]
    # error_2 = new_intensity_2.sum() - img.shape[0] * img.shape[1]
    # if error_1 > 0:
    #     new_intensity = new_intensity_2
    # elif error_2 > 0:
    #     new_intensity = new_intensity_1
    # else:
    #     print('Intensity_matrix error!!!')
    #     exit(0)

    return new_intensity


def load_TransformNetwork(checkpoints_name):
    test_network = TransformNetwork(**meta_params)
    test_network = torch.nn.DataParallel(test_network)
    # checkpoints_dir = os.path.join(root_path, 'template')
    # utils.cond_mkdir(checkpoints_dir)
    test_network.load_state_dict(torch.load(checkpoints_name))
    return test_network

def train_light_transformer(light_transformer, indices, triangles, adj_vertices, adj_boundry, adj_boundry_mask, B, orth_pic_height_fun, pro_img1, pro_img2, optim, iter_i, writer, optim_view_id, init_z, first_output):  
    W = pro_img1.shape[0]
    coordinates = torch.Tensor(indices).cuda()
    
    model_in = {'coords': coordinates}
    model_output = light_transformer(model_in)
    t_pixels = model_output['model_out']
    # if iter_i == 0:
    #     first_output = torch.sigmoid(t_pixels)

    # t_pixels = torch.sigmoid(t_pixels)-first_output + init_z
    t_pixels = torch.sigmoid(t_pixels)

    zi = t_pixels[adj_vertices]
    zi_w_boundry = t_pixels[adj_boundry].squeeze()*adj_boundry_mask
    NN = torch.sum(adj_boundry_mask, dim=1)-1
    zi_sum = torch.sum(zi_w_boundry, dim=1)/NN-zi_w_boundry[:,3]*(1+1/NN)
    ci = torch.matmul(B,zi).squeeze()
    gi = ci[:,3:5]
    hi = ci[:,0:3]
    H1 = get_gradients_and_H1(gi, hi)
    
    # print(torch.det(H1).abs().mean())

    # if iter_i < 1000:
    #     x_cord = indices[:,0]
    #     y_cord = indices[:,1]
    #     cords_z_w_samples = t_pixels
    # else:
    #     tri_cords_xy = coordinates[triangles]
    #     N_sample0_xy=1/3*tri_cords_xy[:,0,:]+2/3*tri_cords_xy[:,1,:]
    #     N_sample1_xy=2/3*tri_cords_xy[:,0,:]+2/3*tri_cords_xy[:,1,:]
    #     N_sample2_xy=1/2*(tri_cords_xy[:,0,:]+tri_cords_xy[:,1,:])
    #     N_sample3_xy=1/3*tri_cords_xy[:,0,:]+2/3*tri_cords_xy[:,2,:]
    #     N_sample4_xy=2/3*tri_cords_xy[:,0,:]+2/3*tri_cords_xy[:,2,:]
    #     N_sample5_xy=1/2*(tri_cords_xy[:,0,:]+tri_cords_xy[:,2,:])
    #     N_sample6_xy=1/3*tri_cords_xy[:,1,:]+2/3*tri_cords_xy[:,2,:]
    #     N_sample7_xy=2/3*tri_cords_xy[:,1,:]+2/3*tri_cords_xy[:,2,:]
    #     N_sample8_xy=1/2*(tri_cords_xy[:,1,:]+tri_cords_xy[:,2,:])
    #     N_sample9_xy=1/3*(tri_cords_xy[:,0,:]+tri_cords_xy[:,1,:]+tri_cords_xy[:,2,:])
    #     cords_xy_w_samples = torch.cat([coordinates,N_sample0_xy,N_sample1_xy,N_sample2_xy,N_sample3_xy,N_sample4_xy,
    #                                     N_sample5_xy,N_sample6_xy,N_sample7_xy,N_sample8_xy,N_sample9_xy],dim=0).cpu().numpy()
        
    #     tri_cords_z = t_pixels[triangles]
    #     N_sample0_z = 1/3*tri_cords_z[:,0,:]+2/3*tri_cords_z[:,1,:]
    #     N_sample1_z = 2/3*tri_cords_z[:,0,:]+2/3*tri_cords_z[:,1,:]
    #     N_sample2_z = 1/2*(tri_cords_z[:,0,:]+tri_cords_z[:,1,:])
    #     N_sample3_z = 1/3*tri_cords_z[:,0,:]+2/3*tri_cords_z[:,2,:]
    #     N_sample4_z = 2/3*tri_cords_z[:,0,:]+2/3*tri_cords_z[:,2,:]
    #     N_sample5_z = 1/2*(tri_cords_z[:,0,:]+tri_cords_z[:,2,:])
    #     N_sample6_z = 1/3*tri_cords_z[:,1,:]+2/3*tri_cords_z[:,2,:]
    #     N_sample7_z = 2/3*tri_cords_z[:,1,:]+2/3*tri_cords_z[:,2,:]
    #     N_sample8_z = 1/2*(tri_cords_z[:,1,:]+tri_cords_z[:,2,:])
    #     N_sample9_z = 1/3*(tri_cords_z[:,0,:]+tri_cords_z[:,1,:]+tri_cords_z[:,2,:])
    #     cords_z_w_samples = torch.cat([t_pixels,N_sample0_z,N_sample1_z,N_sample2_z,N_sample3_z,N_sample4_z,
    #                                     N_sample5_z,N_sample6_z,N_sample7_z,N_sample8_z,N_sample9_z],dim=0)

    #     x_cord = cords_xy_w_samples[:,0]
    #     y_cord = cords_xy_w_samples[:,1]

    x_cord = indices[:,0]*W
    y_cord = indices[:,1]*W
    direct_error = 0.0
    boundry_laplace_weight = 1e1
    gauss_weight = 5e5#zhang150可以优化出一个表面不太平整的结果
    scale = 0.5
    # if iter_i < 5000:
    #     scale = 0.4
    #     # laplace_weight = 0.01
    # else:
    #     scale = 0.4 + 0.4 * np.clip((iter_i - 5000) /3000, 0.0, 1.0)

    for idx in range(W):
        yy = np.where(np.around(y_cord) == idx) #y_cord和x_cord不是整数
        if yy[0].size == 0:
            continue
        if optim_view_id != 3:
            direct_error += orth_pic_height_fun(pro_img1[:,idx],t_pixels[yy].squeeze(),scale,W).sum()
        else:
            direct_error += orth_pic_height_fun(pro_img1[idx,:],t_pixels[yy].squeeze(),scale,W).sum()



    for idx in range(W):
        xx = np.where(np.around(x_cord) == idx)
        if xx[0].size == 0:
            continue
        if optim_view_id == 1:
            direct_error += orth_pic_height_fun(pro_img2[:,idx],t_pixels[xx].squeeze(),scale,W).sum()
        else:
            direct_error += orth_pic_height_fun(pro_img2[idx,:],t_pixels[xx].squeeze(),scale,W).sum()

    # laplace loss
    boundry_laplace_error = torch.mean(zi_sum**2)

    # gauss loss
    low_rank_H1,s_sum = calculate_lowrank_H1(H1)
    print(torch.det(low_rank_H1).abs().mean())
    loss_fn = torch.nn.MSELoss(reduction='mean')
    gauss_error = loss_fn(H1, low_rank_H1)
    # gauss_error = calculate_lowrank_H1(H1)

    # gauss_error = gauss_weight * gauss_error
    regular_weight = 1e5
    loss = gauss_weight * gauss_error + boundry_laplace_weight*boundry_laplace_error + direct_error
    optim.zero_grad()
    loss.backward()
    optim.step()

    writer.add_scalar('Loss/loss', loss, iter_i)
    writer.add_scalar('Loss/direct_error', direct_error,iter_i)
    writer.add_scalar('Loss/boundry_laplace_error', boundry_laplace_error, iter_i)
    writer.add_scalar('Loss/gauss_error', gauss_error, iter_i)


    if record_energy:
        print(float(loss), file=total_energy)
        # print(float(intensity_error), file=img_energy)
        # print(float(regular_loss), file=regular_energy)

    print(
        'iter:{:3>d} total_loss = {:.7f}, direct_error = {:.7f}, boundry_laplace_error = {:.7f}, gauss_error = {:.12f},regular_error = {:.7f}, gauss_weight = {:.5f}, boundry_laplace_weight = {:.5f}'
        .format(iter_i, loss, direct_error, boundry_laplace_error, gauss_error, s_sum, gauss_weight, boundry_laplace_weight))

    if optim_view_id == 1:
        out_cords = torch.cat([coordinates,t_pixels],dim=-1)
    elif optim_view_id == 2:
        out_cords = torch.cat([t_pixels,coordinates[:,[1,0]]],dim=-1) 
    else:
        out_cords = torch.cat([coordinates[:,1:2],t_pixels,coordinates[:,0:1]],dim=-1) 
    return  W*out_cords, direct_error, gauss_error

# def calculate_lowrank_H1(H1):
#     temp_H1 = H1.cpu()
#     u, s, v = torch.svd(temp_H1)
#     return s.sum(dim=1).mean()

# def get_gradients_and_H1(gi, hi):
#     hessians = torch.zeros(hi.shape[0], 2, 2).to('cuda')
#     hessians[:, 0, 0] = hi[:, 0]
#     hessians[:, 0, 1] = hi[:, 2]
#     hessians[:, 1, 0] = hi[:, 2]
#     hessians[:, 1, 1] = hi[:, 1]

#     H1 = torch.cat((hessians, torch.zeros((gi.size()[0], 1, 2)).cuda()), dim=1)

#     return H1

def calculate_lowrank_H1(H1):
    temp_H1 = H1.cpu()
    try:
        u, s, v = torch.svd(temp_H1)
        s_sum = s.sum(dim=1).mean()
        s[:, 3] = 0
    except:
        print(temp_H1)
    return torch.bmm(torch.bmm(u, torch.diag_embed(s)), v.transpose(1, 2)).cuda().detach().clone(), s_sum.cuda().detach().clone()

def get_gradients_and_H1(gi, hi):
    hessians = torch.zeros(hi.shape[0], 2, 2).to('cuda')
    hessians[:, 0, 0] = hi[:, 0]
    hessians[:, 0, 1] = hi[:, 2]
    hessians[:, 1, 0] = hi[:, 2]
    hessians[:, 1, 1] = hi[:, 1]

    H1 = torch.cat((hessians, torch.zeros((gi.size()[0], 1, 2)).cuda()), dim=1)
    H1 = torch.cat((H1, torch.cat(
        (torch.zeros((gi.size()[0], 2, 1)).cuda(), torch.zeros(gi.size()[0], 1, 1).cuda()), dim=1)),
                   dim=-1)
    gi = torch.cat((gi,-torch.ones(gi.size()[0], 1).cuda()), dim=1)

    H1 = torch.cat((H1, torch.reshape(gi, (gi.size()[0], 1, -1))), dim=1)
    H1 = torch.cat((H1, torch.cat(
        (torch.reshape(gi, (gi.size()[0], -1, 1)), torch.zeros(gi.size()[0], 1, 1).cuda()), dim=1)),
                dim=-1)
    return H1

def opt_light_transformer(light_transformer, target_intensity, img, optim, iter_i):
    o_pixels = light_transformer.pixels
    t_pixels = light_transformer()
    #### 得到所有像素点的个数
    total_pixels = o_pixels.size()[0]
    #### 将渲染图片的每个像素点重复总像素点个数
    render_pixels = o_pixels.detach().clone().repeat_interleave(total_pixels, dim=0)
    #### 将光线目标像素复制总像素点个数
    target_pixels = t_pixels.repeat([total_pixels, 1]).cuda()
    old_distance = (target_pixels - render_pixels.cuda()).norm(dim=-1)
    distance = torch.full(old_distance.size(), 3).cuda()
    coef = 2 * int(iter_i / meta_params['in_epochs']) + 3

    # if coef > 21:
    #     coef = 22
    # coef = 20

    max_dis = 75 / coef
    distance[old_distance < max_dis] = old_distance[old_distance < max_dis]
    distance = 1 / (1 + torch.exp((-0.5 + distance) * coef))
    distance = distance.reshape(-1, total_pixels)
    light_intensity = distance.sum(dim=-1)
    target_intensity = torch.tensor(target_intensity).float().cuda().reshape(-1)
    intensity_error = torch.pow(light_intensity.cuda() - target_intensity, 2).sum()

    # d_output = torch.ones_like(intensity_error, requires_grad=False, device=intensity_error.device)
    # gradients = torch.autograd.grad(
    #     outputs=intensity_error,
    #     inputs=target_pixels,
    #     grad_outputs=d_output,
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True)[0]

    displace = (t_pixels - o_pixels)
    z_coord = torch.zeros(displace.size()[0], 1)
    displace = torch.cat([displace, z_coord], dim=-1).cuda()
    feature_axis1 = torch.tensor([1., 0., 0.]).repeat(displace.size()[0], 1).cuda()
    feature_axis2 = torch.tensor([-1., 0., 0.]).repeat(displace.size()[0], 1).cuda()
    feature_axis3 = torch.tensor([0., 1., 0.]).repeat(displace.size()[0], 1).cuda()
    feature_axis4 = torch.tensor([0., -1., 0.]).repeat(displace.size()[0], 1).cuda()
    direct_loss = displace.cross(feature_axis1).norm(dim=-1) * \
                   displace.cross(feature_axis2).norm(dim=-1) * \
                   displace.cross(feature_axis3).norm(dim=-1) * \
                   displace.cross(feature_axis4).norm(dim=-1)
    direct_loss = direct_loss.mean()

    # regular_loss = displace.norm(dim=-1).mean()
    regular_loss = (t_pixels.cuda() - torch.tensor([[img.shape[0] / 2, img.shape[1] / 2]]).cuda()).norm().mean()

    loss = meta_params['opt_intensity'] * intensity_error + meta_params['opt_direct'] * direct_loss + \
           meta_params['opt_regular'] * regular_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    print(
        'iter:{:3>d} total_loss = {:.5f} intensity_error = {:.5f}, direct_loss = {:.5f}, regular_loss = {:.5f}'
        .format(iter_i, loss, intensity_error, direct_loss, regular_loss))
    return intensity_error, direct_loss, regular_loss, light_intensity


def post_processing(target_intensity, final_intensity, pixels, final_pixels):
    ### K, 2
    pixels = pixels.reshape(-1, 2)

    ### 都是整数的
    target_intensity = torch.tensor(target_intensity.reshape(-1))
    final_intensity = final_intensity.reshape(-1)
    intensity_error = target_intensity - final_intensity


    while intensity_error.min() < 0:
        # 误差找到小于0的位置
        min_error_idx = intensity_error.argmin()
        # 找到所有投射到这个位置的光线id
        error_light_idx = ((final_pixels - pixels[min_error_idx]).abs().sum(dim=-1) == 0).nonzero()
        # 计算它们的偏移距离
        distance = (pixels[error_light_idx] - pixels[min_error_idx]).float().norm(dim=-1)
        # 找到偏移距离最大的那个
        max_distance_idx = distance.argmax()
        # 将它的投影位置设为（-1，-1）
        final_pixels[int(error_light_idx[max_distance_idx])] = torch.tensor([-1, -1])
        # 将该位置最终光强减1
        final_intensity[min_error_idx] -= 1
        # 更新光强误差
        intensity_error = target_intensity - final_intensity

    # 找到所有误差大于0的位置
    all_error_idx = (intensity_error > 0).nonzero()
    # 找到所有无映射的棱柱位置
    all_no_light_idx = ((final_pixels - torch.tensor([[-1, -1]])).abs().sum(dim=-1) == 0).nonzero()
    # 取出误差大于0的坐标
    all_error_pixel_coord = pixels[all_error_idx].reshape(-1, 2)
    # 取出无映射棱柱的坐标
    all_no_light_pixel_coord = pixels[all_no_light_idx].reshape(-1, 2)
    # 像素坐标复制
    cols_error_pixel = all_error_pixel_coord.repeat_interleave(all_no_light_pixel_coord.size()[0], dim=0).float()
    # 棱柱坐标复制
    rows_light_pixel = all_no_light_pixel_coord.repeat(all_error_pixel_coord.size()[0], 1).float()
    # 计算距离
    candidate_dist = (cols_error_pixel - rows_light_pixel).norm(dim=-1).reshape(all_error_pixel_coord.size()[0],
                                                                                all_no_light_pixel_coord.size()[0])

    while intensity_error.max() > 0:
        # 取最小位置
        min_f = candidate_dist.argmin()
        cols = int(min_f % all_no_light_pixel_coord.size()[0])
        rows = int(min_f / all_no_light_pixel_coord.size()[0])
        # 棱柱id
        modified_light_idx = all_no_light_idx[cols]
        # 像素id
        target_pixel = all_error_idx[rows]
        # 修改映射
        final_pixels[int(modified_light_idx)] = pixels[int(target_pixel)]
        # 将该位置最终光强加1
        final_intensity[int(target_pixel)] += 1


        # ###################################
        # final_intensity = torch.zeros([16, 16])
        # for i in range(0, final_pixels.size()[0]):
        #     if final_pixels[i, 0] != -1 and final_pixels[i, 1] != -1:
        #         final_intensity[final_pixels[i, 0].long(), final_pixels[i, 1].long()] += 1
        # final_intensity = final_intensity.reshape(-1)
        # if intensity_error.abs().sum() == 6:
        #     print("aaaaa")
        # ###################################

        # 更新光强误差
        intensity_error = target_intensity - final_intensity
        # 将最小位置值设为65535
        if intensity_error[int(all_error_idx[rows])] == 0:
            candidate_dist[rows] = 65535
        else:
            candidate_dist[rows, cols] = 65535
        candidate_dist[:, cols] = 65535
        print(intensity_error.abs().sum())

    return final_pixels


    #
    # t_pixels = torch.zeros(16, 16, 2).reshape(-1, 2, 1)
    # img = torch.zeros(16, 16)
    # final_intensity = torch.zeros([16, 16])
    #
    # for i in range(0, t_pixels.size()[0]):
    #     if t_pixels[i, 0].long() not in range(0, img.size()[0]) or t_pixels[i, 1].long() not in range(0, img.size()[1]) or (
    #             t_pixels[i] - final_pixels[i]).norm() >= 0.5 or final_pixels[i, 0] not in range(0, img.size()[0]) or \
    #             final_pixels[i, 1].long() not in range(0, img.size()[1]):
    #         final_pixels[i] = torch.tensor([-1, -1])
    # for i in range(0, t_pixels.size()[0]):
    #     if final_pixels[i, 0] != -1 and final_pixels[i, 1] != -1:
    #         final_intensity[final_pixels[i, 0].long(), final_pixels[i, 1].long()] += 1
    #
    # intensity_error = target_intensity - final_intensity.reshape(-1)
    # aaa = intensity_error.min()
    # print(aaa)

def render_trick(trained_intensity, target_intensity):
    target_intensity = torch.tensor(target_intensity.reshape(-1)).float()
    trained_intensity1 = trained_intensity.detach().clone().reshape(-1)
    trained_intensity1 = torch.where(target_intensity >= trained_intensity1, trained_intensity1, target_intensity)
    return trained_intensity1

def count_stick_type_and_number(pixel_map):
    stick_type = []
    stick_number = []
    matrix_stick_type = torch.full([pixel_map.size()[0], pixel_map.size()[1]], -1)
    for i in range(0, pixel_map.size()[0]):
        for j in range(0, pixel_map.size()[1]):
            target_pixel = pixel_map[i, j]
            if target_pixel[0] != -1:
                target_pixel = target_pixel - torch.tensor([i, j])
                if (target_pixel[0] <= 0 and target_pixel[1] > 0) or (target_pixel[0] >= 0 and target_pixel[1] < 0):
                    target_pixel = torch.tensor([target_pixel[1].abs(), target_pixel[0].abs()])
                if (target_pixel[0] >= 0 and target_pixel[1] >= 0) or (target_pixel[0] <= 0 and target_pixel[1] <= 0):
                    target_pixel = target_pixel.abs()
            is_found = False
            for k in range(0, len(stick_type)):
                if stick_type[k][0] == int(target_pixel[0]) and stick_type[k][1] == int(target_pixel[1]):
                    stick_number[k] += 1
                    is_found = True
                    matrix_stick_type[i, j] = k
                continue
            if not is_found:
                stick_type.append([int(target_pixel[0]), int(target_pixel[1])])
                stick_number.append(1)
                matrix_stick_type[i, j] = len(stick_type) - 1

    # fp_stick_type = open('stick_type.txt', 'w')
    # fp_target_pixel = open('target_pixel.txt', 'w')
    # for i in range(0, matrix_stick_type.size()[0]):
    #     for j in range(0, matrix_stick_type.size()[1]):
    #         print(int(matrix_stick_type[i, j]), file=fp_stick_type, end=' ')
    #         print('('+str(int(pixel_map[i, j, 0]) - i)+', ' + str(int(pixel_map[i, j, 1]) - j) + ')', file=fp_target_pixel, end=' ')
    #     print(' ', file=fp_stick_type)
    #     print(' ', file=fp_target_pixel)
    # fp_stick_type.close()
    # fp_target_pixel.close()
    return stick_type, stick_number, matrix_stick_type

def get_trained_intensity(light_transformer, pixels, img):
    # o_pixels = pixels.float()
    # model_in = {'coords': o_pixels}
    # model_output = light_transformer(model_in)
    # t_pixels = model_output['model_out'] + o_pixels.cuda()
    # t_pixels = t_pixels.cpu()
    # #### 得到所有像素点的个数
    # total_pixels = o_pixels.size()[0]
    # #### 将渲染图片的每个像素点重复总像素点个数
    # render_pixels = o_pixels.detach().clone().repeat_interleave(total_pixels, dim=0)
    # #### 将光线目标像素复制总像素点个数
    # target_pixels = t_pixels.repeat([total_pixels, 1])
    # distance = (target_pixels - render_pixels).norm(dim=-1)
    # final_intensity = distance.detach().clone()
    # final_intensity[distance <= 0.5] = 1
    # final_intensity[distance > 0.5] = 0
    # return final_intensity

    o_pixels = pixels.float()
    model_in = {'coords': o_pixels}
    model_output = light_transformer(model_in)
    t_pixels = model_output['model_out'] + o_pixels.cuda()
    t_pixels = t_pixels.cpu()
    final_pixels = t_pixels + torch.tensor([[0.5, 0.5]])
    final_pixels = final_pixels.int()
    final_intensity = torch.zeros([img.size()[0], img.size()[1]])

    for i in range(0, t_pixels.size()[0]):
        if t_pixels[i, 0].long() not in range(0, img.size()[0]) or t_pixels[i, 1].long() not in range(0, img.size()[1]) or (
                t_pixels[i] - final_pixels[i]).norm() >= 0.5 or final_pixels[i, 0] not in range(0, img.size()[0]) or \
                final_pixels[i, 1].long() not in range(0, img.size()[1]):
            final_pixels[i] = torch.tensor([-1, -1])
    for i in range(0, t_pixels.size()[0]):
        if final_pixels[i, 0] != -1 and final_pixels[i, 1] != -1:
            final_intensity[final_pixels[i, 0].long(), final_pixels[i, 1].long()] += 1


    # for i in range(0, t_pixels.size()[0]):
    #     if t_pixels[i, 0].long() in range(0, img.size()[0]) and t_pixels[i, 1].long() in range(0, img.size()[1]) and (t_pixels[i] - final_pixels[i]).norm() < 0.5:
    #         final_intensity[t_pixels[i, 0].long(), t_pixels[i, 1].long()] += 1
    return final_intensity, final_pixels

def sigmoid_function(x_tensor, shift, scale):
    return 1 / (1 + torch.exp(-scale * (x_tensor - shift)))

def find_clusters(y):
    clusters = []
    start = None
    current_val = y[0]

    for i in range(len(y)):
        if y[i] == current_val:
            if start is None:
                start = i
        else:
            if start is not None:
                clusters.append((current_val, start, i - 1))
            start = i
            current_val = y[i]

    if start is not None:
        clusters.append((current_val, start, len(y) - 1))
    return clusters

# 定义可微分的函数
def orth_pic_height_fun(y,x_tensor,scale,W):
    # 使用sigmoid函数构造平滑函数
    clusters = find_clusters(y)
    x_tensor_W = x_tensor*W
    y_smooth = 0.0
    for value, start, end in clusters:
        if value == 1:
            y_smooth += sigmoid_function(x_tensor_W, start-0.5, scale) - sigmoid_function(x_tensor_W, end+0.5, scale)

    return y_smooth

# 创建矩形区域的正三角形网格
def generate_regular_triangle_mesh(img, triangle_size):
    width, height = img.shape
    # 生成正三角形网格的顶点
    rows = int(height / (triangle_size * np.sqrt(3) / 2)) + 1
    cols = int(width / triangle_size) + 1

    points = []
    for i in range(rows):
        for j in range(cols):
            x = j * triangle_size + (i % 2) * (triangle_size / 2)
            y = i * (triangle_size * np.sqrt(3) / 2)
            points.append([x, y])

    points = np.array(points)

    # 生成三角形 以为 顶点的邻接顶点集合 按照论文Developability of Heightfields via Rank Minimization 3.1小节图的顺序
    triangles = []
    adj_vertices = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            if i > 0 and j > 0 and i < rows - 1 and j < cols - 1:
                # 创建邻接点的索引
                if i % 2 == 0:
                    adj_vertices.append([idx + cols - 1, idx + cols, idx - 1,
                        idx, idx + 1, idx - cols - 1, idx - cols
                    ])
                else:
                    adj_vertices.append([idx + cols, idx + cols + 1, idx - 1,
                        idx, idx + 1, idx - cols, idx - cols + 1
                    ])
            if i % 2 == 0:
                triangles.append([idx, idx + 1, idx + cols])
                triangles.append([idx + 1, idx + cols + 1, idx + cols])
            else:
                triangles.append([idx, idx + 1, idx + cols + 1])
                triangles.append([idx, idx + cols + 1, idx + cols])

    triangles = np.array(triangles)
    adj_vertices = np.array(adj_vertices)

    colors_feature = F.grid_sample(torch.Tensor(img).unsqueeze(axis=0).unsqueeze(axis=0), torch.Tensor(2.0*points/(width-1.0)-1).unsqueeze(axis=0).unsqueeze(axis=0).permute(0, 2, 1, 3), padding_mode='border', align_corners=False)
    p2del = np.where(colors_feature.squeeze() > 127)
    mask = np.isin(triangles, p2del).any(axis=1)
    mask_adj = np.isin(adj_vertices, p2del).any(axis=1)
    # 由于构造问题 tirangles 中包含0和最后一个元素，而adj_vertices中没有
    # diff = set(np.unique(triangles)) - set(np.unique(adj_vertices))
    # print(diff)
    # 中心点要删除的点，删除该项连接信息，不需要删除中心点的就是边界点
    mask_adj_w_boundry = np.isin(adj_vertices[:,3], p2del)
    adj_vertices_w_boundry = adj_vertices[~mask_adj_w_boundry]
    validate_adj = ~np.isin(adj_vertices_w_boundry, p2del)
    validate_adj_cnt = np.sum(validate_adj,axis=1)
    boundry_mask = (validate_adj_cnt>2)#*(validate_adj_cnt<6) #只保留至少有2个邻居的边界点
    adj_boundry = adj_vertices_w_boundry[boundry_mask]
    adj_boundry_mask = validate_adj[boundry_mask]
    # 删除这些三角形
    new_triangles = triangles[~mask]
    new_adj_vertices = adj_vertices[~mask_adj]

    # 找到所有参与三角形的顶点
    used_indices = np.unique(adj_vertices_w_boundry)

    # 选择参与三角形的顶点
    unique_vertices = points[used_indices]

    # 更新三角形的顶点索引
    vertex_map = {old: new for new, old in enumerate(used_indices)}
    updated_triangles = np.array([[vertex_map[vertex] for vertex in triangle] for triangle in new_triangles])
    updated_adj_vertices = np.array([[vertex_map[vertex] for vertex in adj_vert] for adj_vert in new_adj_vertices])
    updated_adj_boundry = np.array([[vertex_map[vertex] for vertex in adj_vert] for adj_vert in adj_boundry])
    return unique_vertices, updated_triangles, updated_adj_vertices, updated_adj_boundry, torch.Tensor(adj_boundry_mask).to('cuda')

def calc_centroids(voxel_cords):
    # 获取所有 z 的位置
    z_values = np.arange(voxel_cords.shape[2])

    # 对 arr 进行广播扩展，z_values 会广播到 arr 的每个 (x, y) 上
    # 计算每个 (x, y) 对应的 z 轴加权重心
    weighted_sum_z = np.sum(voxel_cords * z_values, axis=2)  # 计算 z 的加权和，形状为 [x, y]
    sum_weights = np.sum(voxel_cords, axis=2)  # 计算每个 (x, y) 的 z 总权重，形状为 [x, y]

    # 计算重心位置，避免被除以 0 的情况
    centroids_z = np.divide(weighted_sum_z, sum_weights, where=(sum_weights != 0))

    # 将没有实体点的重心标记为 None
    centroids_z[sum_weights == 0] = None

        # 计算每个 (x, y) 对应的 z 轴加权重心
    weighted_sum_z = np.sum(voxel_cords * z_values, axis=2)  # 计算 z 的加权和，形状为 [x, y]
    sum_weights = np.sum(voxel_cords, axis=2)  # 计算每个 (x, y) 的 z 总权重，形状为 [x, y]

    # 计算重心位置，避免被除以 0 的情况
    centroids_z = np.divide(weighted_sum_z, sum_weights, where=(sum_weights != 0))

    # 将没有实体点的重心标记为 None
    centroids_z[sum_weights == 0] = None


def main():
    model = run_model
    # scaled_img = scale_img(img, 30)
    # cv2.imshow("img", scaled_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    writer = SummaryWriter(log_dir=os.path.join(meta_params['data_root'], meta_params['experiment_name'],'logs'))
    if model == 'optimize':
        img = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '.png'))
        intensity = intensity_matrix(img)
        img = torch.from_numpy(img).cuda()
        light_transformer = OptModel.Model(img)

        optim = torch.optim.Adam(lr=0.001, params=light_transformer.parameters())
        for iter_i in tqdm.tqdm(range(int(meta_params['iter_num']))):
            intensity_error, direct_loss, regular_loss, light_intensity = opt_light_transformer(light_transformer, intensity, img, optim, iter_i)

        o_pixels = light_transformer.pixels
        t_pixels = light_transformer()
        #### 得到所有像素点的个数
        total_pixels = o_pixels.size()[0]
        #### 将渲染图片的每个像素点重复总像素点个数
        render_pixels = o_pixels.detach().clone().repeat_interleave(total_pixels, dim=0)
        #### 将光线目标像素复制总像素点个数
        target_pixels = t_pixels.repeat([total_pixels, 1])
        distance = (target_pixels - render_pixels).norm(dim=-1)
        final_intensity = distance.detach().clone()
        final_intensity[distance <= 0.5] = 1
        final_intensity[distance > 0.5] = 0
        final_intensity = final_intensity.reshape(-1, total_pixels)
        final_intensity = final_intensity.sum(dim=-1)

        final_intensity = final_intensity / final_intensity.max() * 255
        final_intensity = final_intensity.reshape(img.size()[0], img.size()[1], 1)
        final_intensity = torch.cat([final_intensity, final_intensity, final_intensity], dim=-1)
        final_intensity = np.around(final_intensity.detach().numpy())
        scaled_img = scale_img(final_intensity, 30)
        cv2.imshow("img", scaled_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        cv2.imwrite(os.path.join(os.path.join(exp_path, 'train'), 'opt_result.png'), scaled_img)


    elif model == 'train':
        img1_gray = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '/imgs/aligned_view1_128.png'), cv2.IMREAD_GRAYSCALE)#xy
        img2_gray = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '/imgs/aligned_view2_128.png'), cv2.IMREAD_GRAYSCALE)#xz
        img3_gray = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '/imgs/aligned_view3_128.png'), cv2.IMREAD_GRAYSCALE)#yz
        _,img1_binary=cv2.threshold(np.transpose(img1_gray), 127, 1, cv2.THRESH_BINARY)#对于二值化的图片，0是纯黑色，255是纯白色
        _,img2_binary=cv2.threshold(np.transpose(img2_gray), 127, 1, cv2.THRESH_BINARY)
        _,img3_binary=cv2.threshold(np.transpose(img3_gray), 127, 1, cv2.THRESH_BINARY)

        # 图片尺寸
        H, W = img1_binary.shape

        # 点云初始化
        voxel_cords = np.zeros((H,H,H),dtype=int)
        voxel_cords[img1_binary == 0, :] = 1
        indices = np.where(img2_binary == 0)
        voxel_cords[:,indices[1],indices[0]] = 1
        indices = np.where(img3_binary == 0)
        voxel_cords[indices[1],:,indices[0]] = 1

        # # 创建点云对象
        # point_cloud = o3d.geometry.PointCloud()
        # occupied_index = np.argwhere(cords > 0.6)
        # # 设置点云数据
        # point_cloud.points = o3d.utility.Vector3dVector(occupied_index)
        # # 将点云保存为ply文件
        # o3d.io.write_point_cloud('point_cloud.ply', point_cloud)  # 保存为PLY格式文件




        # 设置当前要拟合的的投影图像编号
        optim_view_id = 2
        if optim_view_id == 1:
            cur_img = img1_gray
            pro_img1 = img2_binary
            pro_img2 = img3_binary
        elif optim_view_id == 2:
            cur_img = img2_gray
            pro_img1 = img1_binary
            pro_img2 = img3_binary

            scanline1 = (img2_binary==0).sum(axis=0)>0
            scanproimg1 = scanline1[:,np.newaxis]*(pro_img1==0).T
            scansum_x = scanproimg1.sum(axis=0)

            scanline2 = (img2_binary==0).sum(axis=1)>0
            scanproimg2 = scanline2[:,np.newaxis]*(pro_img2==0)
            scansum_y = scanproimg2.sum(axis=0)
            scansum_xy =  scansum_x + scansum_y 
            init_z = W-1- np.array(np.where( scansum_xy == scansum_xy.max())).mean()  
        else:
            cur_img = img3_gray
            pro_img1 = img1_binary
            pro_img2 = img2_binary

            scanline1 = (img3_binary==0).sum(axis=0)>0
            scanproimg1 = scanline1[:,np.newaxis]*(pro_img1==0)
            scansum_x = scanproimg1.sum(axis=0)

            scanline2 = (img3_binary==0).sum(axis=1)>0
            scanproimg2 = scanline2[:,np.newaxis]*(pro_img2==0)
            scansum_y = scanproimg2.sum(axis=0)
            scansum_xy =  scansum_x + scansum_y 
            init_z =W-1- np.array(np.where( scansum_xy == scansum_xy.max())).mean()        

        sq3 = np.sqrt(3.0)
        h = 0.25
        points, triangles, adj_vertices, adj_boundry, adj_boundry_mask = generate_regular_triangle_mesh(cur_img, h)
        B = 1/(6*h**2)*torch.Tensor([[0,0,6,-12,6,0,0],[4,4,-2,-12,-2,4,4],[-2*sq3,2*sq3,0,0,0,2*sq3,-2*sq3],
                                     [-h,h,-2*h,0,2*h,-h,h],[sq3*h,sq3*h,0,0,0,-sq3*h,-sq3*h],
                                     [0,0,0,6*h**2,0,0,0]]).to('cuda')
        # B = 1/(6*h**2)*torch.Tensor([[0,0,3,-6,3,0,0],[2,2,-1,-6,-1,2,2],[-2*sq3,2*sq3,0,0,0,2*sq3,-2*sq3],
        #                         [-h,h,-2*h,0,2*h,-h,h],[sq3*h,sq3*h,0,0,0,-sq3*h,-sq3*h],
        #                         [0,0,0,6*h**2,0,0,0]]).to('cuda')
        # 可视化结果
        # plt.figure(figsize=(6, 6))
        # plt.triplot(points[:, 0], points[:, 1], triangles, 'go-', lw=1)
        # plt.gca().set_aspect('equal')
        # plt.show()

        solid_img1_cords = np.argwhere(img1_binary == 0)
        solid_img2_cords = np.argwhere(img2_binary == 0)
        solid_img3_cords = np.argwhere(img3_binary == 0)

        # 创建点云对象
        img1_pcd = o3d.geometry.PointCloud()
        zeros_to_add = np.zeros((solid_img1_cords.shape[0], 1))
        img1_extended = np.concatenate((solid_img1_cords, zeros_to_add), axis=1)
        img1_pcd.points = o3d.utility.Vector3dVector(img1_extended)
        img1_pcd.paint_uniform_color(np.array([0.00, 0.33, 0.67]))

        img2_pcd = o3d.geometry.PointCloud()
        zeros_to_add = np.zeros((solid_img2_cords.shape[0], 1))
        img2_extended = np.concatenate((zeros_to_add, solid_img2_cords[:, [1, 0]]), axis=1)
        img2_pcd.points = o3d.utility.Vector3dVector(img2_extended)
        img2_pcd.paint_uniform_color(np.array([1.0, 1.0, 0.0]))

        img3_pcd = o3d.geometry.PointCloud()
        ones_to_add = img3_gray.shape[0]*np.ones((solid_img3_cords.shape[0], 1))
        img3_extended = np.concatenate((solid_img3_cords[:, [1]], ones_to_add, solid_img3_cords[:, [0]]), axis=1)
        img3_pcd.points = o3d.utility.Vector3dVector(img3_extended)
        img3_pcd.paint_uniform_color(np.array([0.0, 1.0, 1.0]))
        o3d.io.write_point_cloud(f"./images_data/{meta_params['experiment_name']}/Tri_output/point_cloud.ply", img1_pcd+img2_pcd+img3_pcd)  # 保存为PLY格式文件

        light_transformer = TransformNetwork(**meta_params)
        # light_transformer.set_W(0)
        # light_transformer = nn.DataParallel(light_transformer)

        light_transformer.cuda()
        # light_transformer = load_TransformNetwork(os.path.join(os.path.join(exp_path, 'train'), 'light_transformer_network_final.pth'))
        # light_transformer.eval()
        iter_num=int(meta_params['iter_num'])
        optim = torch.optim.Adam(lr=0.00001, params=light_transformer.parameters())
        first_output = 0.0
        for iter_i in tqdm.tqdm(range(iter_num)):
            cords, direct_error, gauss_error = train_light_transformer(light_transformer, points/W, triangles, adj_vertices, adj_boundry, adj_boundry_mask, B, orth_pic_height_fun, pro_img1, pro_img2, optim, iter_i, writer, optim_view_id, init_z/W, first_output)
            # if iter_i % 100 == 0:
            #     torch.save(light_transformer.state_dict(),
            #                os.path.join(checkpoints_dir, 't_n_%05d_%.5f_%.5f_%.5f.pth' % (
            #                    iter_i, intensity_error, direct_loss, regular_loss)))
            if iter_i % 100 == 0:
                # 创建点云对象
                combined_mesh = o3d.geometry.TriangleMesh()
            
                # 设置点云数据
                combined_mesh.vertices = o3d.utility.Vector3dVector(cords.detach().cpu().numpy())
                combined_mesh.triangles = o3d.utility.Vector3iVector(triangles)
                combined_mesh.paint_uniform_color(np.array([1.0, 0.0, 0.0]))
                # 将点云保存为ply文件

                o3d.io.write_triangle_mesh(f"./images_data/{meta_params['experiment_name']}/Tri_output/mesh_{iter_i}_derror_{direct_error:.6f}_gerror{gauss_error:.6f}.ply", combined_mesh)

        torch.save(light_transformer.state_dict(),
                   os.path.join(os.path.join(exp_path, run_model), 'light_transformer_network_final.pth'))
        
    elif model == 'show_train':
        target_img = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '.png'))
        cv2.imwrite(os.path.join(os.path.join(exp_path, 'train'), 'input.png'), scale_img(target_img, 30))
        img = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '.png'))
        intensity = intensity_matrix(img)
        img = torch.from_numpy(img).cuda()

        color_proportion = img.cpu().detach().clone().float()
        color_proportion[color_proportion == 0] = 1
        color_proportion = color_proportion / color_proportion.sum(dim=-1).reshape(color_proportion.size()[0], -1, 1)


        rows = img.shape[0]
        cols = img.shape[1]
        row_ids = torch.arange(0, rows, 1)
        col_ids = torch.arange(0, cols, 1)
        row_ids = row_ids.repeat_interleave(cols)
        col_ids = col_ids.repeat(rows)
        pixels = torch.tensor(list(map(lambda x, y: [x, y], row_ids.tolist(), col_ids.tolist())))

        light_transformer = load_TransformNetwork(os.path.join(os.path.join(exp_path, 'train'), 'light_transformer_network_final.pth'))
        light_transformer.eval()


        # o_pixels = pixels.float()
        # model_in = {'coords': o_pixels}
        # model_output = light_transformer(model_in)
        # t_pixels = model_output['model_out'] + o_pixels.cuda()
        # t_pixels = t_pixels.cpu()
        # #### 得到所有像素点的个数
        # total_pixels = o_pixels.size()[0]
        # #### 将渲染图片的每个像素点重复总像素点个数
        # render_pixels = o_pixels.detach().clone().repeat_interleave(total_pixels, dim=0)
        # #### 将光线目标像素复制总像素点个数
        # target_pixels = t_pixels.repeat([total_pixels, 1])
        # distance = (target_pixels - render_pixels).norm(dim=-1)
        # final_intensity = distance.detach().clone()
        # final_intensity[distance <= 0.5] = 1
        # final_intensity[distance > 0.5] = 0

        total_pixels = pixels.size()[0]
        final_intensity, final_pixel = get_trained_intensity(light_transformer, pixels, img)

        trained_intensity = render_trick(final_intensity, intensity)
        trained_intensity = trained_intensity.reshape(img.size()[0], img.size()[1], 1)
        trained_intensity = trained_intensity * color_proportion
        trained_intensity = trained_intensity / trained_intensity.max() * 255
        # trained_intensity = torch.cat([trained_intensity, trained_intensity, trained_intensity], dim=-1)
        trained_intensity = np.around(trained_intensity.detach().numpy())
        scaled_img = scale_img(trained_intensity, 30)
        cv2.imwrite(os.path.join(os.path.join(exp_path, 'train'), 'train_result.png'), scaled_img)


        final_pixel = post_processing(intensity, final_intensity, pixels, final_pixel)

        output_pixel = final_pixel.reshape([img.size()[0], img.size()[1], 2])
        stick_type, stick_number, matrix_stick_type = count_stick_type_and_number(output_pixel)
        utils.save_pkl_data(os.path.join(intermediate_path, 'output_pixel_map'), output_pixel)
        utils.save_pkl_data(os.path.join(intermediate_path, 'stick_type'), stick_type)
        utils.save_pkl_data(os.path.join(intermediate_path, 'stick_number'), stick_number)
        utils.save_pkl_data(os.path.join(intermediate_path, 'matrix_stick_type'), matrix_stick_type)

        final_intensity = torch.zeros([img.size()[0], img.size()[1]])
        for i in range(0, final_pixel.size()[0]):
            if final_pixel[i, 0] != -1 and final_pixel[i, 1] != -1:
                final_intensity[final_pixel[i, 0].long(), final_pixel[i, 1].long()] += 1



        print((final_intensity - torch.tensor(intensity)).abs().sum())
        # final_intensity = final_intensity.reshape(-1, total_pixels)
        # final_intensity = final_intensity.sum(dim=-1)




        final_intensity = final_intensity.reshape(img.size()[0], img.size()[1], 1)
        final_intensity = final_intensity * color_proportion
        final_intensity = final_intensity / final_intensity.max() * 255
        # final_intensity = torch.cat([final_intensity, final_intensity, final_intensity], dim=-1)
        final_intensity = np.around(final_intensity.detach().numpy())
        scaled_img = scale_img(final_intensity, 30)
        cv2.imwrite(os.path.join(os.path.join(exp_path, 'train'), 'result.png'), scaled_img)
        # cv2.imshow("img", scaled_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()




    elif model == 'OptSurface':
        pixel_map = utils.load_pkl_data(os.path.join(intermediate_path, 'output_pixel_map'))
        stick_type = utils.load_pkl_data(os.path.join(intermediate_path, 'stick_type'))
        stick_number = utils.load_pkl_data(os.path.join(intermediate_path, 'stick_number'))
        surface = utils.read_mesh('./data/surface', 'surface.obj')
        h = 40
        n = 1.515
        for i in range(0, len(stick_type)):
            print(float(i)/ len(stick_type))
            bad = False
            if stick_type[i] == [-1, -1]:
                continue

            x = str(stick_type[i][0])
            y = str(stick_type[i][1])
            stick_mesh_name = x + '_' + y + '.obj'
            try:
                stick_mesh = utils.read_mesh('./data/surface/stick_mesh', stick_mesh_name)
            except RuntimeError:
                surface_opter = opts(surface, h, n, torch.tensor(stick_type[i]))
                begin_f = 1 / (stick_type[i][0] * stick_type[i][0] + stick_type[i][1] * stick_type[i][1])
                # optim = torch.optim.Adam(lr=0.00001, params=surface_opter.parameters())
                f = lambda epoch: 15 if int(epoch / 5) * 0.2 + 0.5 > 15 else int(epoch / 5) * 0.2 + 0.5
                print()


                optim = torch.optim.Adam(lr=0.0005, params=surface_opter.parameters())
                target_T = []
                dist = 65535
                for iter_i in tqdm.tqdm(range(30000)):
                    if iter_i % 10000000 == 0:
                        target_T = surface_opter.compute_target_T().cuda()
                    error, dist = surface_opter(target_T)
                    print('dist = {:.5f}'.format(dist))
                    if dist < 2:
                        break
                    # if iter_i > 150 and dist < 0.001:
                    #     break
                    def closure():
                        error, dist = surface_opter(target_T)
                        optim.zero_grad()
                        dist.backward()
                        return dist
                    optim.step(closure)

                optim = LBFGS(surface_opter.parameters(), lr=0.5)
                schedular = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda= f)
                # target_T = []
                dist = 65535
                for iter_i in tqdm.tqdm(range(10000)):
                    # if iter_i % 10000 == 0:
                        # target_T = surface_opter.compute_target_T().cuda()
                    error, dist = surface_opter(target_T)
                    print('dist = {:.5f}'.format(dist))
                    if dist < 0.0005:
                        break
                    if iter_i > 100:
                        bad = True
                        break
                    def closure():
                        error, dist = surface_opter(target_T)
                        optim.zero_grad()
                        dist.backward()
                        return dist
                    optim.step(closure)
                    schedular.step()



                stick_origin = utils.read_mesh('./data/surface', 'stick.obj')
                stick_origin = surface_opter.output_mesh(os.path.join(os.path.join(exp_path, run_model) + '/' + str(i) +'.obj'), stick_origin)
                om.write_mesh('./data/surface/stick_mesh/' + stick_mesh_name, stick_origin, 10)
                if bad:
                    bad_stick_mesh_name = x + '_' + y + str(dist) + '.obj'
                    om.write_mesh('./data/surface/stick_mesh/' + bad_stick_mesh_name, stick_origin, 10)


    elif model == 'array':
        pixel_map = utils.load_pkl_data(os.path.join(intermediate_path, 'output_pixel_map'))
        matrix_stick_type = utils.load_pkl_data(os.path.join(intermediate_path, 'matrix_stick_type'))
        stick_type = utils.load_pkl_data(os.path.join(intermediate_path, 'stick_type'))


        img = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '.png'))
        intensity = intensity_matrix(img)
        number_light = torch.tensor(intensity)
        img = torch.from_numpy(img).cuda()

        color_proportion = img.cpu().detach().clone().float()
        color_proportion[color_proportion == 0] = 1
        color_proportion = color_proportion / color_proportion.sum(dim=-1).reshape(color_proportion.size()[0], -1, 1)

        intensity_out = torch.from_numpy(intensity).float().reshape(intensity.shape[0], intensity.shape[1], 1)
        intensity_out = intensity_out * color_proportion
        intensity_out = intensity_out / intensity_out.max() * 255
        # intensity_out = torch.cat([intensity_out, intensity_out, intensity_out], dim=-1)
        intensity_out = np.around(intensity_out.detach().numpy())


        #数字代表逆时针旋转90度的次数(0, -1
        #                        1, 0)
        matrix_stick_rotation = torch.zeros_like(matrix_stick_type)
        stick_num = 0
        for i in range(0, pixel_map.size()[0]):
            for j in range(0, pixel_map.size()[1]):
                if (pixel_map[i, j]-torch.tensor([-1, -1])).float().norm() == 0:
                    matrix_stick_rotation[i, j] = 0
                    continue
                real_map = pixel_map[i, j] - torch.tensor([i, j])
                rotation = 0
                cur_stick_type = int(matrix_stick_type[i, j])
                stick_map = torch.tensor(stick_type[cur_stick_type])
                while 1:
                    dist = (real_map - stick_map).float().norm()
                    if dist == 0:
                        matrix_stick_rotation[i, j] = rotation
                        break
                    if rotation == 4:
                        print('rotation error!')
                        exit(0)
                    stick_map = torch.tensor([-stick_map[1], stick_map[0]])
                    rotation += 1

        vertex_offset = np.array([0.5, 0.5, 0])
        stick_array = om.TriMesh()
        for i in range(0, pixel_map.size()[0]):
            for j in range(0, pixel_map.size()[1]):
                print('print (' + str(i)+', '+str(j)+')')
                cur_stick_type = int(matrix_stick_type[i, j])
                x = str(stick_type[cur_stick_type][0])
                y = str(stick_type[cur_stick_type][1])
                stick_mesh_name = x + '_' + y + '.obj'
                stick_model = utils.read_mesh('./data/surface/stick_mesh', stick_mesh_name)
                cur_rotation = matrix_stick_rotation[i, j]
                for v_iter in stick_model.vertices():
                    vertex_pos = stick_model.point(v_iter)
                    if cur_rotation == 1:
                        vertex_pos = vertex_pos - vertex_offset
                        vertex_pos = np.array([-vertex_pos[1], vertex_pos[0], vertex_pos[2]])
                        vertex_pos = vertex_pos + vertex_offset
                    if cur_rotation == 2:
                        vertex_pos = vertex_pos - vertex_offset
                        vertex_pos = np.array([-vertex_pos[0], -vertex_pos[1], vertex_pos[2]])
                        vertex_pos = vertex_pos + vertex_offset
                    if cur_rotation == 3:
                        vertex_pos = vertex_pos - vertex_offset
                        vertex_pos = np.array([vertex_pos[1], -vertex_pos[0], vertex_pos[2]])
                        vertex_pos = vertex_pos + vertex_offset
                    stick_model.set_point(v_iter, vertex_pos + np.array([i, j, 0]))
                stick_array = utils.add_mesh(stick_array, stick_model)

                target_pixel = pixel_map[i, j]
                lights = int(number_light[target_pixel[0], target_pixel[1]])
                pixel_color = intensity_out[target_pixel[0], target_pixel[1]]
                if lights == 0:
                    om.write_mesh(os.path.join(exp_path, run_model) + '/' + str(stick_num) + '_' + str(0)
                              + '_' + str(0) + '_' + str(0) + '.obj', stick_model, 10)
                else:
                    om.write_mesh(os.path.join(exp_path, run_model) + '/' + str(stick_num) + '_' + str(
                        int(pixel_color[2] / lights))
                                  + '_' + str(int(pixel_color[1] / lights)) + '_' + str(
                        int(pixel_color[0] / lights)) + '.obj', stick_model, 10)
                stick_num += 1
        om.write_mesh(os.path.join(exp_path, run_model) + '/' + 'result.obj', stick_array, 10)


    elif model == 'check':
        pixel_map = utils.load_pkl_data(os.path.join(intermediate_path, 'output_pixel_map'))
        stick_type = utils.load_pkl_data(os.path.join(intermediate_path, 'stick_type'))
        stick_number = utils.load_pkl_data(os.path.join(intermediate_path, 'stick_number'))
        type_name = 0
        surface = utils.read_mesh(os.path.join(os.path.join(exp_path, 'OptSurface')), str(type_name) +'.obj')
        h = 40
        n = 1.515
        points = surface.points()
        surface_opter = opts(surface, h, n, torch.tensor(stick_type[type_name]))
        surface_opter.check()

    elif model == 'test':
        img = cv2.imread(os.path.join(meta_params['data_root'], meta_params['experiment_name'] + '.png'))
        intensity = intensity_matrix(img)
        img = torch.from_numpy(img).cuda()

        color_proportion = img.cpu().detach().clone().float()
        color_proportion[color_proportion == 0] = 1
        color_proportion = color_proportion / color_proportion.sum(dim=-1).reshape(color_proportion.size()[0], -1, 1)


        intensity_out = torch.from_numpy(intensity).float().reshape(intensity.shape[0], intensity.shape[1], 1)
        intensity_out = intensity_out * color_proportion
        intensity_out = intensity_out / intensity_out.max() * 255
        # intensity_out = torch.cat([intensity_out, intensity_out, intensity_out], dim=-1)
        intensity_out = np.around(intensity_out.detach().numpy())
        scaled_img = scale_img(intensity_out, 30)
        cv2.imwrite(os.path.join(os.path.join(exp_path, 'test'), 'target.png'), scaled_img)

        rows = img.shape[0]
        cols = img.shape[1]
        row_ids = torch.arange(0, rows, 1)
        col_ids = torch.arange(0, cols, 1)
        row_ids = row_ids.repeat_interleave(cols)
        col_ids = col_ids.repeat(rows)
        pixels = torch.tensor(list(map(lambda x, y: [x, y], row_ids.tolist(), col_ids.tolist())))


        target_intensity = scale_img(intensity, 100) #图像的坐标范围是0到1600
        target_intensity = target_intensity[:100, :100]

        interplote_points_z = torch.full([10+1, 10+1], 40) #65X65个插值点，初始是40  曲面的坐标范围是0到64

        my_test = spline_surface.OptSurface(interplote_points_z, torch.from_numpy(target_intensity).type(torch.float64), 1.515)

        optim = torch.optim.Adam(lr=0.01, params=my_test.parameters())
        target_T = []
        dist = 65535
        for iter_i in tqdm.tqdm(range(30000)):
            target = my_test()
            dist = (target - torch.tensor([[3., 0.]])).pow(2).sum(dim=-1).mean()
            print('dist = {:.5f}'.format(dist))
            if dist < 2:
                break

            # if iter_i > 150 and dist < 0.001:
            #     break
            def closure():
                target = my_test()
                dist = (target - torch.tensor([[3., 0.]])).pow(2).sum(dim=-1).mean()
                optim.zero_grad()
                dist.backward()
                return dist

            optim.step(closure)


        optim = LBFGS(my_test.parameters(), lr=0.5)
        f = lambda epoch: 15 if int(epoch / 5) * 0.2 + 0.5 > 15 else int(epoch / 5) * 0.2 + 0.5
        schedular = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=f)
        # target_T = []
        dist = 65535
        for iter_i in tqdm.tqdm(range(10000)):
            # if iter_i % 10000 == 0:
            # target_T = surface_opter.compute_target_T().cuda()
            target = my_test()
            dist = (target - torch.tensor([[3., 0.]])).pow(2).sum(dim=-1).mean()
            print('dist = {:.5f}'.format(dist))
            if dist < 0.0005:
                break

            def closure():
                target = my_test()
                dist = (target - torch.tensor([[3., 0.]])).pow(2).sum(dim=-1).mean()
                optim.zero_grad()
                dist.backward()
                return dist

            optim.step(closure)
            schedular.step()

    return

if __name__ == '__main__':
    main()
