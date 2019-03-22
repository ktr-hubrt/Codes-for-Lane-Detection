#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:29
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_cluster.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中实例分割的聚类部分
"""
import numpy as np
import glog as log
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
# from lanenet_model import lanenet_postprocess
from scipy.stats import norm
from skimage import morphology
from scipy import optimize
from scipy import stats
import time
import warnings
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


class LaneNetCluster(object):
    """
    实例分割聚类器
    """

    def __init__(self):
        """

        """
        self._color_map = [
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100]),
                           np.array([255, 0, 0]),
                           ]
        pass

    @staticmethod
    def _cluster(prediction, bandwidth):
        """
        实现论文SectionⅡ的cluster部分
        :param prediction:
        :param bandwidth:
        :return:
        """
        ms = MeanShift(bandwidth, bin_seeding=True)
        # log.info('开始Mean shift聚类 ...')
        tic = time.time()
        try:
            ms.fit(prediction)
        except ValueError as err:
            # log.error(err)
            return 0, [], []
        # log.info('Mean Shift耗时: {:.5f}s'.format(time.time() - tic))
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        # log.info('聚类簇个数为: {:d}'.format(num_clusters))

        return num_clusters, labels, cluster_centers

    @staticmethod
    def _cluster_v2(prediction):
        """
        dbscan cluster
        :param prediction:
        :return:
        """
        db = DBSCAN(eps=10, min_samples=1500).fit(prediction)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        unique_labels = [tmp for tmp in unique_labels if tmp != -1]
        log.info('聚类簇个数为: {:d}'.format(len(unique_labels)))

        num_clusters = len(unique_labels)
        cluster_centers = db.components_
        #len(np.unique(DBSCAN(eps=10, min_samples=2000).fit(points).labels_))
        return num_clusters, db_labels, cluster_centers
        
    @staticmethod
    def _get_lane_area(binary_seg_ret, instance_seg_ret):
        """
        通过二值分割掩码图在实例分割图上获取所有车道线的特征向量
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 1)

        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])
            lane_coordinate.append([idx[0][i], idx[1][i]])

        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):
        """
        过滤实例车道线位置坐标点,假设车道线是连续的, 因此车道线点的坐标变换应该是平滑变化的不应该出现跳变
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x)

        return coord[idx[0]]

    @staticmethod
    def _lane_fit(lane_pts):
        """
        车道线多项式拟合
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        x_fit = []
        y_fit = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f1 = np.polyfit(x, y, 3)
                p1 = np.poly1d(f1)
                x_min = int(np.min(x))
                x_max = int(np.max(x))
                x_fit = []
                for i in range(x_min, x_max + 1):
                    x_fit.append(i)
                y_fit = p1(x_fit)
            except Warning as e:
                x_fit = x
                y_fit = y
            finally:
                return zip(x_fit, y_fit)

    def get_lane_mask(self, binary_seg_ret, instance_seg_ret):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, instance_seg_ret)

        num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1.5)

        # 聚类簇超过八个则选择其中类内样本最多的八个聚类簇保留下来
        if num_clusters > 8:
            cluster_sample_nums = []
            for i in range(num_clusters):
                cluster_sample_nums.append(len(np.where(labels == i)[0]))
            sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
            cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
        else:
            cluster_index = range(num_clusters)

        mask_image = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1], 3], dtype=np.uint8)

        for index, i in enumerate(cluster_index):
            idx = np.where(labels == i)
            coord = lane_coordinate[idx]
            # coord = self._thresh_coord(coord)
            coord = np.flip(coord, axis=1)
            # coord = (coord[:, 0], coord[:, 1])
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))
            coord = np.array([coord])
            cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=2)
            # mask_image[coord] = color

        return mask_image

    def get_lane_mask_iou(self, binary_seg_ret, binary_score_images, gt_seg_ret):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        mask_img: points and fitted curve
        mask_image: lane centre
        """

        
        #import pdb;pdb.set_trace()  
           
        # mask_img = np.zeros(shape=[bpe[0], binary_seg_ret.shape[1]], dtype=np.uint8)
        if np.sum(binary_seg_ret) <10:
            return None
        # gt_seg_ret = cv2.resize(gt_seg_ret.astype(np.uint8), (size[1],size[0]), interpolation=cv2.INTER_NEAREST)

        # # compute iou
        iou_mask_image = binary_seg_ret.reshape((-1))
        #iou_mask_image = (iou_mask_image>0)*1.0
        index = np.where(gt_seg_ret>0)
        _gt_seg_ret = gt_seg_ret.copy()
        #_gt_seg_ret[index] = 1
        _gt_seg_ret = _gt_seg_ret.reshape((-1))
        cm = confusion_matrix(_gt_seg_ret, iou_mask_image)
        sz = cm.shape[0]
        single_class_iou = []
        for i in range(sz):
            tmp_iou = 1.0*cm[i,i]/(np.sum(cm[i,:])+np.sum(cm[:,i])-cm[i,i])
            single_class_iou.append(tmp_iou)
            #print('seg_class {:d} iou {:.3f}'.format(i,tmp_iou))
        #import pdb;pdb.set_trace() 
        return single_class_iou   

    def get_lane_mask_boom(self, binary_seg_ret, binary_score_images, gt_seg_ret, gt_lane_ret, reg_image):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        mask_img: points and fitted curve
        mask_image: lane centre
        """
        size=[288,800]

        
        #import pdb;pdb.set_trace()  
           
        mask_img = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1],3], dtype=np.uint8)
        mask_png = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1]], dtype=np.uint8)
        if np.sum(binary_seg_ret) <10:
            return mask_img,None,None
        # gt_seg_ret = cv2.resize(gt_seg_ret.astype(np.uint8), (size[1],size[0]), interpolation=cv2.INTER_NEAREST)

        # # compute iou
        #iou_mask_image = binary_seg_ret.reshape((-1))
        #iou_mask_image = (iou_mask_image>0)*1.0
        #index = np.where(gt_seg_ret>0)
        #_gt_seg_ret = gt_seg_ret.copy()
        #_gt_seg_ret[index] = 1
        #_gt_seg_ret = _gt_seg_ret.reshape((-1))
        #cm = confusion_matrix(_gt_seg_ret, iou_mask_image)
        #sz = cm.shape[0]
        #single_class_iou = []
        single_class_iou = [0,0]
        #for i in range(sz):
        #    tmp_iou = 1.0*cm[i,i]/(np.sum(cm[i,:])+np.sum(cm[:,i])-cm[i,i])
        #    single_class_iou.append(tmp_iou)
            #print('seg_class {:d} iou {:.3f}'.format(i,tmp_iou))
        #import pdb;pdb.set_trace()

        mask_image = mask_img.copy()
        postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
        binary_seg_ret = postprocessor.postprocess(binary_seg_ret)
        # if 1:
        #     mm = gt_seg_ret==1
        #     binary_seg_ret *= mm
        if 0:
            # ind = gt_seg_ret==1
            ind = binary_score_images[:,:,1]>0.8
            binary_seg_ret *= ind

        Use_gt = 2 
        mask_mp = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1]], dtype=np.uint8)
        label_ind = []
        label_x = []
        size_y = range(288)
        wide = range(size[1])
        if Use_gt==0:
            binary_seg_ret = (binary_seg_ret>0)*1.0
            cluster_index = np.unique(gt_seg_ret)
            cluster_index = [tmp for tmp in cluster_index if tmp != 0]
            gt_seg_ret = gt_seg_ret#*binary_seg_ret
            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(gt_seg_ret == i)
                coord = idx
                # import pdb;pdb.set_trace()
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[1]))
                mask_mp[coord] = i+1
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                mask_image[coord]=color
        elif Use_gt ==1:
            lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, binary_score_images)

            num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1.5)

            # 聚类簇超过八个则选择其中类内样本最多的八个聚类簇保留下来
            if num_clusters > 3:
                cluster_sample_nums = []
                for i in range(num_clusters):
                    cluster_sample_nums.append(len(np.where(labels == i)[0]))
                sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
                cluster_index = np.array(range(num_clusters))[sort_idx[0:3]]
            else:
                cluster_index = range(num_clusters)

            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(labels == i)
                coord = lane_coordinate[idx]
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[:, 1]))
                coord_ = (coord[:, 0], coord[:, 1])
                mask_mp[coord_] = i+1
                coord = np.flip(coord, axis=1)
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                coord = np.array([coord])
                cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=2)
        else:
            num_cluster = np.max(binary_seg_ret)

            assert num_cluster<=3

            cluster_index = np.unique(binary_seg_ret)
            cluster_index = [tmp for tmp in cluster_index if tmp != 0]
            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(binary_seg_ret == i)
                coord = idx
                # import pdb;pdb.set_trace()
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[1]))
                mask_mp[coord] = i+1
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                mask_image[coord]=color
                
        cv2.imwrite('2.jpg',mask_image)
        # cv2.imwrite('1.png',binary_seg_ret_*255)
        # cv2.imwrite('1.png',gt_lane_ret*255)
        # cv2.imwrite('1.png',reg_image_l*10)
        # cv2.imwrite('2.png',mask_mp*55)
        import pdb;pdb.set_trace()

        points_all=[]
        params_l = []
        params_r = []
        draw_img = mask_img.copy()

        y_lst_r_pre = []
        x_lst_pre =[]
        label_x_ = np.argsort(label_x)
        for k in range(len(label_ind)):
                flag_pass = [5]
                mm = mask_mp==label_ind[label_x_[k]]
                binary_seg_ret_ = mm#*binary_seg_ret
                #import pdb;pdb.set_trace()
                pass_flag = False
                if np.sum(binary_seg_ret_)<2888:
                    pass_flag = True
                #import pdb;pdb.set_trace()
                #binary_seg_ret_ = binary_seg_ret_[size_y][:,wide]
                reg_image_l = reg_image[:,:,0]*binary_seg_ret_
                reg_image_r = reg_image[:,:,1]*binary_seg_ret_
                x_lst_r = []
                x_lst = []
                x_lst_l = [] 
                y_lst_l = [] 
                y_lst_r = [] 
                min_x = 288
                max_x = 0
                for i in range(size[0]):
                    mm = (reg_image_r[i,:]>0)*(reg_image_l[i,:]>0)
                    if np.sum(mm)<=0:
                        continue
                    min_x = min(i,min_x)
                    max_x = max(i,max_x)
                    if np.sum(binary_seg_ret_[i])<np.mean(flag_pass[-3:]):
                        continue
                    
                    point_x_l = (wide - reg_image_l[i,:]*8)*mm
                    point_x_r = (wide + reg_image_r[i,:]*8)*mm
                    # mask = np.where(point_x_l>0)                
                    # point_x_l = point_x_l[mask]
                    # # mask = np.where(point_x_r>0)
                    # point_x_r = point_x_r[mask]
                    # if k>0 and i>180:
                    #     import pdb;pdb.set_trace()
                    
                    # mu, std = norm.fit(point_x_l)
                    tmp = []
                    tmp_2 =[]
                    ind_l = []
                    for ja in range(len(point_x_l)):
                        value = point_x_l[ja]
                        if value <=0:
                            continue
                        if gt_lane_ret[i,int(min(value,size[1]-1))]>0:
                            tmp.append(min(value,size[1]-1))
                            ind_l.append(i)
                            tmp_2.append(wide[ja])
                    point_x_l = tmp
                    # if k>0 and i>180:
                    #     import pdb;pdb.set_trace()
                    # mu, std = norm.fit(point_x_r)
                    tmp = []
                    ind_r = []
                    tmp_3 =[]
                    for jb in range(len(point_x_r)):
                        value = point_x_r[jb]
                        if value <=0:
                            continue
                        if gt_lane_ret[i,int(min(value,size[1]-1))]>0:
                            tmp.append(min(value,size[1]-1))
                            ind_r.append(i)
                            tmp_3.append(wide[jb])
                    point_x_r = tmp
                    # if k>0 and i>220:
                    #     import pdb;pdb.set_trace()

                    point = [(i,value) for value in point_x_l]
                    # import pdb;pdb.set_trace()
                    x_lst.extend(ind_l)
                    x_lst_r.extend(ind_r)
                    # import pdb;pdb.set_trace()
                    y_lst_l.extend(point_x_l)
                    y_lst_r.extend(point_x_r)
                    # x_lst.append(i)
                    # y_lst_l.append(np.mean(point_x_l))
                    # y_lst_r.append(np.mean(point_x_r))
                    # if len(point_x_l)!=len(point_x_r):
                    #import pdb;pdb.set_trace()
                    # if point_x_l.any
                    # points.append(point)
                    point = [(i,value) for value in point_x_r]
                    # points.append(point)
                    mask_img[i][[int(jc) for jc in tmp_2]] = (int(self._color_map[2][0]),
                                              int(self._color_map[2][1]),
                                              int(self._color_map[2][2]))
                    # mask_img[i][[int(jc) for jc in tmp_3]] = (int(self._color_map[1][0]),
                    #                           int(self._color_map[1][1]),
                    #                           int(self._color_map[1][2]))
                    flag_pass.append(np.sum(binary_seg_ret_[i]))
                    # if len(flag_pass)>28:
                    #     if np.max(flag_pass)<15:
                    #         pass_flag = True
                    #         # import pdb;pdb.set_trace()
                    #         break
                    #     del flag_pass[0]
                    # mask_png[i][point_x_r] = 100
                    # mask_png[i][point_x_l] = 1
                # skeleton = morphology.skeletonize(mask_png)
                # skeleton = skeleton.astype(np.uint8)
                # skel_inds = np.where(skeleton > 0)
                # mask_png[...] = 0
                # mask_png[skel_inds] = 100

                if pass_flag:
                    x_lst_l = x_lst_pre
                    y_lst_l = y_lst_r_pre
                    y_lst_r_pre = []
                    x_lst_pre = []
                else:
                    x_lst_l = x_lst_pre + x_lst
                    y_lst_l = y_lst_r_pre + y_lst_l
                    y_lst_r_pre = y_lst_r
                    x_lst_pre = x_lst_r
                if len(set(x_lst_l))<3:
                    # import pdb;pdb.set_trace()
                    continue
                if len(x_lst_l)!=len(y_lst_l):
                    import pdb;pdb.set_trace()
                parameter_l = self.curve_fit(np.array(x_lst_l), np.array(y_lst_l), self.f_2)
                draw_img, points = self.draw_curve(mask_img, parameter_l, min(np.min(x_lst_l),min_x),max(np.max(x_lst_l),max_x))
                if len(points)<4:
                    # import pdb;pdb.set_trace()
                    continue

                points_all.append(points)
                params_l.append(parameter_l)
                # cv2.imwrite('1.jpg',draw_img)
                # import pdb;pdb.set_trace()
                if k ==(len(label_ind)-1) and pass_flag != True and len(set(x_lst_r))>2:
                # try:
                    parameter_r = self.curve_fit(np.array(x_lst_r), np.array(y_lst_r), self.f_2)
                    draw_img, points = self.draw_curve(mask_img, parameter_r, min_x, max_x)
                # else:
                    points_all.append(points)
                    params_r.append(parameter_r)


        cv2.imwrite('1.jpg',draw_img)
        params = params_l + params_r
        #print(len(params))
        import pdb;pdb.set_trace()
        #if len(params)<=len(cluster_index) or len(params)<2:
        #if len(params)<2:
        #    import pdb;pdb.set_trace()
        return draw_img, mask_image, single_class_iou, params, points_all

    def get_lane_mask_new(self, binary_seg_ret, instance_seg_ret, binary_score_images, gt_seg_ret, raw_image, reg_image):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        mask_img: points and fitted curve
        mask_image: lane centre
        """
        size=[288,800]

        
        #import pdb;pdb.set_trace()  
           
        mask_img = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1],3], dtype=np.uint8)
        mask_png = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1]], dtype=np.uint8)
        if np.sum(binary_seg_ret) <10:
            return mask_img,None,None
        # gt_seg_ret = cv2.resize(gt_seg_ret.astype(np.uint8), (size[1],size[0]), interpolation=cv2.INTER_NEAREST)

        # # compute iou
        #iou_mask_image = binary_seg_ret.reshape((-1))
        #iou_mask_image = (iou_mask_image>0)*1.0
        #index = np.where(gt_seg_ret>0)
        #_gt_seg_ret = gt_seg_ret.copy()
        #_gt_seg_ret[index] = 1
        #_gt_seg_ret = _gt_seg_ret.reshape((-1))
        #cm = confusion_matrix(_gt_seg_ret, iou_mask_image)
        #sz = cm.shape[0]
        #single_class_iou = []
        single_class_iou = [0,0]
        #for i in range(sz):
        #    tmp_iou = 1.0*cm[i,i]/(np.sum(cm[i,:])+np.sum(cm[:,i])-cm[i,i])
        #    single_class_iou.append(tmp_iou)
            #print('seg_class {:d} iou {:.3f}'.format(i,tmp_iou))
        #import pdb;pdb.set_trace()

        mask_image = mask_img.copy()
        mid = mask_img.copy()
        # postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
        # binary_seg_ret = postprocessor.postprocess(binary_seg_ret)
        # if 1:
        #     mm = gt_seg_ret==1
        #     binary_seg_ret *= mm
        if 0:
            # ind = gt_seg_ret==1
            ind = binary_score_images[:,:,1]>0.8
            binary_seg_ret *= ind

        Use_gt = 3 
        mask_mp = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1]], dtype=np.uint8)
        label_ind = []
        label_x = []
        size_y = range(288)
        wide = range(size[1])
        if Use_gt==0:
            binary_seg_ret = (binary_seg_ret>0)*1.0
            cluster_index = np.unique(gt_seg_ret)
            cluster_index = [tmp for tmp in cluster_index if tmp != 0]
            gt_seg_ret = gt_seg_ret#*binary_seg_ret
            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(gt_seg_ret == i)
                coord = idx
                # import pdb;pdb.set_trace()
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[1]))
                mask_mp[coord] = i+1
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                mask_image[coord]=color
        elif Use_gt ==1:
            lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, binary_score_images)

            num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1.5)

            # 聚类簇超过八个则选择其中类内样本最多的八个聚类簇保留下来
            if num_clusters > 3:
                cluster_sample_nums = []
                for i in range(num_clusters):
                    cluster_sample_nums.append(len(np.where(labels == i)[0]))
                sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
                cluster_index = np.array(range(num_clusters))[sort_idx[0:3]]
            else:
                cluster_index = range(num_clusters)

            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(labels == i)
                coord = lane_coordinate[idx]
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[:, 1]))
                coord_ = (coord[:, 0], coord[:, 1])
                mask_mp[coord_] = i+1
                coord = np.flip(coord, axis=1)
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                coord = np.array([coord])
                cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=2)
        elif Use_gt ==2:
            binary_seg_ret = binary_seg_ret>0.2
            instance_seg_ret = binary_seg_ret*gt_seg_ret
            num_cluster = np.max(instance_seg_ret)

            assert num_cluster<=3

            cluster_index = np.unique(instance_seg_ret)
            cluster_index = [tmp for tmp in cluster_index if tmp != 0]
            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(instance_seg_ret == i)
                coord = idx
                # import pdb;pdb.set_trace()
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[1]))
                mask_mp[coord] = i+1
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                mask_image[coord]=color
        else:
            binary_seg_ret = binary_seg_ret>0.2
            mid[:,:,1] = binary_seg_ret * 255
            mask_tmp = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
            _, mask_tmp = cv2.threshold(mask_tmp, 1, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i in range(len(contours)):
                # label_ind.append(i+1)
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area >10:
                    color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                    # cv2.drawContours(mask_mp,contours,i,i,3)
                    cv2.drawContours(mask_image,contours,i,color,3)

        cv2.imwrite('png/2.jpg',mask_image)
        cv2.imwrite('png/2.png',binary_score_images*255)
        cv2.imwrite('png/1.png',binary_seg_ret*255)
        cv2.imwrite('png/3.png',reg_image[:,:,0]*10)
        cv2.imwrite('png/0.png',instance_seg_ret*55)
        import pdb;pdb.set_trace()

        points_all=[]
        params_l = []
        params_r = []
        draw_img = mask_img.copy()

        y_lst_r_pre = []
        x_lst_pre =[]
        label_x_ = np.argsort(label_x)
        for k in range(len(label_ind)):
                flag_pass = [5]
                mm = mask_mp==label_ind[label_x_[k]]
                binary_seg_ret_ = mm#*binary_seg_ret
                #import pdb;pdb.set_trace()
                pass_flag = False
                if np.sum(binary_seg_ret_)<888:
                    pass_flag = True
                #import pdb;pdb.set_trace()
                #binary_seg_ret_ = binary_seg_ret_[size_y][:,wide]
                reg_image_l = reg_image[:,:,0]*binary_seg_ret_[96:,:]
                reg_image_r = reg_image[:,:,1]*binary_seg_ret_[96:,:]
                x_lst_r = []
                x_lst = []
                x_lst_l = [] 
                y_lst_l = [] 
                y_lst_r = [] 
                min_x = 288
                max_x = 0
                for i in range(size[0]):
                    if i<96:
                        continue    
                    mm = (reg_image_r[i-96,:]>0)*(reg_image_l[i-96,:]>0)
                    if np.sum(mm)<=0:
                        continue
                    min_x = min(i,min_x)
                    max_x = max(i,max_x)
                    if np.sum(binary_seg_ret_[i])<np.mean(flag_pass[-3:]):
                        continue
                    
                    point_x_l = (wide - reg_image_l[i-96,:]*8)*mm
                    point_x_r = (wide + reg_image_r[i-96,:]*8)*mm
                    mask = np.where(point_x_l>0)                
                    point_x_l = point_x_l[mask]
                    # mask = np.where(point_x_r>0)
                    point_x_r = point_x_r[mask]
                    if k==10 and i>220:
                        import pdb;pdb.set_trace()
                    
                    mu, std = norm.fit(point_x_l)
                    tmp = []
                    ind_l = []
                    for value in point_x_l:
                        if value>=(mu-std) and value<=(mu+std) and binary_seg_ret[i,int(max(min(value,size[1]-1),0))]==0:
                            tmp.append(max(min(value,size[1]-1),0))
                            ind_l.append(i)
                    point_x_l = tmp
                    mu, std = norm.fit(point_x_r)
                    tmp = []
                    ind_r = []
                    for value in point_x_r:
                        if value>=(mu-std) and value<=(mu+std) and binary_seg_ret[i,int(min(value,size[1]-1))]==0:
                            tmp.append(min(value,size[1]-1))
                            ind_r.append(i)
                    point_x_r = tmp
                    if k==10 and i>220:
                        import pdb;pdb.set_trace()

                    point = [(i,value) for value in point_x_l]
                    # import pdb;pdb.set_trace()
                    x_lst.extend(ind_l)
                    x_lst_r.extend(ind_r)
                    # import pdb;pdb.set_trace()
                    y_lst_l.extend(point_x_l)
                    y_lst_r.extend(point_x_r)
                    # x_lst.append(i)
                    # y_lst_l.append(np.mean(point_x_l))
                    # y_lst_r.append(np.mean(point_x_r))
                    # if len(point_x_l)!=len(point_x_r):
                    #import pdb;pdb.set_trace()
                    # if point_x_l.any
                    # points.append(point)
                    point = [(i,value) for value in point_x_r]
                    # points.append(point)
                    mask_img[i][[int(jb) for jb in point_x_l]] = (int(self._color_map[2][0]),
                                              int(self._color_map[2][1]),
                                              int(self._color_map[2][2]))
                    mask_img[i][[int(jb) for jb in point_x_r]] = (int(self._color_map[1][0]),
                                              int(self._color_map[1][1]),
                                              int(self._color_map[1][2]))
                    flag_pass.append(np.sum(binary_seg_ret_[i]))
                    # if len(flag_pass)>28:
                    #     if np.max(flag_pass)<15:
                    #         pass_flag = True
                    #         # import pdb;pdb.set_trace()
                    #         break
                    #     del flag_pass[0]
                    # mask_png[i][point_x_r] = 100
                    # mask_png[i][point_x_l] = 1
                # skeleton = morphology.skeletonize(mask_png)
                # skeleton = skeleton.astype(np.uint8)
                # skel_inds = np.where(skeleton > 0)
                # mask_png[...] = 0
                # mask_png[skel_inds] = 100

                if pass_flag:
                    x_lst_l = x_lst_pre
                    y_lst_l = y_lst_r_pre
                    y_lst_r_pre = []
                    x_lst_pre = []
                else:
                    x_lst_l = x_lst_pre + x_lst
                    y_lst_l = y_lst_r_pre + y_lst_l
                    y_lst_r_pre = y_lst_r
                    x_lst_pre = x_lst_r
                if len(set(x_lst_l))<3:
                    # import pdb;pdb.set_trace()
                    continue
                if len(x_lst_l)!=len(y_lst_l):
                    import pdb;pdb.set_trace()
                parameter_l = self.curve_fit(np.array(x_lst_l), np.array(y_lst_l), self.f_2)
                draw_img, points = self.draw_curve(mask_img, parameter_l, min(np.min(x_lst_l),min_x),max(np.max(x_lst_l),max_x))
                if len(points)<4:
                    # import pdb;pdb.set_trace()
                    continue

                points_all.append(points)
                params_l.append(parameter_l)
                #cv2.imwrite('1.jpg',draw_img)
                #import pdb;pdb.set_trace()
                if k ==(len(label_ind)-1) and pass_flag != True and len(set(x_lst_r))>2:
                # try:
                    parameter_r = self.curve_fit(np.array(x_lst_r), np.array(y_lst_r), self.f_2)
                    draw_img, points = self.draw_curve(mask_img, parameter_r, min_x, max_x)
                # else:
                    points_all.append(points)
                    params_r.append(parameter_r)


        # cv2.imwrite('1.jpg',draw_img)
        params = params_l + params_r
        #print(len(params))
        # import pdb;pdb.set_trace()
        #if len(params)<=len(cluster_index) or len(params)<2:
        #if len(params)<2:
        #    import pdb;pdb.set_trace()
        return draw_img, mask_image, single_class_iou, params, points_all

    def get_lane_mask_centre(self, binary_seg_ret, gt_seg_ret, raw_image, reg_image):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        mask_img: points and fitted curve
        mask_image: lane centre
        """
        size=[288,800]

        
        # import pdb;pdb.set_trace()  
           
        mask_img = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1],3], dtype=np.uint8)
        mask_png = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1]], dtype=np.uint8)
        # if np.sum(binary_seg_ret) <10:
        #     return mask_img,None,None
        # gt_seg_ret = cv2.resize(gt_seg_ret.astype(np.uint8), (size[1],size[0]), interpolation=cv2.INTER_NEAREST)

        # # compute iou
        #iou_mask_image = binary_seg_ret.reshape((-1))
        #iou_mask_image = (iou_mask_image>0)*1.0
        #index = np.where(gt_seg_ret>0)
        #_gt_seg_ret = gt_seg_ret.copy()
        #_gt_seg_ret[index] = 1
        #_gt_seg_ret = _gt_seg_ret.reshape((-1))
        #cm = confusion_matrix(_gt_seg_ret, iou_mask_image)
        #sz = cm.shape[0]
        #single_class_iou = []
        single_class_iou = [0,0]
        #for i in range(sz):
        #    tmp_iou = 1.0*cm[i,i]/(np.sum(cm[i,:])+np.sum(cm[:,i])-cm[i,i])
        #    single_class_iou.append(tmp_iou)
            #print('seg_class {:d} iou {:.3f}'.format(i,tmp_iou))
        #import pdb;pdb.set_trace()

        mask_image = mask_img.copy()
        mid = mask_img.copy()
        # postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
        # binary_seg_ret = postprocessor.postprocess(binary_seg_ret)
        # if 1:
        #     mm = gt_seg_ret==1
        #     binary_seg_ret *= mm
        if 0:
            # ind = gt_seg_ret==1
            ind = binary_score_images[:,:,1]>0.8
            binary_seg_ret *= ind

        Use_gt = 2 
        mask_mp = np.zeros(shape=[binary_seg_ret.shape[0], binary_seg_ret.shape[1]], dtype=np.uint8)
        label_ind = []
        label_x = []
        size_y = range(288)
        wide = range(size[1])
        if Use_gt==0:
            binary_seg_ret = (binary_seg_ret>0)*1.0
            cluster_index = np.unique(gt_seg_ret)
            cluster_index = [tmp for tmp in cluster_index if tmp != 0]
            gt_seg_ret = gt_seg_ret#*binary_seg_ret
            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(gt_seg_ret == i)
                coord = idx
                # import pdb;pdb.set_trace()
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[1]))
                mask_mp[coord] = i+1
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                mask_image[coord]=color
        elif Use_gt ==1:
            lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, binary_score_images)

            num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=1.5)

            # 聚类簇超过八个则选择其中类内样本最多的八个聚类簇保留下来
            if num_clusters > 3:
                cluster_sample_nums = []
                for i in range(num_clusters):
                    cluster_sample_nums.append(len(np.where(labels == i)[0]))
                sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
                cluster_index = np.array(range(num_clusters))[sort_idx[0:3]]
            else:
                cluster_index = range(num_clusters)

            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(labels == i)
                coord = lane_coordinate[idx]
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[:, 1]))
                coord_ = (coord[:, 0], coord[:, 1])
                mask_mp[coord_] = i+1
                coord = np.flip(coord, axis=1)
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                coord = np.array([coord])
                cv2.polylines(img=mask_image, pts=coord, isClosed=False, color=color, thickness=2)
        elif Use_gt ==2:
            binary_seg_ret = binary_seg_ret>0.2
            instance_seg_ret = binary_seg_ret*gt_seg_ret
            num_cluster = np.max(instance_seg_ret)

            assert num_cluster<=3

            cluster_index = np.unique(instance_seg_ret)
            cluster_index = [tmp for tmp in cluster_index if tmp != 0]
            for index, i in enumerate(cluster_index):
                label_ind.append(i+1)
                idx = np.where(instance_seg_ret == i)
                coord = idx
                # import pdb;pdb.set_trace()
                # coord = self._thresh_coord(coord)
                label_x.append(np.mean(coord[1]))
                mask_mp[coord] = i+1
                color = (int(self._color_map[(index+2)%6][0]),
                         int(self._color_map[(index+2)%6][1]),
                         int(self._color_map[(index+2)%6][2]))
                mask_image[coord]=color
        else:
            #import pdb;pdb.set_trace()
            binary_seg_ret = binary_seg_ret>0.8
            #import pdb;pdb.set_trace()
            mid[:,:,1] = binary_seg_ret * 255
            mask_tmp = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
            _, mask_tmp = cv2.threshold(mask_tmp, 1, 255, cv2.THRESH_BINARY)
            # cv2.imwrite('png/2.jpg',mask_tmp)
            # import pdb;pdb.set_trace()
            _, contours, hierarchy = cv2.findContours(mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area >1288:
                    label_ind.append(i+1)
                    label_x.append(np.mean(cnt[:,:,0]))
                    # import pdb;pdb.set_trace()
                    color = (int(self._color_map[(i+2)%6][0]),
                         int(self._color_map[(i+2)%6][1]),
                         int(self._color_map[(i+2)%6][2]))
                    cv2.drawContours(mask_mp,contours,i,i+1,-1)
                    # cv2.imwrite('png/4.png',mask_mp*55)
                    # import pdb;pdb.set_trace()
                    cv2.drawContours(mask_image,contours,i,color,-1)

        # cv2.imwrite('png/3.jpg',mask_image)
        # cv2.imwrite('png/4.png',mask_mp*55)
        # cv2.imwrite('png/2.png',reg_image[:,:,1])
        # cv2.imwrite('png/1.png',binary_seg_ret*255)
        # cv2.imwrite('png/3.png',reg_image[:,:,0]*binary_seg_ret[96:]*800)
        # cv2.imwrite('png/0.png',instance_seg_ret*55)
        # import pdb;pdb.set_trace()
        # reg_image = reg_image*55

        points_all=[]
        params_l = []
        params_r = []
        draw_img = mask_img.copy()

        y_lst_r_pre = []
        x_lst_pre =[]
        label_x_ = np.argsort(label_x)
        for k in range(len(label_ind)):
                flag_pass = [5]
                mm = mask_mp==label_ind[label_x_[k]]
                binary_seg_ret_ = mm#*binary_seg_ret
                #import pdb;pdb.set_trace()
                pass_flag = False
                if np.sum(binary_seg_ret_)<10:
                    pass_flag = True
                # import pdb;pdb.set_trace()
                #binary_seg_ret_ = binary_seg_ret_[size_y][:,wide]
                reg_image_l = reg_image[:,:,0]*binary_seg_ret_[96:,:]
                reg_image_r = reg_image[:,:,1]*binary_seg_ret_[96:,:]
                x_lst_r = []
                x_lst = []
                x_lst_l = [] 
                y_lst_l = [] 
                y_lst_r = [] 
                min_x = 288
                max_x = 0
                for i in range(size[0]):
                    if i<96:
                        continue    
                    mm = (reg_image_r[i-96,:]>0)*(reg_image_l[i-96,:]>0)
                    if np.sum(mm)<=0:
                        continue
                    min_x = min(i,min_x)
                    max_x = max(i,max_x)
                    if np.sum(binary_seg_ret_[i])<np.mean(flag_pass[-3:]):
                        continue
                    
                    point_x_l = (wide - reg_image_l[i-96,:]*800)*mm
                    point_x_r = (wide + reg_image_r[i-96,:]*800)*mm
                    mask = np.where(point_x_l>0)                
                    point_x_l = point_x_l[mask]
                    # mask = np.where(point_x_r>0)
                    point_x_r = point_x_r[mask]
                    if k==10 and i>220:
                        import pdb;pdb.set_trace()
                    
                    mu, std = norm.fit(point_x_l)
                    tmp = []
                    ind_l = []
                    for value in point_x_l:
                        if value>=(mu-std) and value<=(mu+std) and binary_seg_ret[i,int(max(min(value,size[1]-1),0))]==0:
                            tmp.append(max(min(value,size[1]-1),0))
                            ind_l.append(i)
                    point_x_l = tmp
                    mu, std = norm.fit(point_x_r)
                    tmp = []
                    ind_r = []
                    for value in point_x_r:
                        if value>=(mu-std) and value<=(mu+std) and binary_seg_ret[i,int(min(value,size[1]-1))]==0:
                            tmp.append(min(value,size[1]-1))
                            ind_r.append(i)
                    point_x_r = tmp
                    if k==10 and i>220:
                        import pdb;pdb.set_trace()

                    point = [(i,value) for value in point_x_l]
                    # import pdb;pdb.set_trace()
                    x_lst.extend(ind_l)
                    x_lst_r.extend(ind_r)
                    # import pdb;pdb.set_trace()
                    y_lst_l.extend(point_x_l)
                    y_lst_r.extend(point_x_r)
                    # x_lst.append(i)
                    # y_lst_l.append(np.mean(point_x_l))
                    # y_lst_r.append(np.mean(point_x_r))
                    # if len(point_x_l)!=len(point_x_r):
                    #import pdb;pdb.set_trace()
                    # if point_x_l.any
                    # points.append(point)
                    point = [(i,value) for value in point_x_r]
                    # points.append(point)
                    mask_img[i][[int(jb) for jb in point_x_l]] = (int(self._color_map[2][0]),
                                              int(self._color_map[2][1]),
                                              int(self._color_map[2][2]))
                    mask_img[i][[int(jb) for jb in point_x_r]] = (int(self._color_map[1][0]),
                                              int(self._color_map[1][1]),
                                              int(self._color_map[1][2]))
                    flag_pass.append(np.sum(binary_seg_ret_[i]))
                    # if len(flag_pass)>28:
                    #     if np.max(flag_pass)<15:
                    #         pass_flag = True
                    #         # import pdb;pdb.set_trace()
                    #         break
                    #     del flag_pass[0]
                    # mask_png[i][point_x_r] = 100
                    # mask_png[i][point_x_l] = 1
                # skeleton = morphology.skeletonize(mask_png)
                # skeleton = skeleton.astype(np.uint8)
                # skel_inds = np.where(skeleton > 0)
                # mask_png[...] = 0
                # mask_png[skel_inds] = 100

                if pass_flag:
                    x_lst_l = x_lst_pre
                    y_lst_l = y_lst_r_pre
                    y_lst_r_pre = []
                    x_lst_pre = []
                else:
                    x_lst_l = x_lst_pre + x_lst
                    y_lst_l = y_lst_r_pre + y_lst_l
                    y_lst_r_pre = y_lst_r
                    x_lst_pre = x_lst_r
                if len(set(x_lst_l))<3:
                    # import pdb;pdb.set_trace()
                    continue
                if len(x_lst_l)!=len(y_lst_l):
                    import pdb;pdb.set_trace()
                parameter_l = self.curve_fit(np.array(x_lst_l), np.array(y_lst_l), self.f_2)
                draw_img, points = self.draw_curve(mask_img, parameter_l, min(np.min(x_lst_l),min_x),max(np.max(x_lst_l),max_x))
                if len(points)<4:
                    # import pdb;pdb.set_trace()
                    continue

                points_all.append(points)
                params_l.append(parameter_l)
                if k ==(len(label_ind)-1) and pass_flag != True and len(set(x_lst_r))>2:
                # try:
                    parameter_r = self.curve_fit(np.array(x_lst_r), np.array(y_lst_r), self.f_2)
                    draw_img, points = self.draw_curve(mask_img, parameter_r, min_x, max_x)
                # else:
                    points_all.append(points)
                    params_r.append(parameter_r)
        # cv2.imwrite('1.jpg',draw_img)
        # import pdb;pdb.set_trace()

        params = params_l + params_r
        #print(len(params))
        #if len(params)<=len(cluster_index) or len(params)<2:
        # if len(params)<2:
        #    import pdb;pdb.set_trace()
        return draw_img, mask_image, single_class_iou, params, points_all

    def draw_curve(self, img, curve, y_beg=96, y_end=288):
        if curve is None:
          return img
        assert len(curve) == 3
        # h = img.shape[0]
        # # import pdb;pdb.set_trace()
        # y_beg = h//3
        # y_end = h*5//6
        first = True
        def f_2(x, A, B, C):
          return A*x*x + B*x + C
        points=[]
        for m in range(18):
            if (float(590-m*20)-1)*288/590 >=(y_beg-16) and (float(590-m*20)-1)*288/590 <=(y_end+16):
                points.append(f_2((float(590-m*20)-1)*288/590, curve[0], curve[1], curve[2])*1640/800)
                points.append((int(590-m*20)-1))
        for y in range(y_beg, y_end):
          x = int(f_2(y, curve[0], curve[1], curve[2]))

          if first:
            first_pt = (x,y)
            first = False
            continue
          cv2.line(img, first_pt, (x,y), (0,255,0), 2)
          first_pt = (x,y)
        return img, points

    def f_2(self, x, A, B, C):
        return A*x*x + B*x + C

    def curve_fit(self, x_lst, y_lst, fun):
        parameter = optimize.curve_fit(fun, x_lst, y_lst, [1, 1, 1])
        return parameter[0]

if __name__ == '__main__':
    binary_seg_image = cv2.imread('binary_ret.png', cv2.IMREAD_GRAYSCALE)
    binary_seg_image[np.where(binary_seg_image == 255)] = 1
    instance_seg_image = cv2.imread('instance_ret.png', cv2.IMREAD_UNCHANGED)
    ele_mex = np.max(instance_seg_image, axis=(0, 1))
    for i in range(3):
        if ele_mex[i] == 0:
            scale = 1
        else:
            scale = 255 / ele_mex[i]
        instance_seg_image[:, :, i] *= int(scale)
    embedding_image = np.array(instance_seg_image, np.uint8)
    cluster = LaneNetCluster()
    mask_image = cluster.get_lane_mask(instance_seg_ret=instance_seg_image, binary_seg_ret=binary_seg_image)
    plt.figure('embedding')
    plt.imshow(embedding_image[:, :, (2, 1, 0)])
    plt.figure('mask_image')
    plt.imshow(mask_image[:, :, (2, 1, 0)])
    plt.show()
