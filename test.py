import torch
import os
import torchvision
import glob
from PIL import Image
import cv2
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"
# 参数设置


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, default=r"")
    parser.add_argument("--weights", type=str, default="", help="model path")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="test image size")
    opt = parser.parse_known_args()[0]
    return opt
# 测试图片


class Test_model():
    def __init__(self, opt):
        self.imgsz = opt.imgsz  # 测试图片尺寸
        self.img_dir = opt.test_dir  # 测试图片路径

        self.model = (torch.load(opt.weights)).to(device)  # 加载模型
        self.model.eval()
        self.class_name = []  # 类别信息

    def __call__(self):
        # 图像转换
        data_transorform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_list = glob.glob(self.img_dir+os.sep+"*.jpg")

        for imgpath in img_list:
            img = cv2.imread(imgpath)
            new_img = self.expend_img(img)  # 补边
            img = Image.fromarray(new_img)
            img = data_transorform(img)  # 转换
            img = torch.reshape(
                img, (-1, 3, self.imgsz, self.imgsz)).to(device)  # 维度转换[B,C,H,W]
            pred = self.model(img)
            _, pred = torch.max(pred, 1)
            outputs = self.class_name[pred]
            print("Image path:", imgpath, " pred:", outputs)

    # 补边为正方形
    def expend_img(self, img, fill_pix=122):
        '''
        :param img: 图片数据
        :param fill_pix: 填充像素，默认为灰色，自行更改
        :return:
        '''
        h, w = img.shape[:2]  # 获取图像的宽高
        if h >= w:  # 左右填充
            padd_width = int(h-w)//2
            padd_h, padd_b, padd_l, padd_r = 0, 0, padd_width, padd_width  # 获取上下左右四个方向需要填充的像素

        elif h < w:  # 上下填充
            padd_high = int(w-h)//2
            padd_h, padd_b, padd_l, padd_r = padd_high, padd_high, 0, 0

        new_img = cv2.copyMakeBorder(img, padd_h, padd_b, padd_l, padd_r, borderType=cv2.BORDER_CONSTANT,
                                     value=[fill_pix, fill_pix, fill_pix])
        return new_img


if __name__ == '__main__':
    opt = parser_opt()
    test_img = Test_model(opt)
    test_img()
