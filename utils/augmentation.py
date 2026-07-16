import xml.etree.ElementTree as ET
import albumentations as A
import shutil
import cv2
import os


# 读取出图像中的目标框
def read_xml_annotation(xml_path):

    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):

        name = object.find('name').text
        bndbox = object.find('bndbox')  

        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        bndboxlist.append([xmin,ymin,xmax,ymax,name])

    return bndboxlist

def writeXml(xml_path, jpgname, width, height, channels, bboxes):
        with open(xml_path, "w") as xml:
            xml.write("<annotation>\n")
            xml.write("    <folder>JPEGImages</folder>\n")
            xml.write("    <filename>{}</filename>\n".format(jpgname))
            xml.write("    <path>./VOCdevkit/VOC2007/JPEGImages/{}</path>\n".format(jpgname))
            xml.write("    <source>\n")
            xml.write("        <database>Unknown</database>\n")
            xml.write("    </source>\n")
            xml.write("    <size>\n")
            xml.write("        <width>{}</width>\n".format(width))
            xml.write("        <height>{}</height>\n".format(height))
            xml.write("        <depth>{}</depth>\n".format(channels))
            xml.write("    </size>\n")
            xml.write("    <segmented>0</segmented>\n")
            if bboxes == []:
                pass
            else:
                for object in bboxes:
                    xmin = object[0]
                    ymin = object[1]
                    xmax = object[2]
                    ymax = object[3]
                    name = object[4]
                    xml.write("    <object>\n")
                    xml.write("        <name>{}</name>\n".format(name))
                    xml.write("        <pose>Unspecified</pose>\n")
                    xml.write("        <truncated>0</truncated>\n")
                    xml.write("        <difficult>0</difficult>\n")
                    xml.write("        <bndbox>\n")
                    xml.write("            <xmin>{}</xmin>\n".format(xmin))
                    xml.write("            <ymin>{}</ymin>\n".format(ymin))
                    xml.write("            <xmax>{}</xmax>\n".format(xmax))
                    xml.write("            <ymax>{}</ymax>\n".format(ymax))
                    xml.write("        </bndbox>\n")
                    xml.write("    </object>\n")
            xml.write("</annotation>\n")

def aug(img, num=1):

    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    xml_path = img[:-4].replace("JPEGOriginal", "AnnoOriginal")+".xml"
    jpegOpath = os.path.dirname(img)
    augpath = jpegOpath.replace("JPEGOriginal", "Augmentation")
    img_name = os.path.basename(img)
    width, height, channels = image.shape[1], image.shape[0], image.shape[2]
    bboxes = read_xml_annotation(xml_path)

    for i in range(num):
        transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=1),
            A.Perspective(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.05, p=0.2),
            A.RandomShadow(shadow_roi=(0,0,1,1), p=0.7),
            A.RandomSunFlare(flare_roi=(0,0,0.2,0.5), p=0.1),
            A.GaussNoise(p=0.5),
            A.ISONoise(p=0.5),
            A.GaussianBlur(p=0.5),
            # A.RandomCrop(width=width//1.5, height=height//1.5, p=1),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=20, min_visibility=0.4))

        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        format_ = img[-4:]
        count = "" if num==1 else i+1
        cv2.imwrite(os.path.join(augpath, "JPEGImages", img_name[:-4]+"_aug{}".format(count)+format_), transformed_image)
        imgaug = cv2.imread(os.path.join(augpath, "JPEGImages", img_name[:-4]+"_aug{}".format(count)+format_))
        writeXml(xml_path.replace(".xml", "_aug{}.xml".format(count)), img_name, imgaug.shape[1], imgaug.shape[0], imgaug.shape[2], transformed_bboxes)
        shutil.move(xml_path.replace(".xml", "_aug{}.xml".format(count)), os.path.join(augpath, "Annotations"))

    return transformed_image, transformed_bboxes


if __name__ == '__main__':
    
    # aug("../VOCdevkit/VOC2007/JPEGImages/EQ2_PRJ2141_EQ2_901.jpg", 1)
    

    img_dir = './logs/augmented_photos'
    # img_name = '0_C6D800_C6D800_C6D800_20200917191543'
    # img_name = '1_aug_after_vertical_flip'
    # img_name = '2_aug_after_horizontal_flip'
    # img_name = '3_aug_after_random_rotate_90'
    # img_name = '4_aug_after_shift_scale_rotate'
    # img_name = '5_aug_after_perspective'
    # img_name = '6_aug_after_hue_saturation_value'
    # img_name = '7_aug_after_random_brightness_contrast'
    # img_name = '8_aug_after_random_shadow'
    # img_name = '9_aug_after_random_sun_flare'
    # img_name = '10_aug_after_gauss_noise'
    img_name = '11_aug_after_ISO_noise'
    img_path = os.path.join(img_dir, img_name+'.jpg')
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    xml_path = img_path.replace('.jpg', '.xml')
    bboxes = read_xml_annotation(xml_path)
    
    transform = A.Compose([
        # A.VerticalFlip(p=1),
        # A.HorizontalFlip(p=1),
        # A.RandomRotate90(p=1),
        # A.ShiftScaleRotate(p=1),
        # A.Perspective(p=1),
        # A.HueSaturationValue(p=1),
        # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.05, p=1),
        # A.RandomShadow(shadow_roi=(0,0,1,1), p=1),
        # A.RandomSunFlare(flare_roi=(0,0,0.2,0.5), p=1),
        # A.GaussNoise(p=1),
        # A.ISONoise(p=1),
        A.GaussianBlur(p=1),
        ], bbox_params=A.BboxParams(format='pascal_voc', min_area=20, min_visibility=0.4))

    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    # aug_img_name = '1_aug_after_vertical_flip'
    # aug_img_name = '2_aug_after_horizontal_flip'
    # aug_img_name = '3_aug_after_random_rotate_90'
    # aug_img_name = '4_aug_after_shift_scale_rotate'
    # aug_img_name = '5_aug_after_perspective'
    # aug_img_name = '6_aug_after_hue_saturation_value'
    # aug_img_name = '7_aug_after_random_brightness_contrast'
    # aug_img_name = '8_aug_after_random_shadow'
    # aug_img_name = '9_aug_after_random_sun_flare'
    # aug_img_name = '10_aug_after_gauss_noise'
    # aug_img_name = '11_aug_after_ISO_noise'
    aug_img_name = '12_aug_after_gaussian_blur'
    aug_img_path = os.path.join(img_dir, aug_img_name+'.jpg')
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(aug_img_path, transformed_image)
    
    transformed_bboxes = transformed['bboxes']
    aug_xml_path = aug_img_path.replace('.jpg', '.xml')
    writeXml(aug_xml_path, img_name, transformed_image.shape[1], transformed_image.shape[0], transformed_image.shape[2], transformed_bboxes)