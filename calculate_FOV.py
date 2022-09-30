import math

#焦距：focal_distance (mm)
#图像宽度：img_weight (pix)
#像元尺寸: pixel_size (μm/pix)
def calculate(img_weight, focal_distance, pixel_size):

    weights = (img_weight * pixel_size) * 0.5
    distance = focal_distance * 1000

    return math.degrees(math.atan(weights / distance)) * 2

#焦距：focal_distance (mm)
#当前视场角：current_FOV
#1/3视场角：part_FOV

def calculate_part_FOV(focal_distance, current_FOV):
    distance = focal_distance * 1000
    width = math.tan(current_FOV / 360 * math.pi) * distance / 3
    part_FOV = math.degrees(math.atan(width / distance)) * 2
    return part_FOV

if __name__  == '__main__':
    img_weight = 1920
    focal_distance = 10
    pixel_size = 1.4
    c_FOV = calculate(img_weight, focal_distance, pixel_size)
    print(c_FOV)
    part_FOV = calculate_part_FOV(focal_distance, c_FOV)
    print(part_FOV)